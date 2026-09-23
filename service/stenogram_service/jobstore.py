"""Очередь и состояние задач: интерфейс и реализация на SQLite.

Одна видеокарта — один воркер, поэтому redis не нужен: API и воркер — два процесса
на одном файле SQLite в режиме WAL. Замена на redis — это другой класс за тем же
`JobStore`, вызывающий код не меняется.

Каждый захват задачи выдаёт новую аренду (`lease`). Все записи воркера идут с ней
и возвращают False, если аренда уже не его: задачу, вернувшуюся в очередь после
«зависания», прежний владелец не перезапишет.
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"

# Код отказа задачи, воркер которой пропал без следа (SIGKILL от OOM, перезагрузка).
WORKER_LOST = "worker_lost"


@dataclass
class Job:
    id: str
    status: str
    source_uri: str
    params: dict[str, Any]
    created_at: float
    updated_at: float
    stage: str | None = None
    fraction: float | None = None
    sha256: str | None = None
    recognition_key: str | None = None
    result_key: str | None = None
    manifest: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    heartbeat_at: float | None = None
    attempts: int = 0  # сколько раз задачу захватывали
    lease: str | None = None  # аренда текущего захвата; у не-running задач None


@dataclass
class Recovered:
    job_id: str
    status: str  # queued — вернулась в очередь, failed — попыток больше нет


class JobStore(ABC):
    @abstractmethod
    def create(self, source_uri: str, params: dict[str, Any]) -> Job: ...

    @abstractmethod
    def get(self, job_id: str) -> Job | None: ...

    @abstractmethod
    def claim_next(self) -> Job | None:
        """Атомарно переводит старейшую `queued` в `running` с новой арендой. Пусто — None."""

    @abstractmethod
    def set_keys(self, job_id: str, lease: str, *, sha256: str, recognition_key: str, result_key: str) -> bool: ...

    @abstractmethod
    def progress(self, job_id: str, lease: str, stage: str, fraction: float | None = None) -> bool:
        """Этап и доля; заодно обновляет пульс."""

    @abstractmethod
    def heartbeat(self, job_id: str, lease: str) -> bool: ...

    @abstractmethod
    def complete(self, job_id: str, lease: str, manifest: dict[str, Any]) -> bool: ...

    @abstractmethod
    def fail(self, job_id: str, lease: str, code: str, message: str) -> bool: ...

    @abstractmethod
    def recover_stale(self, stale_after: float, max_attempts: int = 2) -> list[Recovered]:
        """`running` с пульсом старше `stale_after` секунд: в очередь, пока попыток
        меньше `max_attempts`, иначе в `failed` с кодом WORKER_LOST."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    id              TEXT NOT NULL UNIQUE,
    status          TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
    stage           TEXT,
    fraction        REAL,
    params          TEXT NOT NULL,
    source_uri      TEXT NOT NULL,
    sha256          TEXT,
    recognition_key TEXT,
    result_key      TEXT,
    manifest        TEXT,
    error_code      TEXT,
    error_message   TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    heartbeat_at    REAL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    lease           TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status_seq ON jobs (status, seq);
"""

_COLUMNS = (
    "id, status, source_uri, params, created_at, updated_at, stage, fraction, sha256, recognition_key, result_key, "
    "manifest, error_code, error_message, heartbeat_at, attempts, lease"
)


class SqliteJobStore(JobStore):
    """Соединение на операцию: так объект можно делить между потоками API, а два
    процесса на одном файле разводит блокировка SQLite, а не наш код."""

    def __init__(self, path: Path, *, busy_timeout: float = 10.0, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.busy_timeout = busy_timeout
        self.clock = clock
        self._init_schema()

    def _init_schema(self) -> None:
        """WAL и таблица — с повтором. Смена режима журнала на новом файле при
        одновременном старте нескольких процессов отдаёт «database is locked» сразу,
        обработчик ожидания SQLite её не ждёт (замер: 48 отказов из 160 стартов
        четырёх процессов). Повтор с паузой — до истечения того же busy_timeout."""
        deadline = time.monotonic() + self.busy_timeout
        while True:
            try:
                with self._connect() as db:
                    mode = db.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                    if mode.lower() != "wal":
                        # Например, сетевая ФС: без WAL API и воркер блокируют друг друга.
                        raise RuntimeError(f"SQLite не включил WAL (режим журнала: {mode})")
                    db.executescript(_SCHEMA)
                return
            except sqlite3.OperationalError as exc:
                busy = "locked" in str(exc) or "busy" in str(exc)
                if not busy or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01 + random.random() * 0.05)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # isolation_level=None — транзакции только явные: BEGIN IMMEDIATE в захвате.
        db = sqlite3.connect(self.path, timeout=self.busy_timeout, isolation_level=None)
        try:
            # Ожидание чужой блокировки для обычных операций. Смену режима журнала оно
            # не покрывает — для неё повтор в _init_schema.
            db.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout * 1000)}")
            db.execute("PRAGMA synchronous = NORMAL")
            yield db
        finally:
            db.close()

    @staticmethod
    def _job(row: tuple) -> Job:
        (job_id, status, source_uri, params, created, updated, stage, fraction, sha256, rkey, reskey, manifest, code, message, heartbeat, attempts, lease) = row
        return Job(
            id=job_id,
            status=status,
            source_uri=source_uri,
            params=json.loads(params),
            created_at=created,
            updated_at=updated,
            stage=stage,
            fraction=fraction,
            sha256=sha256,
            recognition_key=rkey,
            result_key=reskey,
            manifest=None if manifest is None else json.loads(manifest),
            error_code=code,
            error_message=message,
            heartbeat_at=heartbeat,
            attempts=attempts,
            lease=lease,
        )

    def create(self, source_uri: str, params: dict[str, Any]) -> Job:
        now = self.clock()
        job_id = str(uuid.uuid4())
        with self._connect() as db:
            db.execute(
                "INSERT INTO jobs (id, status, params, source_uri, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, QUEUED, json.dumps(params, ensure_ascii=False, sort_keys=True), source_uri, now, now),
            )
        job = self.get(job_id)
        assert job is not None
        return job

    def get(self, job_id: str) -> Job | None:
        with self._connect() as db:
            row = db.execute(f"SELECT {_COLUMNS} FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return None if row is None else self._job(row)

    def claim_next(self) -> Job | None:
        now = self.clock()
        lease = uuid.uuid4().hex
        with self._connect() as db:
            # IMMEDIATE берёт блокировку записи до чтения: второй процесс ждёт на BEGIN
            # и после нашего COMMIT видит задачу уже running.
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("SELECT id FROM jobs WHERE status = ? ORDER BY seq LIMIT 1", (QUEUED,)).fetchone()
                if row is None:
                    db.execute("COMMIT")
                    return None
                db.execute(
                    "UPDATE jobs SET status = ?, lease = ?, attempts = attempts + 1, heartbeat_at = ?, updated_at = ?, stage = NULL, fraction = NULL WHERE id = ?",
                    (RUNNING, lease, now, now, row[0]),
                )
                job = self._job(db.execute(f"SELECT {_COLUMNS} FROM jobs WHERE id = ?", (row[0],)).fetchone())
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return job

    def _update_leased(self, job_id: str, lease: str, assignments: str, values: tuple) -> bool:
        now = self.clock()
        with self._connect() as db:
            cur = db.execute(
                f"UPDATE jobs SET {assignments}, updated_at = ? WHERE id = ? AND status = ? AND lease = ?",
                (*values, now, job_id, RUNNING, lease),
            )
            return cur.rowcount == 1

    def set_keys(self, job_id: str, lease: str, *, sha256: str, recognition_key: str, result_key: str) -> bool:
        return self._update_leased(job_id, lease, "sha256 = ?, recognition_key = ?, result_key = ?", (sha256, recognition_key, result_key))

    def progress(self, job_id: str, lease: str, stage: str, fraction: float | None = None) -> bool:
        return self._update_leased(job_id, lease, "stage = ?, fraction = ?, heartbeat_at = ?", (stage, fraction, self.clock()))

    def heartbeat(self, job_id: str, lease: str) -> bool:
        return self._update_leased(job_id, lease, "heartbeat_at = ?", (self.clock(),))

    def complete(self, job_id: str, lease: str, manifest: dict[str, Any]) -> bool:
        return self._update_leased(
            job_id,
            lease,
            "status = ?, manifest = ?, fraction = 1.0, lease = NULL",
            (DONE, json.dumps(manifest, ensure_ascii=False)),
        )

    def fail(self, job_id: str, lease: str, code: str, message: str) -> bool:
        return self._update_leased(job_id, lease, "status = ?, error_code = ?, error_message = ?, lease = NULL", (FAILED, code, message))

    def recover_stale(self, stale_after: float, max_attempts: int = 2) -> list[Recovered]:
        now = self.clock()
        recovered: list[Recovered] = []
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                rows = db.execute(
                    "SELECT id, attempts FROM jobs WHERE status = ? AND (heartbeat_at IS NULL OR heartbeat_at < ?) ORDER BY seq",
                    (RUNNING, now - stale_after),
                ).fetchall()
                for job_id, attempts in rows:
                    if attempts < max_attempts:
                        db.execute(
                            "UPDATE jobs SET status = ?, lease = NULL, stage = NULL, fraction = NULL, heartbeat_at = NULL, updated_at = ? WHERE id = ?",
                            (QUEUED, now, job_id),
                        )
                        recovered.append(Recovered(job_id, QUEUED))
                    else:
                        db.execute(
                            "UPDATE jobs SET status = ?, lease = NULL, error_code = ?, error_message = ?, updated_at = ? WHERE id = ?",
                            (FAILED, WORKER_LOST, f"воркер пропал во время обработки, попыток: {attempts}", now, job_id),
                        )
                        recovered.append(Recovered(job_id, FAILED))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return recovered
