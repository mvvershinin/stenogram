"""Воркер: одна задача за раз, одна модель whisper на процесс.

    python -m stenogram_service.worker [--once]

При старте — проверка моделей (падать на старте, а не на первой задаче), загрузка
модели, возврат зависших задач. Дальше цикл: захват → исходник → ключи → кэш
результата → кэш распознавания → `pipeline.run` → кэш → выгрузка по пути задачи.

Логи несут только job_id, sha256, этап, время и код отказа: имя файла и ключ
исходника обычно содержат фамилию и номер дела.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import logging
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import uuid
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from stenogram import asr, diarization, models, pipeline, render

from . import blocks, manifest
from . import cache as cache_mod
from .config import Settings
from .jobstore import QUEUED, Job, JobStore, SqliteJobStore
from .params import JobParams
from .storage import AccessDenied, InvalidURI, Location, MinioStorage, ObjectNotFound, Storage, StorageError, StorageUnavailable, fetch_source, parse_uri

log = logging.getLogger("stenogram_service.worker")

WORKSPACE_PREFIX = "stenogram-worker-"


class JobFailure(Exception):
    """Отказ с машиночитаемым кодом и текстом без имени файла."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class LeaseLost(Exception):
    """Задачу вернули в очередь (пульс сочли пропавшим) — писать в неё больше нельзя."""


class Interrupted(Exception):
    """Воркер останавливают (SIGTERM): задача здорова и возвращается в очередь."""


class Workspace:
    """Свой подкаталог воркера в общем каталоге работы под блокировкой `flock`.

    Временные каталоги задач лежат внутри. SIGKILL не даёт им прибраться, а в них —
    запись и звук целиком; `sweep` при старте удаляет подкаталоги только тех воркеров,
    чья блокировка свободна, то есть процесс мёртв. Блокировка — на файле рядом
    (`<каталог>.lock`) и берётся до создания каталога: сосед, увидевший каталог,
    видит и живую блокировку."""

    def __init__(self, root: Path):
        self.root = Path(root)
        # STENOGRAM_WORKDIR — часто свежий том (docker-compose.yml), а не каталог,
        # который кто-то создал заранее: без этого первый же запуск падает на
        # открытии lock-файла в несуществующей директории.
        self.root.mkdir(parents=True, exist_ok=True)
        name = f"{WORKSPACE_PREFIX}{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.path = self.root / name
        self._lock_path = self.root / f"{name}.lock"
        self._lock = open(self._lock_path, "w")  # noqa: SIM115 -- держится открытым всю жизнь воркера
        fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path.mkdir()

    def sweep(self) -> list[Path]:
        removed = []
        for path in sorted(self.root.glob(f"{WORKSPACE_PREFIX}*")):
            if not path.is_dir() or path == self.path:
                continue
            lock_path = self.root / f"{path.name}.lock"
            with open(lock_path, "a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue  # хозяин жив
                shutil.rmtree(path, ignore_errors=True)
                lock_path.unlink(missing_ok=True)
                (self.root / f"{path.name}.alive").unlink(missing_ok=True)  # файл-пульс мёртвого воркера
            removed.append(path)
        if removed:
            log.warning("удалено каталогов мёртвых воркеров: %d", len(removed))
        return removed

    def close(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
        self._lock_path.unlink(missing_ok=True)
        self._lock.close()


class Heartbeat:
    """Пульс отдельным потоком: этап отпечатков молчит минутами, и по сообщениям
    прогресса живую задачу сочли бы зависшей.

    `on_beat`, если задан, зовётся на каждый тик до записи в БД — этим пользуется
    докер-healthcheck воркера (`Worker._touch_alive`): файл на диске обновляется
    тем же тактом, что и пульс в БД, независимо от того, дошла ли запись до неё.

    `stall_timeout` — сторож зависания, другая забота, чем сам пульс: пульс тикает
    по расписанию НЕЗАВИСИМО от прогресса (иначе тихий этап сочли бы зависшим),
    а `stall_timeout` следит именно за прогрессом — сколько времени НИ ОДНОГО
    события не было. `mark_progress()` зовёт `Progress` на каждом событии
    (включая чтение исходника по чанкам, `Worker._fetch`).

    Сработавший сторож не просто перестаёт бить — он завершает процесс
    `os._exit(1)`, немедленно и без `finally`/контекст-менеджеров. Так надо:
    при ОДНОМ воркере (обычная топология — «одна видеокарта, один воркер»,
    см. jobstore.py) `recover_stale` некому вызвать, пока этот же воркер сам
    застрял внутри `process()` — «просто замолчать и ждать снаружи» никого не
    разбудит. `os._exit` рвёт процесс сразу, каким бы ни было тело главного
    потока (застрявший `subprocess.run` ffmpeg включительно — Linux убивает
    все процессы PID-неймспейса контейнера вместе с его init-процессом);
    `restart: unless-stopped` в `docker-compose.yml` поднимает контейнер
    заново, а задачу, повисшую без пульса в БД, следующий(-ий после `stale_after`)
    `recover_stale` — уже НОВОГО процесса — вернёт в очередь. Самоисцеления
    (возобновиться, если прогресс всё-таки пришёл) больше нет: процесса к этому
    моменту уже нет."""

    def __init__(
        self,
        store: JobStore,
        job: Job,
        interval: float,
        on_beat: Callable[[], None] | None = None,
        *,
        stall_timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.store, self.job, self.interval, self.on_beat = store, job, interval, on_beat
        self.stall_timeout, self.clock = stall_timeout, clock
        self.lost = threading.Event()
        self.stalled = threading.Event()
        self._progress_lock = threading.Lock()
        self._last_progress = clock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, name=f"heartbeat-{job.id[:8]}", daemon=True)

    def mark_progress(self) -> None:
        with self._progress_lock:
            self._last_progress = self.clock()

    def _since_progress(self) -> float:
        with self._progress_lock:
            return self.clock() - self._last_progress

    def _beat(self) -> None:
        while not self._stop.wait(self.interval):
            since = self._since_progress()
            if self.stall_timeout is not None and since > self.stall_timeout:
                self.stalled.set()
                # Без ключа/имени исходника — только id задачи и длительность тишины.
                log.error(
                    "job=%s сторож зависания: %.0f с без единого события прогресса (порог %.0f с) — принудительный выход процесса",
                    self.job.id,
                    since,
                    self.stall_timeout,
                )
                os._exit(1)  # не возвращается; restart: unless-stopped поднимет контейнер заново
            if self.on_beat is not None:
                with contextlib.suppress(OSError):
                    self.on_beat()
            try:
                if not self.store.heartbeat(self.job.id, self.job.lease):
                    self.lost.set()
                    return
            except Exception as exc:  # база занята — следующий удар через интервал
                log.warning("job=%s пульс не записан: %s", self.job.id, type(exc).__name__)

    def __enter__(self) -> Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval + 5)


class Progress:
    """Колбэк `pipeline.run`: этап и доля в базу не чаще раза в `interval` секунд,
    смена этапа — сразу. Потерянная аренда обрывает прогон на следующем событии."""

    def __init__(self, store: JobStore, job: Job, heartbeat: Heartbeat, interval: float, clock: Callable[[], float], stopping: threading.Event):
        self.store, self.job, self.heartbeat, self.interval, self.clock, self.stopping = store, job, heartbeat, interval, clock, stopping
        self.stage: str | None = None
        self.sha256: str | None = job.sha256
        # После pipeline.run остановка не прерывает: дешевле доделать кэш и выгрузку,
        # чем выбросить готовый результат дорогого этапа.
        self.interruptible = True
        self._written: tuple[str | None, float] = (None, -math.inf)

    def enter(self, stage: str) -> None:
        """Собственные этапы воркера: download, cache, upload."""
        self._check()
        self.heartbeat.mark_progress()
        self.stage = stage
        self._write(stage, None, force=True)

    def __call__(self, event: pipeline.Event) -> None:
        self._check()
        self.heartbeat.mark_progress()
        self.stage = event.stage
        self._write(event.stage, event.fraction, force=event.stage != self._written[0])

    def _check(self) -> None:
        if self.heartbeat.lost.is_set():
            raise LeaseLost
        if self.stopping.is_set() and self.interruptible:
            raise Interrupted

    def _write(self, stage: str, fraction: float | None, force: bool) -> None:
        now = self.clock()
        if not force and now - self._written[1] < self.interval:
            return
        try:
            ok = self.store.progress(self.job.id, self.job.lease, stage, fraction)
        except Exception as exc:  # прогресс — не повод ронять задачу
            log.warning("job=%s этап=%s прогресс не записан: %s", self.job.id, stage, type(exc).__name__)
            return
        if not ok:
            self.heartbeat.lost.set()
            raise LeaseLost
        self._written = (stage, now)
        log.info("job=%s sha256=%s этап=%s", self.job.id, self.sha256, stage)


def needs_source(intermediate: pipeline.Intermediate, options: pipeline.Options) -> bool:
    """Нужен ли `pipeline.run` исходник при повторе из кэша — ровно его условия:
    видео и разметка включены, а плиток в кэше нет или плитки есть, а меток подсветки нет."""
    if not (options.diarization and options.video):
        return False
    return intermediate.tiles is None or (bool(intermediate.tiles) and intermediate.labels is None)


def classify(exc: BaseException, stage: str | None) -> tuple[str, str]:
    """Исключение → (код, текст). Порядок важен: TileSpeakersMismatch — подкласс ValueError."""
    if isinstance(exc, JobFailure):
        return exc.code, exc.message
    if isinstance(exc, diarization.NoSpeech):
        return "no_speech", "в записи не найдено речи достаточной длительности"
    if isinstance(exc, pipeline.TileSpeakersMismatch):
        return "invalid_params", str(exc)
    if isinstance(exc, StorageUnavailable):
        return "storage_unavailable", f"хранилище недоступно: {exc}"
    if isinstance(exc, StorageError):
        return "internal", f"хранилище кэша или результатов отказало: {exc}"
    if isinstance(exc, urllib.error.URLError):
        # Сеть в библиотеке трогает только скачивание моделей разметки.
        return "model_unavailable", "модели разметки участников нет на диске, а скачать её не вышло"
    if stage == "audio" and isinstance(exc, (RuntimeError, subprocess.CalledProcessError, ValueError, wave.Error, EOFError)):
        return "no_audio", "не удалось извлечь звук из исходника"
    if isinstance(exc, MemoryError):
        return "internal", "не хватило памяти"
    return "internal", f"{type(exc).__name__}: {exc}"


def _scrub(text: str, job: Job) -> str:
    """Убирает адрес и имя исходника из текста — на случай, если чужое исключение их несёт."""
    secrets = [job.source_uri]
    try:
        key = parse_uri(job.source_uri).key
    except InvalidURI:
        key = ""
    name = key.rsplit("/", 1)[-1]
    secrets += [key, name, name.rsplit(".", 1)[0]]
    for secret in sorted({s for s in secrets if len(s) >= 3}, key=len, reverse=True):
        text = text.replace(secret, "<источник>")
    return text


class Worker:
    def __init__(self, settings: Settings, store: JobStore, storage: Storage, whisper_model: Any, *, clock: Callable[[], float] = time.monotonic):
        if not settings.cache_bucket:
            raise ValueError("не задан бакет кэша")
        self.settings, self.store, self.storage, self.whisper_model, self.clock = settings, store, storage, whisper_model, clock
        self.cache = cache_mod.Cache(storage, settings.cache_bucket, settings.cache_prefix)
        self.workspace = Workspace(Path(settings.workdir))
        # Файл-пульс для докер-healthcheck: свежесть mtime — без доступа к SQLite
        # изнутри контейнера. Обновляется в трёх местах: здесь (старт), на каждом
        # круге serve() (воркер жив и простаивает между задачами) и на каждом тике
        # Heartbeat (воркер жив и работает, даже если пайплайн молчит минутами).
        self._alive_path = self.workspace.root / f"{self.workspace.path.name}.alive"
        self._touch_alive()
        # Остановка: текущая задача прерывается на ближайшем событии и возвращается
        # в очередь без списания попытки, цикл завершается.
        self.stopping = threading.Event()

    def _touch_alive(self) -> None:
        # не критично: healthcheck просто не увидит свежий файл на этом такте
        with contextlib.suppress(OSError):
            self._alive_path.touch()

    def close(self) -> None:
        self.workspace.close()
        self._alive_path.unlink(missing_ok=True)

    # --- цикл ---------------------------------------------------------------------

    def recover(self) -> None:
        """Каждый круг, а не только при старте: после быстрого перезапуска пульс убитой
        задачи ещё свежий, и стартовая проверка её не видит."""
        for item in self.store.recover_stale(self.settings.stale_after, self.settings.max_attempts):
            outcome = "возвращена в очередь" if item.status == QUEUED else "отказ worker_lost"
            log.warning("job=%s воркер пропал во время обработки — %s", item.job_id, outcome)

    def run_once(self) -> bool:
        self.recover()
        job = self.store.claim_next()
        if job is None:
            return False
        self.process(job)
        return True

    def serve(self, *, once: bool = False) -> None:
        """Цикл до `stopping`. Сбой круга (база занята, диск) не роняет воркер:
        лог, пауза, следующий круг. `once` — разобрать очередь и выйти."""
        while not self.stopping.is_set():
            self._touch_alive()
            try:
                took = self.run_once()
            except Exception as exc:
                # В круге нет ни задачи, ни исходника — только база и каталог работы.
                log.error("круг воркера не удался: %s: %s", type(exc).__name__, str(exc)[:200])
                self.stopping.wait(self.settings.poll_interval)
                continue
            if not took:
                if once:
                    return
                self.stopping.wait(self.settings.poll_interval)

    # --- задача -------------------------------------------------------------------

    def process(self, job: Job) -> None:
        started = self.clock()
        log.info("job=%s взята, попытка %d", job.id, job.attempts)
        heartbeat = Heartbeat(self.store, job, self.settings.heartbeat_interval, on_beat=self._touch_alive, stall_timeout=self.settings.stall_timeout, clock=self.clock)
        progress = Progress(self.store, job, heartbeat, self.settings.progress_interval, self.clock, self.stopping)
        try:
            with heartbeat, tempfile.TemporaryDirectory(dir=self.workspace.path, prefix="job-") as tmp:
                result = self._execute(job, Path(tmp), progress, heartbeat)
                result["timing"] = {"seconds": round(self.clock() - started, 3)}
                if heartbeat.lost.is_set() or not self.store.complete(job.id, job.lease, result):
                    raise LeaseLost
            log.info("job=%s sha256=%s готово за %.1f с, кэш: %s", job.id, progress.sha256, self.clock() - started, result["cache"])
        except LeaseLost:
            log.warning("job=%s аренда потеряна — задача ушла другому, результат не записан", job.id)
        except Interrupted:
            try:
                released = self.store.release(job.id, job.lease)
            except Exception as exc:
                released = False
                log.error("job=%s не возвращена в очередь: %s", job.id, type(exc).__name__)
            if released:
                log.warning("job=%s этап=%s остановка воркера — задача возвращена в очередь без списания попытки", job.id, progress.stage)
        except Exception as exc:
            self._fail(job, exc, progress)
        finally:
            # Исключение с кадрами пайплайна (массивы на сотни мегабайт) уже отпущено;
            # сборка — чтобы циклические ссылки не дожили до следующей задачи.
            gc.collect()

    def _fail(self, job: Job, exc: Exception, progress: Progress) -> None:
        code, message = classify(exc, progress.stage)
        message = _scrub(message, job)[:1000]
        log.error("job=%s sha256=%s этап=%s отказ: %s", job.id, progress.sha256, progress.stage, code)
        if code == "internal":
            log.debug("job=%s трассировка:\n%s", job.id, _scrub("".join(traceback.format_exception(exc)), job))
        try:
            if not self.store.fail(job.id, job.lease, code, message):
                log.warning("job=%s отказ не записан: аренда потеряна", job.id)
        except Exception as store_exc:
            log.error("job=%s отказ не записан в базу: %s", job.id, type(store_exc).__name__)

    def _execute(self, job: Job, tmp: Path, progress: Progress, heartbeat: Heartbeat) -> dict[str, Any]:
        progress.enter("prepare")
        try:
            params = JobParams.model_validate(job.params)
        except ValidationError as exc:
            details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise JobFailure("invalid_params", f"параметры задачи не прошли проверку: {details}") from None
        s = self.settings
        options = params.options(model=s.whisper_model, device=s.device, compute_type=s.compute_type, threads=s.threads)
        wav = tmp / "audio16k.wav"

        sha, source = job.sha256, None
        if sha is None:
            source, sha = self._fetch(job, tmp, progress)
        while True:
            progress.sha256 = sha
            rkey, reskey = cache_mod.keys_for(sha, options)
            if not self.store.set_keys(job.id, job.lease, sha256=sha, recognition_key=rkey, result_key=reskey):
                raise LeaseLost
            progress.enter("cache")
            document = self.cache.get_result(reskey)
            if document is not None:
                state, quality = "hit", {**(self.cache.get_meta(reskey) or {}), "degraded": False}
                break
            intermediate = self.cache.get_intermediate(rkey, wav)
            if source is None and (intermediate is None or needs_source(intermediate, options)):
                # Перезапуск без скачивания не обошёлся: исходник нужен, и заодно
                # проверяется, что по тому же адресу лежит тот же файл.
                source, actual = self._fetch(job, tmp, progress)
                if actual != sha:
                    log.warning("job=%s исходник по тому же адресу изменился: sha256 %s вместо %s", job.id, actual, sha)
                    sha = actual
                    continue
            document, quality, state = self._decode(sha, rkey, reskey, source, wav, options, intermediate, progress)
            break

        if not document["blocks"]:
            raise JobFailure("no_speech", "в записи не найдено речи")
        if heartbeat.lost.is_set():
            raise LeaseLost
        progress.enter("upload")
        return {
            "source": {"uri": job.source_uri, "sha256": sha, "mime": manifest.guess_mime(job.source_uri)},
            "decoder": manifest.decoder_info(s, params.model_dump()),
            "result": self._publish(job, document, params.names_by_index()),
            "quality": quality,
            "cache": state,
        }

    def _fetch(self, job: Job, tmp: Path, progress: Progress) -> tuple[Path, str]:
        progress.enter("download")
        try:
            # on_chunk — тот же mark_progress, что и у событий пайплайна: без него
            # медленный канал на большом файле выглядел бы зависшим этапом download
            # для сторожа (Heartbeat.stall_timeout), хотя байты исправно идут.
            return fetch_source(self.storage, job.source_uri, tmp, on_chunk=progress.heartbeat.mark_progress)
        except InvalidURI as exc:
            raise JobFailure("invalid_params", str(exc)) from None
        except ObjectNotFound:
            raise JobFailure("source_not_found", "исходник не найден в хранилище") from None
        except AccessDenied:
            raise JobFailure("source_access_denied", "нет доступа к исходнику в хранилище") from None

    def _decode(self, sha, rkey, reskey, source, wav, options, intermediate, progress) -> tuple[dict[str, Any], dict[str, Any], str]:
        # Модель нужна только полному прогону: при повторе из кэша распознавания её нет смысла передавать.
        held = self.whisper_model if intermediate is None else None
        result = pipeline.run(source, wav, options, reuse=intermediate, whisper_model=held, progress=progress)
        progress.interruptible = False
        document = blocks.build(result.utterances, sha)
        tiles = result.intermediate.tiles
        quality = {
            "speaker_source": "видео + голос" if result.video_used else "голос",
            "tiles": len(tiles) if tiles else 0,
            "speakers": len({u.speaker for u in result.utterances}),
            "language": result.intermediate.language,
            "duration": round(result.duration, 3),
        }
        progress.enter("cache")
        # Результат уже посчитан: сбой записи кэша стоит следующей задаче времени,
        # но эту задачу не роняет.
        try:
            if intermediate is None:
                self.cache.put_intermediate(rkey, result.intermediate, wav)
            elif result.intermediate.to_dict() != intermediate.to_dict():
                self.cache.put_intermediate(rkey, result.intermediate, None)  # дописаны плитки
            self.cache.put_result(reskey, document, render.to_text(result.utterances), video_error=result.video_error, meta=quality)
        except StorageError as exc:
            log.warning("sha256=%s кэш не записан: %s", sha, type(exc).__name__)
        return document, {**quality, "degraded": result.video_error is not None}, "miss" if intermediate is None else "asr-hit"

    def _publish(self, job: Job, document: dict[str, Any], names: dict[int, str]) -> dict[str, str]:
        """Документ и стенограмма с именами задачи — по пути задачи, не по ключу кэша."""
        bucket = self.settings.results_bucket
        assert bucket is not None
        base = f"{self.settings.result_prefix}{job.id}/"
        blocks_loc, text_loc = Location(bucket, base + "blocks.json"), Location(bucket, base + "transcript.txt")
        text = render.to_text(blocks.to_utterances(document), names or None)
        self.storage.put_bytes(text_loc, text.encode("utf-8"), "text/plain; charset=utf-8")
        self.storage.put_json(blocks_loc, blocks.for_job(document, source_uri=job.source_uri, names=names))
        return {"blocks": str(blocks_loc), "transcript": str(text_loc)}


def cuda_preflight(whisper_model: Any) -> None:
    """Пробное кодирование секунды тишины уже загруженной моделью.

    `ctranslate2.get_cuda_device_count()` видит только драйвер и отвечает `1`,
    даже когда faster-whisper не находит свои же CUDA-библиотеки (cuBLAS/cuDNN
    приходят pip-пакетами `nvidia-cublas-cu12`/`nvidia-cudnn-cu12`, а без них на
    `LD_LIBRARY_PATH` отказ приходит только на первом вызове энкодера — см.
    `docs/ускорение-на-gpu.md` и `service/Dockerfile`). Не проверять на старте — значит
    уронить первую же настоящую задачу минутами позже, после долгого download/decode.
    Секунда тишины — дёшево и достаточно: до первого вызова `ctranslate2.models.Whisper`
    дело просто не доходит, если библиотек нет."""
    silence = np.zeros(16000, dtype=np.float32)  # 1 секунда, 16 кГц, как весь пайплайн
    segments, _ = whisper_model.transcribe(silence, beam_size=1, word_timestamps=False, vad_filter=False)
    list(segments)  # генератор — сам проход энкодера происходит только при обходе


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stenogram_service.worker", description="Воркер стенограмм: берёт задачи из очереди по одной")
    parser.add_argument("--once", action="store_true", help="разобрать очередь и выйти")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    settings = Settings.from_env()
    problems = settings.worker_problems() + models.check_all(whisper_model=settings.whisper_model, whisper_dir=settings.whisper_dir)
    if problems:
        print("Воркер не запущен:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1

    def cuda_not_working(exc: Exception) -> int:
        print("Воркер не запущен: CUDA не работает.", file=sys.stderr)
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  обычная причина — нет драйвера/карты либо faster-whisper не находит свои CUDA-библиотеки (LD_LIBRARY_PATH, см. service/Dockerfile)", file=sys.stderr)
        return 1

    try:
        whisper = asr.load_model(
            settings.whisper_model,
            device=settings.device,
            compute_type=settings.compute_type,
            num_threads=settings.resolved_threads(),
            local_files_only=settings.local_files_only,
            download_root=settings.whisper_dir,
        )
    except Exception as exc:
        # На cpu (и на auto, который сам решает, просто до выбора устройства дело
        # тут ещё не дошло) отказ загрузки модели — что угодно (битый файл,
        # нехватка памяти) — летит наружу как есть: это не тот класс проблем, для
        # которого здесь придумана классификация. На явном cuda отказ уже здесь —
        # почти наверняка «нет ни драйвера, ни карты» (ctranslate2 падает прямо
        # внутри WhisperModel(...), до всякого cuda_preflight).
        if settings.device != "cuda":
            raise
        return cuda_not_working(exc)

    # STENOGRAM_DEVICE=auto сам решает cpu или cuda — какое устройство досталось
    # на самом деле, знает только загруженная модель (ctranslate2.models.Whisper.device),
    # а не запрошенное имя. Проверяем именно его: иначе на auto+cuda-железе
    # пробное кодирование не запускалось бы вовсе, и та же гонка (get_cuda_device_count()
    # видит только драйвер, без cuBLAS/cuDNN на LD_LIBRARY_PATH падает лишь первый
    # настоящий энкод) ловилась бы уже на боевой задаче.
    resolved_device = getattr(getattr(whisper, "model", None), "device", settings.device)
    if resolved_device == "cuda":
        try:
            cuda_preflight(whisper)
        except Exception as exc:
            return cuda_not_working(exc)

    store = SqliteJobStore(Path(settings.db_path))
    storage = MinioStorage.connect(settings.s3_endpoint, settings.s3_access_key, settings.s3_secret_key, secure=settings.s3_secure, region=settings.s3_region)
    worker = Worker(settings, store, storage, whisper)
    worker.workspace.sweep()

    # SIGTERM — прервать текущую задачу на ближайшем событии, вернуть её в очередь
    # без списания попытки и выйти. Иначе оркестратор, не дождавшись, добьёт SIGKILL,
    # и здоровая задача сгорит одну попытку.
    def on_signal(signum, frame) -> None:
        if not worker.stopping.is_set():
            log.warning("получен сигнал %d — останавливаюсь", signum)
        worker.stopping.set()

    previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        log.info("воркер запущен: модель %s, %s/%s", settings.whisper_model, settings.device, settings.compute_type)
        worker.serve(once=args.once)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        worker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
