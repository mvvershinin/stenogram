"""Очередь задач на SQLite.

Что здесь держится:
  - захват атомарен между процессами: при настоящей гонке ни одна задача не достаётся
    двоим, и ни одна не теряется;
  - аренда: после возврата задачи в очередь прежний владелец ничего в неё не пишет;
  - зависшая задача (воркер убит SIGKILL, `except` этого не видел) один раз
    возвращается в очередь, а на второй — отказ `worker_lost`: задача, которая сама
    роняет воркер по памяти, не зацикливает его.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from stenogram_service.jobstore import DONE, FAILED, QUEUED, RUNNING, WORKER_LOST, SqliteJobStore


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path, clock) -> SqliteJobStore:
    return SqliteJobStore(tmp_path / "jobs.sqlite", clock=clock)


def test_create_and_get(store):
    job = store.create("s3://records/x.mp4", {"speakers": 3, "names": {"0": "Иванов"}})

    got = store.get(job.id)
    assert got == job
    assert got.status == QUEUED
    assert got.params == {"speakers": 3, "names": {"0": "Иванов"}}
    assert got.attempts == 0 and got.lease is None
    assert store.get("нет-такой") is None


def test_claim_is_fifo_and_empty_queue_gives_none(store):
    first = store.create("s3://b/1.mp4", {})
    second = store.create("s3://b/2.mp4", {})

    a, b = store.claim_next(), store.claim_next()

    assert (a.id, b.id) == (first.id, second.id)
    assert a.status == RUNNING and a.attempts == 1 and a.lease and a.lease != b.lease
    assert store.claim_next() is None


def test_progress_keys_and_completion(store, clock):
    job = store.create("s3://b/1.mp4", {})
    claimed = store.claim_next()

    clock.now += 5
    assert store.set_keys(job.id, claimed.lease, sha256="a" * 64, recognition_key="r" * 64, result_key="k" * 64)
    assert store.progress(job.id, claimed.lease, "asr", 0.25)
    running = store.get(job.id)
    assert (running.stage, running.fraction, running.heartbeat_at) == ("asr", 0.25, 1005.0)
    assert (running.sha256, running.recognition_key, running.result_key) == ("a" * 64, "r" * 64, "k" * 64)

    assert store.complete(job.id, claimed.lease, {"blocks": "s3://cache/result/k/blocks.json"})
    done = store.get(job.id)
    assert done.status == DONE and done.manifest == {"blocks": "s3://cache/result/k/blocks.json"} and done.lease is None
    # Завершённая задача больше не принимает записей ни с какой арендой.
    assert not store.progress(job.id, claimed.lease, "asr", 0.5)


def test_fail_stores_machine_readable_code(store):
    job = store.create("s3://b/1.mp4", {})
    claimed = store.claim_next()

    assert store.fail(job.id, claimed.lease, "source_not_found", "исходник не найден")

    failed = store.get(job.id)
    assert (failed.status, failed.error_code, failed.error_message) == (FAILED, "source_not_found", "исходник не найден")


def test_foreign_lease_cannot_write(store):
    job = store.create("s3://b/1.mp4", {})
    store.claim_next()

    assert not store.heartbeat(job.id, "чужая")
    assert not store.complete(job.id, "чужая", {})
    assert not store.fail(job.id, "чужая", "x", "y")
    assert store.get(job.id).status == RUNNING


def test_fresh_running_job_is_not_recovered(store, clock):
    store.create("s3://b/1.mp4", {})
    claimed = store.claim_next()
    clock.now += 50
    store.heartbeat(claimed.id, claimed.lease)
    clock.now += 50

    assert store.recover_stale(stale_after=60) == []
    assert store.get(claimed.id).status == RUNNING


def test_stale_job_is_requeued_once_then_failed(store, clock):
    job = store.create("s3://b/1.mp4", {})

    first = store.claim_next()
    clock.now += 600  # воркер убит: пульса больше нет
    recovered = store.recover_stale(stale_after=60, max_attempts=2)
    assert [(r.job_id, r.status) for r in recovered] == [(job.id, QUEUED)]
    requeued = store.get(job.id)
    assert (requeued.status, requeued.attempts, requeued.lease, requeued.stage) == (QUEUED, 1, None, None)

    # Прежний владелец, если он всё-таки жив, в задачу больше не пишет.
    assert not store.progress(job.id, first.lease, "asr", 0.9)
    assert not store.complete(job.id, first.lease, {})

    second = store.claim_next()
    assert second.id == job.id and second.attempts == 2
    clock.now += 600
    recovered = store.recover_stale(stale_after=60, max_attempts=2)
    assert [(r.job_id, r.status) for r in recovered] == [(job.id, FAILED)]
    lost = store.get(job.id)
    assert (lost.status, lost.error_code, lost.lease) == (FAILED, WORKER_LOST, None)
    assert store.claim_next() is None


def test_two_stores_on_one_file_see_each_other(tmp_path):
    """API и воркер — разные объекты (и процессы) на одном файле."""
    api = SqliteJobStore(tmp_path / "jobs.sqlite")
    worker = SqliteJobStore(tmp_path / "jobs.sqlite")

    job = api.create("s3://b/1.mp4", {})
    claimed = worker.claim_next()
    worker.progress(claimed.id, claimed.lease, "video", 0.5)

    assert claimed.id == job.id
    assert api.get(job.id).stage == "video"


# --- гонка настоящих процессов -------------------------------------------------


def _claim_all(path: str, barrier, out) -> None:
    store = SqliteJobStore(Path(path))
    barrier.wait()  # все стартуют разом — иначе гонки нет и тест ничего не доказывает
    claimed = []
    while (job := store.claim_next()) is not None:
        claimed.append(job.id)
    out.put(claimed)


def test_concurrent_processes_never_claim_the_same_job(tmp_path):
    path = tmp_path / "jobs.sqlite"
    store = SqliteJobStore(path)
    ids = {store.create(f"s3://b/{i}.mp4", {}).id for i in range(60)}

    ctx = multiprocessing.get_context("spawn")
    workers = 4
    barrier, out = ctx.Barrier(workers), ctx.Queue()
    procs = [ctx.Process(target=_claim_all, args=(str(path), barrier, out)) for _ in range(workers)]
    for p in procs:
        p.start()
    results = [out.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    claimed = [job_id for r in results for job_id in r]
    assert len(claimed) == len(set(claimed)), "одна задача захвачена дважды"
    assert set(claimed) == ids
    assert all(store.get(i).status == RUNNING and store.get(i).attempts == 1 for i in ids)


def _open_fresh_files(paths: list[str], barrier, out) -> None:
    errors = []
    for path in paths:
        barrier.wait()  # все открывают один и тот же новый файл одновременно
        try:
            SqliteJobStore(Path(path))
        except Exception as exc:
            errors.append(repr(exc))
    out.put(errors)


def test_concurrent_first_start_on_new_file(tmp_path):
    """API и воркер стартуют разом на ещё не созданной базе. Смена режима журнала
    на WAL при этом отдаёт «database is locked» мимо busy_timeout — без повтора
    падала примерно треть стартов."""
    paths = [str(tmp_path / f"jobs-{i}.sqlite") for i in range(15)]
    ctx = multiprocessing.get_context("spawn")
    workers = 4
    barrier, out = ctx.Barrier(workers), ctx.Queue()
    procs = [ctx.Process(target=_open_fresh_files, args=(paths, barrier, out)) for _ in range(workers)]
    for p in procs:
        p.start()
    errors = [e for _ in procs for e in out.get(timeout=120)]
    for p in procs:
        p.join(timeout=60)

    assert errors == []
    store = SqliteJobStore(Path(paths[0]))
    assert store.claim_next() is None  # таблица на месте


def test_create_with_known_sha256(store):
    """Перезапуск несёт хэш родителя: воркер сможет обойтись без скачивания."""
    job = store.create("s3://b/1.mp4", {}, sha256="c" * 64)
    assert store.get(job.id).sha256 == "c" * 64
    assert store.claim_next().sha256 == "c" * 64


def test_release_requeues_without_spending_an_attempt(store):
    """Остановка воркера по SIGTERM — не сбой задачи: попытка не списывается."""
    job = store.create("s3://b/1.mp4", {})
    claimed = store.claim_next()
    store.progress(job.id, claimed.lease, "asr", 0.5)

    assert store.release(job.id, claimed.lease)

    released = store.get(job.id)
    assert (released.status, released.attempts, released.lease, released.stage, released.heartbeat_at) == (QUEUED, 0, None, None, None)
    assert not store.release(job.id, claimed.lease)  # аренды больше нет
    assert not store.progress(job.id, claimed.lease, "asr", 0.6)
    assert store.claim_next().attempts == 1
