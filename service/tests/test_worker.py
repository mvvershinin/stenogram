"""Воркер: путь задачи через кэш, отказы с кодами, пульс, память, старт.

`pipeline.run` и `asr.load_model` подменены, хранилище в памяти, кодек звука кэша —
копирование байтов (сам FLAC проверен в test_cache). Сети, моделей и ffmpeg нет.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import time
import urllib.error
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import write_tone
from fastapi.testclient import TestClient

from stenogram import asr, diarization, models, pipeline
from stenogram.align import Utterance
from stenogram.asr import Segment, Word
from stenogram.video import Tile
from stenogram_service import cache, worker
from stenogram_service.api import create_app
from stenogram_service.config import Settings
from stenogram_service.jobstore import SqliteJobStore
from stenogram_service.params import JobRequest
from stenogram_service.storage import AccessDenied, Location, MinioStorage, StorageUnavailable

SOURCE_KEY = "заседания/Петров_А40-12345-2026.mp4"
SOURCE = f"s3://records/{SOURCE_KEY}"
DATA = b"synthetic video bytes, not a real recording " * 50
SHA = hashlib.sha256(DATA).hexdigest()
TILES = [Tile(10, 0, 3, 20), Tile(50, 0, 3, 20)]


class FakeRun:
    """Вместо pipeline.run. `hook(progress)` — что сделать посреди прогона."""

    def __init__(self):
        self.calls: list[dict] = []
        self.hook = None
        self.utterances = [Utterance(0.0, 2.0, 0, "Добрый день, коллеги."), Utterance(2.5, 4.0, 1, "Начнём с первого вопроса.")]
        self.video_error: str | None = None
        self.tiles_found = TILES

    def __call__(self, source, wav, options=None, *, reuse=None, whisper_model=None, progress=None):
        self.calls.append({"source": source, "reuse": reuse, "whisper_model": whisper_model, "options": options})
        progress(pipeline.Event("audio", "[1/4] ..."))
        if self.hook is not None:
            self.hook(progress)
        want_video = options.video and options.diarization
        if reuse is None:
            assert source.name == f"{SHA}.mp4"  # настоящее имя в пайплайн не идёт
            write_tone(wav, 0.1)
            segments = [Segment(0.0, 4.0, "Добрый день, коллеги. Начнём с первого вопроса.", [Word(0.0, 0.5, " Добрый")])]
            intermediate = pipeline.Intermediate(segments, "ru")
        else:
            assert wav.exists()  # звук из кэша на месте
            intermediate = reuse
        if want_video and intermediate.tiles is None:
            tiles = [] if self.video_error else self.tiles_found
            intermediate = dataclasses.replace(intermediate, tiles=tiles, labels=np.zeros(20, dtype=int) if tiles else None, step=0.2, video_error=self.video_error)
        used = want_video and bool(intermediate.tiles)
        return pipeline.Result(list(self.utterances), [], 4.0, intermediate, video_error=self.video_error if want_video else None, video_used=used)


def _copy(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)


@pytest.fixture
def env(tmp_path, monkeypatch, storage):
    monkeypatch.setattr(cache, "wav_to_flac", _copy)
    monkeypatch.setattr(cache, "flac_to_wav", _copy)
    run = FakeRun()
    monkeypatch.setattr(pipeline, "run", run)
    workdir = tmp_path / "work"
    workdir.mkdir()
    settings = Settings(
        db_path=str(tmp_path / "jobs.sqlite"),
        workdir=str(workdir),
        s3_endpoint="minio:9000",
        cache_bucket="cache",
        result_bucket="out",
        heartbeat_interval=0.05,
        stale_after=60,
        poll_interval=0.01,
    )
    store = SqliteJobStore(Path(settings.db_path))
    storage.objects[Location("records", SOURCE_KEY)] = DATA
    return SimpleNamespace(
        run=run,
        settings=settings,
        store=store,
        storage=storage,
        workdir=workdir,
        worker=worker.Worker(settings, store, storage, whisper_model="held-model"),
        api=TestClient(create_app(settings, store=store, check_models=lambda: [])),
    )


def _submit(env, **params) -> str:
    return env.store.create(SOURCE, JobRequest(source=SOURCE, **params).params()).id


def _rerun(env, parent_id: str, **overrides) -> str:
    response = env.api.post(f"/jobs/{parent_id}/rerun", json=overrides)
    assert response.status_code == 202
    return response.json()["job_id"]


def _drain(env) -> None:
    env.worker.serve(once=True)


def _manifest(env, job_id: str) -> dict:
    return env.api.get(f"/jobs/{job_id}").json()


def _source_reads(env) -> int:
    return sum(1 for loc in env.storage.reads if loc.bucket == "records")


# --- путь через кэш ---------------------------------------------------------------


def test_miss_runs_pipeline_and_publishes_named_outputs(env):
    job_id = _submit(env, speakers=2, names={"0": "Председатель", "1": "Секретарь"})

    _drain(env)

    manifest = _manifest(env, job_id)
    assert manifest["status"] == "done", manifest
    assert manifest["cache"] == "miss"
    assert manifest["source"] == {"uri": SOURCE, "sha256": SHA, "mime": "video/mp4"}
    assert manifest["quality"] == {"speaker_source": "видео + голос", "tiles": 2, "speakers": 2, "language": "ru", "duration": 4.0, "degraded": False}
    assert manifest["timing"]["seconds"] >= 0
    assert manifest["result"] == {"blocks": f"s3://out/stenogram/jobs/{job_id}/blocks.json", "transcript": f"s3://out/stenogram/jobs/{job_id}/transcript.txt"}
    assert env.run.calls[0]["whisper_model"] == "held-model"

    document = json.loads(env.storage.objects[Location("out", f"stenogram/jobs/{job_id}/blocks.json")])
    assert document["speakers"] == {"0": "Председатель", "1": "Секретарь"}
    assert document["source_uri"] == SOURCE and document["doc_id"] == SHA
    assert [b["text"] for b in document["blocks"]] == ["Добрый день, коллеги.", "Начнём с первого вопроса."]
    text = env.storage.objects[Location("out", f"stenogram/jobs/{job_id}/transcript.txt")].decode()
    assert "Председатель: Добрый день" in text and "Секретарь: Начнём" in text

    job = env.store.get(job_id)
    assert env.storage.exists(env.worker.cache.intermediate_location(job.recognition_key))
    assert env.storage.exists(env.worker.cache.blocks_location(job.result_key))
    assert list(env.worker.workspace.path.iterdir()) == []


def test_result_hit_skips_pipeline_and_rerun_skips_download(env):
    first = _submit(env, speakers=2)
    _drain(env)
    env.run.calls.clear()
    reads_before = _source_reads(env)

    again = _submit(env, speakers=2, names={"0": "Иванов"})  # новая задача: sha256 неизвестен
    _drain(env)
    rerun = _rerun(env, first)  # перезапуск: sha256 от родителя
    _drain(env)

    assert env.run.calls == []
    assert _manifest(env, again)["cache"] == "hit"
    assert _source_reads(env) == reads_before + 1  # скачала только новая задача, не перезапуск
    manifest = _manifest(env, rerun)
    assert manifest["status"] == "done" and manifest["cache"] == "hit"
    assert manifest["quality"]["speaker_source"] == "видео + голос"  # сводка из кэша
    named = json.loads(env.storage.objects[Location("out", f"stenogram/jobs/{again}/blocks.json")])
    assert named["speakers"] == {"0": "Иванов"}
    assert "Иванов: Добрый день" in env.storage.objects[Location("out", f"stenogram/jobs/{again}/transcript.txt")].decode()


def test_rerun_with_other_speakers_reuses_recognition_without_source(env):
    first = _submit(env, speakers=2)
    _drain(env)
    env.run.calls.clear()
    reads_before = _source_reads(env)

    rerun = _rerun(env, first, speakers=3)
    _drain(env)

    assert _manifest(env, rerun)["cache"] == "asr-hit"
    [call] = env.run.calls
    assert call["reuse"] is not None and call["reuse"].tiles == TILES
    assert call["source"] is None  # плитки и метки в кэше — исходник не нужен
    assert call["options"].speakers == 3
    assert call["whisper_model"] is None  # распознавания нет — и модель не нужна
    assert _source_reads(env) == reads_before
    assert env.store.get(rerun).params["speakers"] == 3


def test_asr_hit_with_empty_tiles_never_reads_source(env):
    """«Искали и не нашли» — знание о записи: повтор по видео исходник не трогает."""
    env.run.tiles_found = []
    first = _submit(env)
    _drain(env)
    reads_before = _source_reads(env)

    rerun = _rerun(env, first, speakers=2)
    _drain(env)

    assert _manifest(env, rerun)["cache"] == "asr-hit"
    assert env.run.calls[-1]["source"] is None
    assert _source_reads(env) == reads_before


def test_asr_hit_without_tiles_downloads_source_for_video(env):
    first = _submit(env, video=False)
    _drain(env)
    first_job = env.store.get(first)
    assert json.loads(env.storage.objects[env.worker.cache.intermediate_location(first_job.recognition_key)])["tiles"] is None

    rerun = _rerun(env, first, video=True)
    _drain(env)

    assert _manifest(env, rerun)["cache"] == "asr-hit"
    call = env.run.calls[-1]
    assert call["reuse"] is not None and call["source"].name == f"{SHA}.mp4"
    assert call["whisper_model"] is None
    # Найденные плитки дописаны в кэш распознавания — следующему повтору исходник не нужен.
    assert json.loads(env.storage.objects[env.worker.cache.intermediate_location(first_job.recognition_key)])["tiles"] == [[10, 0, 3, 20], [50, 0, 3, 20]]


def test_inherited_sha256_is_rechecked_when_source_is_downloaded(env, tmp_path):
    """Файл по адресу родителя заменили: ключи пересчитываются по настоящему содержимому."""
    stale = "f" * 64
    params = JobRequest(source=SOURCE).params()
    options = JobRequest(source=SOURCE).options(model=env.settings.whisper_model, device="cpu", compute_type="int8", threads=0)
    rkey, _ = cache.keys_for(stale, options)
    old = pipeline.Intermediate([Segment(0.0, 1.0, "старое")], "ru")  # плиток нет, видео нужно
    env.worker.cache.put_intermediate(rkey, old, write_tone(tmp_path / "old.wav", 0.1))
    job_id = env.store.create(SOURCE, params, sha256=stale).id

    _drain(env)

    job = env.store.get(job_id)
    assert job.status == "done" and job.sha256 == SHA
    assert [c["reuse"] for c in env.run.calls] == [None]  # полный прогон по новому файлу
    assert _manifest(env, job_id)["cache"] == "miss"


def test_video_fallback_is_not_cached_as_result(env):
    env.run.video_error = "MemoryError"
    job_id = _submit(env)
    _drain(env)

    manifest = _manifest(env, job_id)
    assert manifest["status"] == "done"
    assert manifest["quality"]["degraded"] is True and manifest["quality"]["speaker_source"] == "голос"
    assert env.worker.cache.get_result(env.store.get(job_id).result_key) is None


# --- отказы -------------------------------------------------------------------------


def _fail_with(env, hook=None, **params) -> dict:
    env.run.hook = hook
    job_id = _submit(env, **params)
    _drain(env)
    manifest = _manifest(env, job_id)
    assert manifest["status"] == "failed", manifest
    assert list(env.worker.workspace.path.iterdir()) == []  # временный каталог убран и при отказе
    return manifest["error"]


def test_tile_speakers_mismatch_is_invalid_params_without_retry(env):
    def mismatch(progress):
        raise pipeline.TileSpeakersMismatch("--tile-speakers: чисел передано — 3, а плиток в видео найдено — 2")

    error = _fail_with(env, mismatch, tile_speakers=[1, 1, 1])

    assert error["code"] == "invalid_params" and error["retryable"] is False
    assert "плиток в видео найдено — 2" in error["message"]


def test_bad_params_row_is_invalid_params(env):
    job_id = env.store.create(SOURCE, {"speakers": 0}).id
    _drain(env)
    assert _manifest(env, job_id)["error"]["code"] == "invalid_params"
    assert env.run.calls == []


@pytest.mark.parametrize(
    ("setup", "code", "retryable"),
    [
        (lambda s: s.fail.__setitem__(Location("records", SOURCE_KEY), StorageUnavailable("нет связи с хранилищем", "ProtocolError")), "storage_unavailable", True),
        (lambda s: s.fail.__setitem__(Location("records", SOURCE_KEY), AccessDenied("нет доступа к хранилищу", "AccessDenied")), "source_access_denied", False),
        (lambda s: s.objects.pop(Location("records", SOURCE_KEY)), "source_not_found", False),
    ],
)
def test_source_storage_failures_have_codes(env, setup, code, retryable):
    setup(env.storage)
    error = _fail_with(env)
    assert (error["code"], error["retryable"]) == (code, retryable)
    assert env.run.calls == []


def test_result_upload_outage_is_retryable(env):
    original = env.storage.put_bytes

    def put_bytes(loc, data, content_type="application/octet-stream"):
        if loc.bucket == "out":
            raise StorageUnavailable("хранилище временно недоступно", "SlowDown")
        return original(loc, data, content_type)

    env.storage.put_bytes = put_bytes
    error = _fail_with(env)
    assert (error["code"], error["retryable"]) == ("storage_unavailable", True)


def test_ffmpeg_failure_during_audio_is_no_audio(env):
    def broken(progress):
        raise RuntimeError(f"ffmpeg не смог извлечь аудио из /tmp/{SHA}.mp4: Invalid data found")

    assert _fail_with(env, broken)["code"] == "no_audio"


def test_empty_transcript_is_no_speech(env):
    env.run.utterances = []
    assert _fail_with(env)["code"] == "no_speech"


def test_model_download_attempt_is_model_unavailable(env):
    def offline(progress):
        progress(pipeline.Event("diarization", "[4/4] ..."))
        raise urllib.error.URLError("Temporary failure in name resolution")

    assert _fail_with(env, offline)["code"] == "model_unavailable"


def test_unexpected_error_is_internal(env):
    def bug(progress):
        progress(pipeline.Event("asr", "[3/4] ..."))
        raise KeyError("words")

    error = _fail_with(env, bug)
    assert error["code"] == "internal" and error["retryable"] is False


def test_logs_and_errors_never_carry_source_key(env, caplog):
    """Ключ исходника — фамилия и номер дела. Ни в логах, ни в тексте отказа его нет,
    даже когда чужое исключение несёт его в сообщении."""
    caplog.set_level("DEBUG")

    ok = _submit(env, names={"0": "Председатель"})
    _drain(env)

    def leaky(progress):
        progress(pipeline.Event("asr", "[3/4] ..."))
        raise RuntimeError(f"cannot read {SOURCE} (key {SOURCE_KEY}, name Петров_А40-12345-2026)")

    error = _fail_with(env, leaky, speakers=4)  # другие параметры — не попадание в кэш результата
    env.run.hook = None
    env.storage.fail[Location("records", SOURCE_KEY)] = AccessDenied("нет доступа к хранилищу", "AccessDenied")
    denied = _submit(env)
    _drain(env)

    assert _manifest(env, ok)["status"] == "done"
    assert ok in caplog.text  # логи действительно перехвачены
    assert "трассировка" in caplog.text  # и отладочная трассировка internal тоже
    for text in (caplog.text, error["message"], _manifest(env, denied)["error"]["message"]):
        assert "Петров" not in text
        assert "А40-12345" not in text
        assert "заседания/" not in text
    assert "<источник>" in error["message"]


# --- пульс, прогресс, аренда ---------------------------------------------------------


def test_heartbeat_runs_during_silent_stage(env):
    """Этап отпечатков молчит минутами: пульс должен идти без событий прогресса."""
    job_id = _submit(env)
    seen: list[float] = []

    def silent(progress):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            beat = env.store.get(job_id).heartbeat_at
            if not seen or beat != seen[-1]:
                seen.append(beat)
            if len(seen) >= 3:
                return
            time.sleep(0.01)
        raise AssertionError("пульс не шёл во время молчащего этапа")

    env.run.hook = silent
    _drain(env)

    assert _manifest(env, job_id)["status"] == "done"
    assert len(seen) >= 3


def test_progress_is_throttled_but_stage_changes_are_written(env, monkeypatch):
    now = [100.0]
    env.worker.clock = lambda: now[0]
    written: list[tuple[str, float | None]] = []
    original = env.store.progress

    def spy(job_id, lease, stage, fraction=None):
        written.append((stage, fraction))
        return original(job_id, lease, stage, fraction)

    monkeypatch.setattr(env.store, "progress", spy)

    def chatty(progress):
        for i in range(50):
            progress(pipeline.Event("asr", "...", i / 50))
        now[0] += 10  # прошло больше интервала
        progress(pipeline.Event("asr", "...", 0.99))

    env.run.hook = chatty
    _submit(env)
    _drain(env)

    asr = [f for stage, f in written if stage == "asr"]
    assert asr == [0.0, 0.99]
    assert [s for s, _ in written if s != "asr"] == ["prepare", "download", "cache", "audio", "cache", "upload"]


def test_lost_lease_stops_the_job_without_writing(env):
    job_id = _submit(env)
    job = env.store.claim_next()

    def requeued_meanwhile(progress):
        env.store.recover_stale(stale_after=-1)  # задачу сочли зависшей и вернули в очередь
        progress(pipeline.Event("asr", "..."))

    env.run.hook = requeued_meanwhile
    env.worker.process(job)

    assert env.store.get(job_id).status == "queued"
    assert not any(loc.bucket == "out" for loc in env.storage.objects)
    assert list(env.worker.workspace.path.iterdir()) == []


def test_stale_job_is_recovered_by_idle_loop_not_only_at_start(tmp_path, env):
    """После быстрого перезапуска пульс убитой задачи ещё свежий; её подберёт
    не стартовая проверка, а очередной круг цикла."""
    now = [1000.0]
    store = SqliteJobStore(tmp_path / "clocked.sqlite", clock=lambda: now[0])
    w = worker.Worker(env.settings, store, env.storage, whisper_model="held-model")
    job_id = store.create(SOURCE, JobRequest(source=SOURCE).params()).id
    store.claim_next()  # прежний воркер взял задачу и умер

    assert w.run_once() is False  # пульс свежий — не трогаем
    assert store.get(job_id).status == "running"

    now[0] += env.settings.stale_after + 1
    assert w.run_once() is True

    job = store.get(job_id)
    assert job.status == "done" and job.attempts == 2


# --- память ---------------------------------------------------------------------------


def test_nothing_large_survives_a_job(env, monkeypatch):
    """Ни результат прогона, ни кадры упавшего прогона (массивы на сотни мегабайт
    на настоящей записи) не доживают до следующей задачи."""
    refs: list[weakref.ref] = []
    fake = env.run

    def with_big_array(*args, **kwargs):
        result = fake(*args, **kwargs)
        big = np.zeros(2_000_000, dtype=np.float32)
        refs.append(weakref.ref(big))
        result.turns = [big]  # живёт внутри Result
        return result

    def explode(progress):
        frame_local = np.zeros(2_000_000, dtype=np.float32)
        refs.append(weakref.ref(frame_local))
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "run", with_big_array)
    _submit(env)
    _drain(env)
    fake.hook = explode
    _submit(env, speakers=5)
    _drain(env)

    assert len(refs) == 2
    assert [r() is None for r in refs] == [True, True]


# --- старт процесса ------------------------------------------------------------------


def test_sweep_removes_only_dead_workers_dirs(env):
    """Каталог работы может быть общим: удаляется только подкаталог воркера, чья
    блокировка свободна (процесс мёртв), а в нём — запись целиком."""
    dead = env.workdir / f"{worker.WORKSPACE_PREFIX}host-1-deadbeef"
    (dead / "job-x").mkdir(parents=True)
    (dead / "job-x" / f"{SHA}.mp4").write_bytes(DATA)
    (env.workdir / f"{dead.name}.lock").touch()  # файл блокировки есть, держателя нет
    neighbour = worker.Workspace(env.workdir)  # живой сосед со своей блокировкой
    (neighbour.path / "job-y").mkdir()
    unrelated = env.workdir / "чужой-файл.txt"
    unrelated.write_text("не трогать")

    removed = env.worker.workspace.sweep()

    assert removed == [dead]
    assert not dead.exists() and not (env.workdir / f"{dead.name}.lock").exists()
    assert (neighbour.path / "job-y").exists()
    assert env.worker.workspace.path.exists()  # свой не тронут
    assert unrelated.exists()
    neighbour.close()
    assert not neighbour.path.exists()


def test_loop_survives_database_error_and_takes_next_job(env, monkeypatch):
    """«database is locked» на захвате не должен ронять воркер целиком."""
    job_id = _submit(env)
    original = env.store.claim_next
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return original()

    monkeypatch.setattr(env.store, "claim_next", flaky)
    _drain(env)

    assert len(calls) >= 2
    assert env.store.get(job_id).status == "done"


def test_silence_is_no_speech(env):
    def silence(progress):
        progress(pipeline.Event("diarization", "[4/4] ..."))
        raise diarization.NoSpeech("В записи не найдено речи достаточной длительности")

    error = _fail_with(env, silence)
    assert (error["code"], error["retryable"]) == ("no_speech", False)


def test_stop_requeues_job_without_spending_attempt(env):
    """SIGTERM посреди здоровой задачи: вернуть её в очередь, попытку не списывать, выйти."""
    job_id = _submit(env)

    def sigterm(progress):
        env.worker.stopping.set()
        progress(pipeline.Event("asr", "[3/4] ..."))

    env.run.hook = sigterm
    env.worker.serve()  # без once: выходит сам — по остановке

    job = env.store.get(job_id)
    assert (job.status, job.attempts, job.lease) == ("queued", 0, None)
    assert not any(loc.bucket == "out" for loc in env.storage.objects)
    assert list(env.worker.workspace.path.iterdir()) == []


@pytest.fixture
def worker_env(tmp_path, monkeypatch, storage):
    monkeypatch.setenv("STENOGRAM_DB", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setenv("STENOGRAM_WORKDIR", str(tmp_path))
    monkeypatch.setenv("STENOGRAM_S3_ENDPOINT", "minio:9000")
    monkeypatch.setenv("STENOGRAM_CACHE_BUCKET", "cache")
    monkeypatch.setenv("STENOGRAM_WHISPER_MODEL", "small")
    monkeypatch.setenv("STENOGRAM_WHISPER_DIR", "/models/whisper")
    monkeypatch.setenv("STENOGRAM_THREADS", "0")
    loaded: list[tuple] = []
    monkeypatch.setattr(asr, "load_model", lambda *a, **k: loaded.append((a, k)) or "model")
    monkeypatch.setattr(MinioStorage, "connect", classmethod(lambda cls, *a, **k: storage))
    return loaded


def test_worker_exits_1_before_loading_when_models_missing(worker_env, monkeypatch, capsys):
    monkeypatch.setattr(models, "check_all", lambda **k: ["нет модели whisper small в /models/whisper"])

    assert worker.main(["--once"]) == 1

    assert worker_env == []  # модель не грузили
    assert "нет модели whisper small" in capsys.readouterr().err


def test_worker_exits_1_without_s3_settings(worker_env, monkeypatch, capsys):
    monkeypatch.setattr(models, "check_all", lambda **k: [])
    monkeypatch.delenv("STENOGRAM_CACHE_BUCKET")

    assert worker.main(["--once"]) == 1
    assert "STENOGRAM_CACHE_BUCKET" in capsys.readouterr().err


def test_worker_loads_one_model_offline_and_drains_queue(worker_env, monkeypatch, tmp_path):
    checked: list[dict] = []
    monkeypatch.setattr(models, "check_all", lambda **k: checked.append(k) or [])
    leftover = tmp_path / f"{worker.WORKSPACE_PREFIX}host-1-killed"
    leftover.mkdir()

    assert worker.main(["--once"]) == 0

    assert not leftover.exists()  # уборка после SIGKILL — при старте
    assert not list(tmp_path.glob(f"{worker.WORKSPACE_PREFIX}*"))  # свой каталог убран при выходе

    assert checked == [{"whisper_model": "small", "whisper_dir": "/models/whisper"}]
    [(args, kwargs)] = worker_env
    assert args == ("small",)
    assert kwargs["local_files_only"] is True and kwargs["download_root"] == "/models/whisper"
    assert kwargs["num_threads"] == Settings(threads=0).resolved_threads() >= 1  # не умолчание load_model (8)


def test_sigterm_through_main_requeues_and_exits(worker_env, monkeypatch, storage, tmp_path):
    """Настоящий сигнал: обработчик из main останавливает воркер, задача возвращается
    в очередь без списания попытки, процесс выходит с кодом 0."""
    monkeypatch.setattr(models, "check_all", lambda **k: [])
    monkeypatch.setattr(cache, "wav_to_flac", _copy)
    monkeypatch.setattr(cache, "flac_to_wav", _copy)
    run = FakeRun()

    def sigterm(progress):
        os.kill(os.getpid(), signal.SIGTERM)
        progress(pipeline.Event("asr", "[3/4] ..."))

    run.hook = sigterm
    monkeypatch.setattr(pipeline, "run", run)
    storage.objects[Location("records", SOURCE_KEY)] = DATA
    store = SqliteJobStore(tmp_path / "jobs.sqlite")
    job_id = store.create(SOURCE, JobRequest(source=SOURCE).params()).id
    handler_before = signal.getsignal(signal.SIGTERM)

    assert worker.main([]) == 0  # без --once: вышел сам, по сигналу

    job = store.get(job_id)
    assert (job.status, job.attempts) == ("queued", 0)
    assert signal.getsignal(signal.SIGTERM) is handler_before


def test_stop_after_pipeline_finishes_keeps_the_result(env, monkeypatch):
    """Остановка пришла на последнем событии прогона: готовый результат не выбрасывается —
    кэш, выгрузка и завершение доделываются, выход после."""
    job_id = _submit(env)
    fake = env.run

    def run_then_stop(*args, **kwargs):
        result = fake(*args, **kwargs)
        kwargs["progress"](pipeline.Event("align", "Собираю стенограмму ..."))  # последнее событие прогона
        env.worker.stopping.set()  # SIGTERM пришёл, пока шла сшивка: событий до конца run больше нет
        return result

    monkeypatch.setattr(pipeline, "run", run_then_stop)
    env.worker.serve()  # без once: выходит по остановке

    job = env.store.get(job_id)
    assert job.status == "done"
    assert env.worker.cache.get_result(job.result_key) is not None
    assert any(loc.bucket == "out" for loc in env.storage.objects)
