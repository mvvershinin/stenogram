"""REST API: постановка задач, проверка параметров, манифест, перезапуск, здоровье."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stenogram_service.api import create_app
from stenogram_service.config import Settings
from stenogram_service.jobstore import SqliteJobStore

SOURCE = "s3://records/заседания/Петров_А40-12345-2026.mp4"


@pytest.fixture
def store(tmp_path) -> SqliteJobStore:
    return SqliteJobStore(tmp_path / "jobs.sqlite")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(db_path=str(tmp_path / "jobs.sqlite"), whisper_model="small", device="cuda", compute_type="float16")


@pytest.fixture
def client(settings, store) -> TestClient:
    return TestClient(create_app(settings, store=store, check_models=lambda: []))


def test_post_job_queues_and_manifest_shows_it(client, store):
    response = client.post("/jobs", json={"source": SOURCE, "language": "ru", "speakers": 3, "names": {"0": "Председатель"}})

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    job = store.get(job_id)
    assert job.source_uri == SOURCE
    assert job.params["speakers"] == 3 and job.params["names"] == {"0": "Председатель"}
    assert job.params["highlight_lag"] == 1.2 and job.params["video"] is True  # умолчания Options

    manifest = client.get(f"/jobs/{job_id}").json()
    assert manifest["status"] == "queued"
    assert manifest["source"] == {"uri": SOURCE, "sha256": None, "mime": "video/mp4"}
    decoder = manifest["decoder"]
    assert decoder["name"] == "stenogram" and decoder["models"] == ["small", "pyannote-segmentation-3.0", "CAM++"]
    assert (decoder["device"], decoder["compute_type"]) == ("cuda", "float16")
    assert "names" not in decoder["params"]


@pytest.mark.parametrize(
    "extra",
    [
        {"speakers": 3.7},
        {"speakers": 3.0},  # целое, но дробной записью: строго int
        {"speakers": 0},
        {"speakers": -1},
        {"speakers": True},
        {"speakers": "3"},
        {"max_speakers": 0},
        {"max_speakers": 2.5},
        {"tile_speakers": []},
        {"tile_speakers": [2, 0]},
        {"tile_speakers": [2, 1.5]},
        {"names": {"первый": "Иванов"}},
        {"names": {"-1": "Иванов"}},
        {"highlight_lag": -0.5},
        {"language": "Russian"},
        {"model": "small"},  # настройка воркера, не задачи
        {"device": "cuda"},
        {"source": "https://records/x.mp4"},
        {"source": "s3://records"},
    ],
)
def test_invalid_params_are_rejected_with_422(client, store, extra):
    response = client.post("/jobs", json={"source": SOURCE, **extra})

    assert response.status_code == 422
    assert store.claim_next() is None


def test_nan_lag_is_rejected(client):
    """json.loads принимает NaN; в ключе кэша ему не место."""
    body = f'{{"source": "{SOURCE}", "highlight_lag": NaN}}'
    response = client.post("/jobs", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_unknown_job_is_404(client):
    assert client.get("/jobs/нет-такой").status_code == 404
    assert client.post("/jobs/нет-такой/rerun", json={}).status_code == 404


def _job(client) -> str:
    return client.post("/jobs", json={"source": SOURCE, "speakers": 4, "tile_speakers": [3, 1], "language": "ru", "prompt": "Петров, Сидорова"}).json()["job_id"]


def test_running_manifest_shows_stage_and_fraction(client, store):
    job_id = _job(client)
    claimed = store.claim_next()
    store.progress(job_id, claimed.lease, "asr", 0.4)

    manifest = client.get(f"/jobs/{job_id}").json()

    assert (manifest["status"], manifest["stage"], manifest["fraction"]) == ("running", "asr", 0.4)


def test_done_manifest_carries_worker_fields(client, store):
    job_id = _job(client)
    claimed = store.claim_next()
    stored = {
        "source": {"uri": SOURCE, "sha256": "a" * 64, "mime": "video/mp4"},
        "decoder": {"name": "stenogram", "version": "1.0.0", "models": ["large-v3-turbo", "pyannote-segmentation-3.0", "CAM++"], "params": {}},
        "result": {"blocks": "s3://out/j/blocks.json", "transcript": "s3://out/j/transcript.txt"},
        "quality": {"speaker_source": "видео + голос", "tiles": 2, "speakers": 4, "degraded": False},
        "timing": {"seconds": 12.5},
        "cache": "asr-hit",
    }
    store.complete(job_id, claimed.lease, stored)

    manifest = client.get(f"/jobs/{job_id}").json()

    assert manifest["status"] == "done"
    for key, value in stored.items():
        assert manifest[key] == value  # сохранённое воркером важнее настроек API
    assert "error" not in manifest and "stage" not in manifest


@pytest.mark.parametrize(("code", "retryable"), [("storage_unavailable", True), ("invalid_params", False), ("source_not_found", False), ("worker_lost", False)])
def test_failed_manifest_has_error_with_retryable(client, store, code, retryable):
    job_id = _job(client)
    claimed = store.claim_next()
    store.fail(job_id, claimed.lease, code, "текст")

    error = client.get(f"/jobs/{job_id}").json()["error"]

    assert error == {"code": code, "message": "текст", "retryable": retryable}


def test_rerun_inherits_params_and_sha256(client, store):
    parent_id = _job(client)
    claimed = store.claim_next()
    store.set_keys(parent_id, claimed.lease, sha256="b" * 64, recognition_key="r", result_key="k")

    response = client.post(f"/jobs/{parent_id}/rerun", json={"speakers": 5, "names": {"1": "Сидорова"}})

    assert response.status_code == 202
    assert response.json()["rerun_of"] == parent_id
    child = store.get(response.json()["job_id"])
    parent = store.get(parent_id)
    assert child.source_uri == parent.source_uri
    assert child.sha256 == "b" * 64  # воркер сможет не качать исходник
    assert child.params == {**parent.params, "speakers": 5, "names": {"1": "Сидорова"}}
    assert child.status == "queued"


def test_rerun_explicit_null_resets_and_absent_inherits(client, store):
    parent_id = _job(client)

    child_id = client.post(f"/jobs/{parent_id}/rerun", json={"speakers": None}).json()["job_id"]

    child = store.get(child_id)
    assert child.params["speakers"] is None  # обратно к автооценке
    assert child.params["tile_speakers"] == [3, 1]  # не передано — унаследовано
    assert child.params["prompt"] == "Петров, Сидорова"


def test_rerun_without_body_copies_params(client, store):
    parent_id = _job(client)
    child_id = client.post(f"/jobs/{parent_id}/rerun").json()["job_id"]
    assert store.get(child_id).params == store.get(parent_id).params


@pytest.mark.parametrize("override", [{"speakers": 2.5}, {"tile_speakers": []}, {"language": "en"}, {"source": "s3://other/x.mp4"}])
def test_rerun_rejects_bad_or_foreign_overrides(client, override):
    """Язык и исходник у перезапуска не меняются: это другая задача, а не перезапуск."""
    parent_id = _job(client)
    assert client.post(f"/jobs/{parent_id}/rerun", json=override).status_code == 422


def test_healthz_ok(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_healthz_503_lists_missing_models(settings, store):
    problems = ["нет модели wespeaker_en_voxceleb_CAM++.onnx в /models", "нет модели whisper small в /hf"]
    client = TestClient(create_app(settings, store=store, check_models=lambda: problems))

    response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "problems": problems}


def test_healthz_503_when_database_is_gone(settings, store, tmp_path):
    client = TestClient(create_app(settings, store=store, check_models=lambda: []))
    store.path = tmp_path / "нет-каталога" / "jobs.sqlite"

    response = client.get("/healthz")

    assert response.status_code == 503
    assert "база задач недоступна" in response.json()["problems"][0]


def test_422_does_not_echo_request_body(client):
    """Стандартный ответ FastAPI повторяет `input` — вместе с адресом исходника."""
    response = client.post("/jobs", json={"source": SOURCE, "speakers": 0, "prompt": "Петров, Сидорова"})

    assert response.status_code == 422
    assert "Петров" not in response.text
    assert response.json()["detail"][0]["loc"] == ["body", "speakers"]
