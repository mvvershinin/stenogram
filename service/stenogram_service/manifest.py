"""Манифест задачи — ответ `GET /jobs/{id}`.

Воркер при завершении кладёт в задачу то, что знает только он (результат, качество
разметки, кэш, время, свои настройки модели). Остальное — из строки задачи.
"""

from __future__ import annotations

import mimetypes
from typing import Any

import stenogram

from .config import Settings
from .jobstore import FAILED, RUNNING, Job
from .storage import InvalidURI, parse_uri

# Отказы, после которых имеет смысл отправить ту же задачу ещё раз.
RETRYABLE = frozenset({"storage_unavailable"})

# Модели разметки — отображаемые имена файлов stenogram.models.SEGMENTATION_FILE и EMBEDDING_FILE.
DIARIZATION_MODEL_NAMES = ("pyannote-segmentation-3.0", "CAM++")


def guess_mime(uri: str) -> str | None:
    try:
        return mimetypes.guess_type(parse_uri(uri).key)[0]
    except InvalidURI:
        return None


def decoder_info(settings: Settings, params: dict[str, Any]) -> dict[str, Any]:
    """Имена в `params` не входят: это таблица задачи, а не параметр декодера."""
    return {
        "name": "stenogram",
        "version": stenogram.__version__,
        "models": [settings.whisper_model, *DIARIZATION_MODEL_NAMES],
        "device": settings.device,
        "compute_type": settings.compute_type,
        "params": {k: v for k, v in params.items() if k != "names"},
    }


def build(job: Job, settings: Settings) -> dict[str, Any]:
    """Сохранённое воркером важнее настроек API: окружения процессов могут разойтись."""
    stored = job.manifest or {}
    manifest: dict[str, Any] = {"job_id": job.id, "status": job.status}
    if job.status == RUNNING:
        manifest["stage"] = job.stage
        manifest["fraction"] = job.fraction
    if job.status == FAILED:
        manifest["error"] = {"code": job.error_code, "message": job.error_message, "retryable": job.error_code in RETRYABLE}
    manifest["source"] = stored.get("source") or {"uri": job.source_uri, "sha256": job.sha256, "mime": guess_mime(job.source_uri)}
    manifest["decoder"] = stored.get("decoder") or decoder_info(settings, job.params)
    for key in ("result", "quality", "timing", "cache"):
        if key in stored:
            manifest[key] = stored[key]
    return manifest
