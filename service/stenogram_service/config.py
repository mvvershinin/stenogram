"""Настройки сервиса из переменных окружения.

Модель распознавания, устройство, compute_type и потоки — здесь, а не в задаче:
одна видеокарта, одна загруженная модель на процесс. API читает те же настройки,
чтобы `/healthz` проверял модели воркера. S3 нужен только воркеру.

ONNX-модели разметки stenogram ищет в `STENOGRAM_HOME` сам — отдельной настройки нет.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass

from stenogram import models


class ConfigError(ValueError):
    """Настройка не задана или не разбирается."""


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    if raw.lower() in {"1", "true", "yes", "on"}:
        return True
    if raw.lower() in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name}: ожидалось true/false, получено «{raw}»")


def _number(env: Mapping[str, str], name: str, default: float, kind=float):
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return kind(raw)
    except ValueError:
        raise ConfigError(f"{name}: ожидалось число, получено «{raw}»") from None


@dataclass(frozen=True)
class Settings:
    db_path: str = "jobs.sqlite"
    workdir: str = tempfile.gettempdir()

    s3_endpoint: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_secure: bool = True
    s3_region: str | None = None
    cache_bucket: str | None = None
    cache_prefix: str = "stenogram/cache/"
    result_bucket: str | None = None  # по умолчанию — бакет кэша
    result_prefix: str = "stenogram/jobs/"

    whisper_model: str = models.WHISPER_MODEL_DEFAULT
    whisper_dir: str | None = None
    local_files_only: bool = True  # закрытый контур: модели только с диска
    device: str = "cpu"
    compute_type: str = "int8"
    threads: int = 0  # 0 — все ядра, кроме двух, как в pipeline.run

    heartbeat_interval: float = 15.0
    stale_after: float = 120.0  # пульс старше — воркер считается пропавшим
    max_attempts: int = 2
    progress_interval: float = 5.0  # не чаще — запись этапа и доли в базу
    poll_interval: float = 2.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        d = cls()
        return cls(
            db_path=env.get("STENOGRAM_DB") or d.db_path,
            workdir=env.get("STENOGRAM_WORKDIR") or d.workdir,
            s3_endpoint=env.get("STENOGRAM_S3_ENDPOINT") or None,
            s3_access_key=env.get("STENOGRAM_S3_ACCESS_KEY") or None,
            s3_secret_key=env.get("STENOGRAM_S3_SECRET_KEY") or None,
            s3_secure=_bool(env, "STENOGRAM_S3_SECURE", d.s3_secure),
            s3_region=env.get("STENOGRAM_S3_REGION") or None,
            cache_bucket=env.get("STENOGRAM_CACHE_BUCKET") or None,
            cache_prefix=env.get("STENOGRAM_CACHE_PREFIX", d.cache_prefix),
            result_bucket=env.get("STENOGRAM_RESULT_BUCKET") or None,
            result_prefix=env.get("STENOGRAM_RESULT_PREFIX", d.result_prefix),
            whisper_model=env.get("STENOGRAM_WHISPER_MODEL") or d.whisper_model,
            whisper_dir=env.get("STENOGRAM_WHISPER_DIR") or None,
            local_files_only=_bool(env, "STENOGRAM_LOCAL_FILES_ONLY", d.local_files_only),
            device=env.get("STENOGRAM_DEVICE") or d.device,
            compute_type=env.get("STENOGRAM_COMPUTE_TYPE") or d.compute_type,
            threads=_number(env, "STENOGRAM_THREADS", d.threads, int),
            heartbeat_interval=_number(env, "STENOGRAM_HEARTBEAT", d.heartbeat_interval),
            stale_after=_number(env, "STENOGRAM_STALE_AFTER", d.stale_after),
            max_attempts=_number(env, "STENOGRAM_MAX_ATTEMPTS", d.max_attempts, int),
            progress_interval=_number(env, "STENOGRAM_PROGRESS_INTERVAL", d.progress_interval),
            poll_interval=_number(env, "STENOGRAM_POLL", d.poll_interval),
        )

    @property
    def results_bucket(self) -> str | None:
        return self.result_bucket or self.cache_bucket

    def worker_problems(self) -> list[str]:
        """Чего не хватает воркеру, кроме моделей."""
        problems = [f"не задана {name}" for name, value in (("STENOGRAM_S3_ENDPOINT", self.s3_endpoint), ("STENOGRAM_CACHE_BUCKET", self.cache_bucket)) if not value]
        if self.stale_after <= 2 * self.heartbeat_interval:
            problems.append("STENOGRAM_STALE_AFTER должен быть больше двух интервалов пульса, иначе живую задачу сочтут зависшей")
        return problems

    def resolved_threads(self) -> int:
        """Та же формула, что в pipeline.run: у asr.load_model своё умолчание (8)."""
        return self.threads or max(1, (os.cpu_count() or 4) - 2)
