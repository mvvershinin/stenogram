"""REST API: постановка задач, манифест, перезапуск, здоровье.

    uvicorn --factory stenogram_service.api:create_app
    python -m stenogram_service.api --host 0.0.0.0 --port 8000

Фабрика, а не готовый `app` на уровне модуля: иначе импорт модуля открывал бы базу.
Обработчики синхронные — SQLite и проверка моделей уходят в пул потоков.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

import stenogram
from stenogram import models

from . import manifest
from .config import Settings
from .jobstore import JobStore, SqliteJobStore
from .params import JobRequest, RerunRequest

log = logging.getLogger("stenogram_service.api")


def create_app(settings: Settings | None = None, *, store: JobStore | None = None, check_models: Callable[[], list[str]] | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or SqliteJobStore(Path(settings.db_path))
    if check_models is None:

        def check_models() -> list[str]:
            return models.check_all(whisper_model=settings.whisper_model, whisper_dir=settings.whisper_dir)

    app = FastAPI(title="stenogram", version=stenogram.__version__)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc: RequestValidationError) -> JSONResponse:
        """422 без эха входных данных: стандартный ответ повторяет `input`, а NaN
        из тела (json.loads его пропускает) роняет сериализацию ответа в 500."""
        errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.post("/jobs", status_code=202)
    def create_job(request: JobRequest) -> dict:
        job = store.create(request.source, request.params())
        log.info("job=%s поставлена в очередь", job.id)
        return {"job_id": job.id}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="задача не найдена")
        return manifest.build(job, settings)

    @app.post("/jobs/{job_id}/rerun", status_code=202)
    def rerun_job(job_id: str, request: RerunRequest | None = None) -> dict:
        """Новая задача по тому же исходнику. sha256 родителя идёт в задачу: при попадании
        в кэш распознавания воркер не скачивает исходник вовсе — и поэтому не проверяет,
        что по адресу лежит всё тот же файл и что он вообще ещё существует."""
        parent = store.get(job_id)
        if parent is None:
            raise HTTPException(status_code=404, detail="задача не найдена")
        try:
            params = (request or RerunRequest()).apply(parent.params)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from None
        job = store.create(parent.source_uri, params, sha256=parent.sha256)
        log.info("job=%s перезапуск задачи %s", job.id, parent.id)
        return {"job_id": job.id, "rerun_of": parent.id}

    @app.get("/healthz")
    def healthz():
        problems: list[str] = []
        try:
            store.get("healthz")
        except Exception as exc:
            problems.append(f"база задач недоступна: {type(exc).__name__}")
        try:
            problems += check_models()
        except Exception as exc:
            problems.append(f"проверка моделей упала: {type(exc).__name__}")
        if problems:
            return JSONResponse(status_code=503, content={"status": "unavailable", "problems": problems})
        return {"status": "ok"}

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn  # noqa: PLC0415 -- нужен только для запуска сервера, не для тестов и фабрики

    parser = argparse.ArgumentParser(prog="python -m stenogram_service.api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run("stenogram_service.api:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
