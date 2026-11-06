"""Параметры задачи: одна модель для API (422 на входе) и воркера (перепроверка строки
из базы и перевод в `pipeline.Options`).

Модель, устройство, compute_type и потоки — настройки воркера, а не задачи: в теле
запроса их нет, и `extra="forbid"` превращает попытку их передать в 422, а не в молча
проигнорированное поле. Числа участников — строго целые: `3.7` иначе склеился бы
в ключе кэша с 3.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

from stenogram.pipeline import Options

from .storage import InvalidURI, parse_uri

_DEFAULTS = Options()

Count = Annotated[int, Field(strict=True, ge=1, le=100)]
TileCounts = Annotated[list[Count], Field(min_length=1, max_length=64)]
Lag = Annotated[float, Field(ge=0.0, le=10.0, allow_inf_nan=False)]  # NaN и Infinity json.loads пропускает


def _names_by_index(value: dict[str, str]) -> dict[str, str]:
    for key in value:
        if not re.fullmatch(r"0|[1-9][0-9]{0,2}", key):
            raise ValueError('ключи names — номера участников с нуля: {"0": "Иванов"}')
    return value


Names = Annotated[dict[str, Annotated[str, Field(min_length=1, max_length=200)]], AfterValidator(_names_by_index)]


class JobParams(BaseModel):
    """Всё, что задача передаёт пайплайну, кроме адреса исходника."""

    model_config = ConfigDict(extra="forbid")

    language: Annotated[str, Field(pattern=r"^[a-z]{2,3}$")] | None = _DEFAULTS.language
    prompt: Annotated[str, Field(max_length=2000)] | None = _DEFAULTS.prompt
    speakers: Count | None = _DEFAULTS.speakers
    tile_speakers: TileCounts | None = _DEFAULTS.tile_speakers
    max_speakers: Count = _DEFAULTS.max_speakers
    names: Names | None = None
    video: Annotated[bool, Field(strict=True)] = _DEFAULTS.video
    diarization: Annotated[bool, Field(strict=True)] = _DEFAULTS.diarization
    denoise: Annotated[bool, Field(strict=True)] = _DEFAULTS.denoise
    highlight_lag: Lag = _DEFAULTS.highlight_lag

    def names_by_index(self) -> dict[int, str]:
        return {int(k): v for k, v in (self.names or {}).items()}

    def options(self, *, model: str, device: str, compute_type: str, threads: int) -> Options:
        """Options для `pipeline.run`: параметры задачи плюс настройки воркера."""
        return Options(
            model=model,
            language=self.language,
            prompt=self.prompt,
            device=device,
            compute_type=compute_type,
            threads=threads,
            diarization=self.diarization,
            denoise=self.denoise,
            video=self.video,
            speakers=self.speakers,
            max_speakers=self.max_speakers,
            tile_speakers=None if self.tile_speakers is None else list(self.tile_speakers),
            highlight_lag=self.highlight_lag,
        )


class JobRequest(JobParams):
    source: str

    @field_validator("source")
    @classmethod
    def _s3_uri(cls, value: str) -> str:
        try:
            parse_uri(value)
        except InvalidURI as exc:
            raise ValueError(str(exc)) from None
        return value

    def params(self) -> dict[str, Any]:
        return self.model_dump(exclude={"source"})


class RerunRequest(BaseModel):
    """Что можно переопределить при перезапуске; остальное наследуется от исходной задачи.

    Поле, которого нет в теле, наследуется; явный `null` (у speakers, tile_speakers,
    names) сбрасывает к умолчанию — например, обратно к автооценке числа участников."""

    model_config = ConfigDict(extra="forbid")

    speakers: Count | None = None
    tile_speakers: TileCounts | None = None
    max_speakers: Count = _DEFAULTS.max_speakers
    names: Names | None = None
    video: Annotated[bool, Field(strict=True)] = _DEFAULTS.video
    highlight_lag: Lag = _DEFAULTS.highlight_lag

    def apply(self, inherited: dict[str, Any]) -> dict[str, Any]:
        """Параметры новой задачи. Результат перепроверяется целиком через JobParams."""
        merged = {**inherited, **self.model_dump(include=self.model_fields_set)}
        return JobParams.model_validate(merged).model_dump()
