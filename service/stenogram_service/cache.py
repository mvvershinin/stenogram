"""Кэш в S3: ключи и чтение/запись.

Ключ двухуровневый. Ключ распознавания — всё, от чего зависят текст, плитки и звук;
по нему лежат промежуточные результаты, и перезапуск с другими `speakers` обходится
без получаса распознавания. Ключ результата — ключ распознавания плюс всё, от чего
зависит разметка; по нему лежит готовый документ. `names` не входят никуда: имена —
таблица в манифесте задачи и в текст блоков не вшиваются.

Раскладка (префикс — настройка):

    <prefix>asr/<ключ распознавания>/audio16k.flac
    <prefix>asr/<ключ распознавания>/intermediate.json   — пишется последним: метка «запись целая»
    <prefix>result/<ключ результата>/transcript.txt
    <prefix>result/<ключ результата>/blocks.json         — пишется последним

Звук хранится во FLAC: без потерь и вдвое меньше WAV, а сжатие часа — секунды на фоне
получаса распознавания, которые кэш экономит.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import stenogram
from stenogram import audio, models
from stenogram.pipeline import Intermediate, Options

from .storage import Location, ObjectNotFound, Storage

log = logging.getLogger(__name__)

# Меняется, когда меняется смысл полей ключа или формат лежащего по нему.
KEY_SCHEMA = 1

# Модели разметки участников — по именам файлов, под которыми их знает stenogram.models.
DIARIZATION_MODELS = (models.SEGMENTATION_FILE, models.EMBEDDING_FILE)


def _digest(payload: dict[str, Any]) -> str:
    """sha256 канонического JSON: порядок ключей и пробелы на хэш не влияют."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Движки распознавания: смена их версии меняет текст при тех же параметрах. Версии
# берутся из метаданных пакетов — сами библиотеки грузить ради ключа незачем.
BACKENDS = ("faster-whisper", "ctranslate2")


def backend_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in BACKENDS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def recognition_key(
    *,
    sha256: str,
    model: str,
    device: str,
    compute_type: str,
    language: str | None,
    prompt: str | None,
    beam_size: int,
    denoise: bool,
    stenogram_version: str = stenogram.__version__,
    backends: dict[str, str | None] | None = None,
) -> str:
    """`device` — в ключе: на процессоре и видеокарте текст считают разные ядра,
    и побайтного совпадения результата никто не обещает."""
    return _digest(
        {
            "kind": "recognition",
            "schema": KEY_SCHEMA,
            "sha256": sha256,
            "stenogram": stenogram_version,
            "backends": backend_versions() if backends is None else backends,
            "model": model,
            "device": device,
            "compute_type": compute_type,
            "language": language,
            "prompt": prompt,
            "beam_size": int(beam_size),
            "denoise": bool(denoise),
        }
    )


def result_key(
    *,
    recognition: str,
    speakers: int | None,
    tile_speakers: list[int] | None,
    max_speakers: int,
    video: bool,
    diarization: bool,
    highlight_lag: float,
    diarization_models: tuple[str, ...] = DIARIZATION_MODELS,
) -> str:
    """Числа приводятся к int: `3` из JSON и `3.0` или numpy-число — один ключ.
    `max_speakers` — только потолок автооценки: при заданном `speakers` его нет в ключе."""
    speakers = None if speakers is None else int(speakers)
    return _digest(
        {
            "kind": "result",
            "schema": KEY_SCHEMA,
            "recognition": recognition,
            "diarization_models": list(diarization_models),
            "speakers": speakers,
            "tile_speakers": None if tile_speakers is None else [int(n) for n in tile_speakers],
            "max_speakers": int(max_speakers) if speakers is None else None,
            "video": bool(video),
            "diarization": bool(diarization),
            # Сдвиг подсветки — вход video.to_turns: другой lag, другая разметка.
            "highlight_lag": float(highlight_lag),
        }
    )


def keys_for(sha256: str, options: Options) -> tuple[str, str]:
    """(ключ распознавания, ключ результата) для прогона `pipeline.run` с этими `options`.

    Модель, device и compute_type — настройки воркера, но в `options` они уже подставлены."""
    rkey = recognition_key(
        sha256=sha256,
        model=options.model,
        device=options.device,
        compute_type=options.compute_type,
        language=options.language,
        prompt=options.prompt,
        beam_size=options.beam_size,
        denoise=options.denoise,
    )
    return rkey, result_key(
        recognition=rkey,
        speakers=options.speakers,
        tile_speakers=options.tile_speakers,
        max_speakers=options.max_speakers,
        video=options.video,
        diarization=options.diarization,
        highlight_lag=options.highlight_lag,
    )


def wav_to_flac(wav: Path, flac: Path) -> None:
    """Без метаданных: ffmpeg по умолчанию переносит теги исходника (title часто и есть
    фамилия с номером дела) в WAV, а из него — в FLAC."""
    ffmpeg, _ = audio.find_ffmpeg()
    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(wav), "-map_metadata", "-1", "-fflags", "+bitexact", "-c:a", "flac", "-sample_fmt", "s16", str(flac)]
    _ffmpeg(cmd)


def flac_to_wav(flac: Path, wav: Path) -> None:
    ffmpeg, _ = audio.find_ffmpeg()
    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(flac), "-map_metadata", "-1", "-fflags", "+bitexact", "-ac", "1", "-ar", str(audio.SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav)]
    _ffmpeg(cmd)


def _ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Пути здесь — временные файлы под хэшами, настоящих имён в них нет.
        raise RuntimeError(f"ffmpeg завершился с кодом {proc.returncode}: {proc.stderr.strip()[-400:]}")


@dataclass(frozen=True)
class Cache:
    storage: Storage
    bucket: str
    prefix: str = "stenogram/cache/"

    def _loc(self, *parts: str) -> Location:
        return Location(self.bucket, self.prefix + "/".join(parts))

    def audio_location(self, rkey: str) -> Location:
        return self._loc("asr", rkey, "audio16k.flac")

    def intermediate_location(self, rkey: str) -> Location:
        return self._loc("asr", rkey, "intermediate.json")

    def blocks_location(self, reskey: str) -> Location:
        return self._loc("result", reskey, "blocks.json")

    def text_location(self, reskey: str) -> Location:
        return self._loc("result", reskey, "transcript.txt")

    def _read_json(self, loc: Location) -> Any | None:
        try:
            return self.storage.read_json(loc)
        except ObjectNotFound:
            return None
        except ValueError as exc:  # JSONDecodeError и UnicodeDecodeError — оба ValueError
            log.warning("объект кэша %s не читается (%s) — считаю промахом", loc.key, type(exc).__name__)
            return None

    def get_intermediate(self, rkey: str, wav: Path) -> Intermediate | None:
        """Промежуточные результаты и звук в `wav`; нет или испорчено — None (промах).

        Ошибки сети и доступа всплывают: их нельзя путать с промахом, иначе сбой S3
        молча превращается в получас распознавания."""
        data = self._read_json(self.intermediate_location(rkey))
        if data is None:
            return None
        try:
            intermediate = Intermediate.from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("кэш распознавания %s не читается (%s) — считаю промахом", rkey, type(exc).__name__)
            return None
        with tempfile.TemporaryDirectory(dir=wav.parent) as tmp:
            flac = Path(tmp) / "audio16k.flac"
            try:
                self.storage.download(self.audio_location(rkey), flac)
            except ObjectNotFound:
                log.warning("в кэше распознавания %s нет звука — считаю промахом", rkey)
                return None
            try:
                flac_to_wav(flac, wav)
            except RuntimeError:
                # Испорченный объект иначе ронял бы каждую задачу по этой записи; пересчёт
                # его перезапишет. Если сломан сам ffmpeg, упадёт и извлечение звука.
                log.warning("звук в кэше распознавания %s не декодируется — считаю промахом", rkey)
                wav.unlink(missing_ok=True)
                return None
        return intermediate

    def put_intermediate(self, rkey: str, intermediate: Intermediate, wav: Path | None) -> None:
        """Звук — первым, JSON — последним: читатель верит только целой записи.

        `wav=None` — звук уже лежит (обновляются плитки после повтора с видео).

        Упавший поиск плиток в кэш не идёт: отказ бывает временным (память, битый
        кусок чтения), а `tiles == []` с причиной запретил бы искать снова навсегда.
        Пишется «плитки не искали» — следующий прогон с видео поищет заново."""
        if intermediate.video_error is not None:
            intermediate = dataclasses.replace(intermediate, tiles=None, labels=None, step=None, video_error=None)
        if wav is not None:
            with tempfile.TemporaryDirectory(dir=wav.parent) as tmp:
                flac = Path(tmp) / "audio16k.flac"
                wav_to_flac(wav, flac)
                self.storage.put_file(self.audio_location(rkey), flac, "audio/flac")
        self.storage.put_json(self.intermediate_location(rkey), intermediate.to_dict())

    def get_result(self, reskey: str) -> dict[str, Any] | None:
        """Документ blocks.json из кэша — без имён и без source_uri, их дописывает задача."""
        return self._read_json(self.blocks_location(reskey))

    def get_text(self, reskey: str) -> str | None:
        try:
            return self.storage.read_bytes(self.text_location(reskey)).decode("utf-8")
        except (ObjectNotFound, UnicodeDecodeError):
            return None

    def put_result(self, reskey: str, document: dict[str, Any], text: str, *, video_error: str | None) -> bool:
        """Кладёт результат; False — не положил.

        Документ должен быть обезличен: одинаковый файл под разными адресами и с разными
        именами даёт один ключ результата. Результат, где разметка по видео откатилась
        на голос (`Result.video_error`), не кэшируется: тот же ключ после временного
        отказа должен дать разметку по видео, а не навсегда закреплённую голосовую."""
        if document.get("source_uri") is not None or document.get("speakers"):
            raise ValueError("в кэш результата кладётся документ без source_uri и имён")
        if video_error is not None:
            return False
        self.storage.put_bytes(self.text_location(reskey), text.encode("utf-8"), "text/plain; charset=utf-8")
        self.storage.put_json(self.blocks_location(reskey), document)
        return True
