"""Кэш: ключи и чтение/запись через хранилище.

Что здесь держится:
  - ключ распознавания не зависит от параметров разметки, ключ результата — от имён;
    всё, что меняет текст или разметку, меняет и свой ключ;
  - промах — только «объекта нет» или «не читается»; сбой сети наружу, а не в промах;
  - метка целостности (intermediate.json, blocks.json) пишется последней;
  - звук в кэше без потерь и без метаданных исходника.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import subprocess
import sys

import numpy as np
import pytest
from fakes import write_tone

import stenogram
from stenogram import audio, models, pipeline
from stenogram.asr import Segment, Word
from stenogram.video import Tile
from stenogram_service import blocks, cache
from stenogram_service.storage import Location, StorageUnavailable

SHA = "0" * 64
BACKENDS = {"faster-whisper": "1.1.0", "ctranslate2": "4.5.0"}
BASE = {
    "sha256": SHA,
    "model": "large-v3-turbo",
    "device": "cpu",
    "compute_type": "int8",
    "language": "ru",
    "prompt": None,
    "beam_size": 5,
    "denoise": False,
    "backends": BACKENDS,
}
RESULT = {"speakers": None, "tile_speakers": None, "max_speakers": 12, "video": True, "diarization": True, "highlight_lag": 1.2}


# --- ключи --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "other"),
    [
        ("sha256", "1" * 64),
        ("model", "small"),
        ("device", "cuda"),
        ("compute_type", "float16"),
        ("language", None),
        ("prompt", "Протокол"),
        ("beam_size", 1),
        ("denoise", True),
        ("backends", {**BACKENDS, "faster-whisper": "1.2.0"}),
        ("backends", {**BACKENDS, "ctranslate2": "4.6.0"}),
    ],
)
def test_recognition_key_changes_with_every_recognition_input(field, other):
    assert cache.recognition_key(**BASE) != cache.recognition_key(**{**BASE, field: other})


def test_recognition_key_changes_with_library_version():
    assert cache.recognition_key(**BASE) != cache.recognition_key(**BASE, stenogram_version="999.0")
    assert cache.recognition_key(**BASE) == cache.recognition_key(**BASE, stenogram_version=stenogram.__version__)


def test_backend_versions_come_from_metadata_without_importing(monkeypatch):
    """Ключ считается и в процессе API — грузить ради него CTranslate2 незачем."""
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    monkeypatch.delitem(sys.modules, "ctranslate2", raising=False)

    versions = cache.backend_versions()

    assert set(versions) == {"faster-whisper", "ctranslate2"}
    assert versions["faster-whisper"] == importlib.metadata.version("faster-whisper")
    assert "faster_whisper" not in sys.modules and "ctranslate2" not in sys.modules
    assert cache.recognition_key(**{**BASE, "backends": None}) == cache.recognition_key(**{**BASE, "backends": versions})


@pytest.mark.parametrize(
    ("field", "other"),
    [("speakers", 3), ("tile_speakers", [2, 1]), ("max_speakers", 8), ("video", False), ("diarization", False), ("highlight_lag", 0.8)],
)
def test_result_key_changes_with_every_diarization_input(field, other):
    rkey = cache.recognition_key(**BASE)
    assert cache.result_key(recognition=rkey, **RESULT) != cache.result_key(recognition=rkey, **{**RESULT, field: other})


def test_result_key_normalizes_numbers():
    """3 из JSON, 3.0 и numpy-число — одна и та же задача."""
    rkey = cache.recognition_key(**BASE)
    key = cache.result_key(recognition=rkey, **{**RESULT, "speakers": 3, "tile_speakers": [2, 1]})
    assert key == cache.result_key(recognition=rkey, **{**RESULT, "speakers": 3.0, "tile_speakers": [2.0, 1]})
    assert key == cache.result_key(recognition=rkey, **{**RESULT, "speakers": np.int64(3), "tile_speakers": [np.int64(2), np.int32(1)]})


def test_max_speakers_matters_only_without_speakers():
    """Потолок автооценки не используется, когда число участников задано явно."""
    rkey = cache.recognition_key(**BASE)
    fixed = {**RESULT, "speakers": 4}
    assert cache.result_key(recognition=rkey, **{**fixed, "max_speakers": 12}) == cache.result_key(recognition=rkey, **{**fixed, "max_speakers": 6})
    assert cache.result_key(recognition=rkey, **{**RESULT, "max_speakers": 12}) != cache.result_key(recognition=rkey, **{**RESULT, "max_speakers": 6})


def test_result_key_depends_on_recognition_and_diarization_models():
    rkey = cache.recognition_key(**BASE)
    key = cache.result_key(recognition=rkey, **RESULT)
    assert key != cache.result_key(recognition=cache.recognition_key(**{**BASE, "beam_size": 1}), **RESULT)
    assert key != cache.result_key(recognition=rkey, **RESULT, diarization_models=("другая-сегментация.onnx", models.EMBEDDING_FILE))
    assert cache.DIARIZATION_MODELS == (models.SEGMENTATION_FILE, models.EMBEDDING_FILE)


def test_keys_for_options_ignore_names_and_threads():
    """Число потоков на результат не влияет и кэш не дробит; имён в Options вообще нет."""
    base = pipeline.Options(language="ru")
    same = dataclasses.replace(base, threads=3)
    other_speakers = dataclasses.replace(base, speakers=4)

    rkey, reskey = cache.keys_for(SHA, base)

    assert cache.keys_for(SHA, same) == (rkey, reskey)
    assert cache.keys_for(SHA, dataclasses.replace(base, device="cuda"))[0] != rkey
    assert cache.keys_for(SHA, other_speakers)[0] == rkey  # распознавание переиспользуется
    assert cache.keys_for(SHA, other_speakers)[1] != reskey
    assert rkey == cache.recognition_key(**{**BASE, "backends": None})
    assert reskey == cache.result_key(recognition=rkey, **RESULT)


def test_keys_are_stable_literal():
    # Зафиксированное значение: случайная смена канонизации (порядок, пробелы, ensure_ascii)
    # молча инвалидирует весь кэш. Меняется только вместе с KEY_SCHEMA.
    assert cache.recognition_key(**BASE, stenogram_version="1.0.0") == "d0b9f24987fbde76a541f1074d2588c99ac87ee933478ae34f8b2d365a6b62eb"


# --- промежуточные результаты ---------------------------------------------------


def _intermediate() -> pipeline.Intermediate:
    words = [Word(0.5, 0.9, " Добрый"), Word(1.0, 1.4, " день")]
    return pipeline.Intermediate([Segment(0.5, 1.4, "Добрый день", words)], "ru", [Tile(10, 0, 3, 20)], np.array([0, 0, -1]), 0.2, None)


@pytest.fixture
def store(storage) -> cache.Cache:
    return cache.Cache(storage, "cache-bucket", "svc/")


@pytest.mark.ffmpeg
def test_intermediate_round_trip_is_lossless(tmp_path, storage, store):
    wav = write_tone(tmp_path / "audio16k.wav")
    store.put_intermediate("r1", _intermediate(), wav)

    assert storage.writes == [Location("cache-bucket", "svc/asr/r1/audio16k.flac"), Location("cache-bucket", "svc/asr/r1/intermediate.json")]
    assert storage.content_types[storage.writes[0]] == "audio/flac"
    assert len(storage.objects[storage.writes[0]]) < wav.stat().st_size

    restored_wav = tmp_path / "restored.wav"
    restored = store.get_intermediate("r1", restored_wav)

    assert restored.segments == _intermediate().segments
    assert restored.tiles == _intermediate().tiles
    assert np.array_equal(restored.labels, _intermediate().labels)
    assert np.array_equal(audio.load_wav(restored_wav), audio.load_wav(wav))  # отсчёт в отсчёт
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audio16k.wav", "restored.wav"]  # временные убраны


@pytest.mark.ffmpeg
def test_cached_audio_carries_no_source_metadata(tmp_path, storage, store):
    """extract_wav переносит теги исходника в WAV; в кэш они попасть не должны."""
    ffmpeg, ffprobe = audio.find_ffmpeg()
    tone = write_tone(tmp_path / "tone.wav")
    tagged = tmp_path / "tagged.wav"
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(tone), "-metadata", "title=Петров А40-12345", "-c:a", "pcm_s16le", str(tagged)], check=True)
    assert "Петров" in subprocess.run([ffprobe, "-v", "error", "-show_format", str(tagged)], capture_output=True, text=True).stdout

    store.put_intermediate("r1", _intermediate(), tagged)
    flac = tmp_path / "cached.flac"
    flac.write_bytes(storage.objects[store.audio_location("r1")])

    probe = subprocess.run([ffprobe, "-v", "error", "-show_format", "-show_streams", str(flac)], capture_output=True, text=True, check=True).stdout
    assert "Петров" not in probe
    assert "А40" not in probe


def test_missing_intermediate_is_a_miss(tmp_path, store):
    assert store.get_intermediate("нет", tmp_path / "a.wav") is None


def test_intermediate_without_audio_is_a_miss(tmp_path, storage, store):
    """JSON есть, звука нет (чужая частичная запись) — переиспользовать нечего."""
    storage.put_json(store.intermediate_location("r1"), _intermediate().to_dict())

    assert store.get_intermediate("r1", tmp_path / "a.wav") is None
    assert not (tmp_path / "a.wav").exists()


@pytest.mark.parametrize("broken", [{"segments": []}, {"segments": "x", "language": 1, "tiles": 5, "labels": None, "step": None, "video_error": None}])
def test_unreadable_intermediate_is_a_miss(tmp_path, storage, store, broken):
    storage.put_json(store.intermediate_location("r1"), broken)

    assert store.get_intermediate("r1", tmp_path / "a.wav") is None


def test_network_failure_is_not_a_miss(tmp_path, storage, store):
    """Иначе сбой S3 молча стоит получаса распознавания."""
    storage.fail[store.intermediate_location("r1")] = StorageUnavailable("нет связи с хранилищем")

    with pytest.raises(StorageUnavailable):
        store.get_intermediate("r1", tmp_path / "a.wav")


def test_intermediate_update_without_audio_keeps_audio(storage, store):
    """Повтор с видео дописал плитки: звук тот же, перезаливать его незачем."""
    storage.put_bytes(store.audio_location("r1"), b"flac")
    store.put_intermediate("r1", _intermediate(), None)

    assert storage.writes == [store.audio_location("r1"), store.intermediate_location("r1")]
    assert json.loads(storage.objects[store.intermediate_location("r1")])["tiles"] == [[10, 0, 3, 20]]


# --- результат ------------------------------------------------------------------


def test_result_round_trip_writes_blocks_last(storage, store):
    doc = {"schema": blocks.SCHEMA, "doc_id": SHA, "source_uri": None, "speakers": {}, "blocks": []}

    assert store.put_result("k1", doc, "[00:00:00] Участник 1: Добрый день\n", video_error=None)

    assert storage.writes == [store.text_location("k1"), store.blocks_location("k1")]
    assert store.get_result("k1") == doc
    assert store.get_text("k1") == "[00:00:00] Участник 1: Добрый день\n"
    assert store.get_result("нет") is None and store.get_text("нет") is None


@pytest.mark.parametrize("personal", [{"source_uri": "s3://records/x.mp4", "speakers": {}}, {"source_uri": None, "speakers": {"0": "Иванов"}}])
def test_result_cache_refuses_per_job_fields(storage, store, personal):
    """Один ключ результата — у разных задач с разными адресами и именами."""
    doc = {"schema": blocks.SCHEMA, "doc_id": SHA, "blocks": [], **personal}

    with pytest.raises(ValueError, match="без source_uri и имён"):
        store.put_result("k1", doc, "", video_error=None)
    assert storage.writes == []


@pytest.mark.ffmpeg
def test_corrupt_cached_audio_is_a_miss(tmp_path, storage, store):
    storage.put_json(store.intermediate_location("r1"), _intermediate().to_dict())
    storage.put_bytes(store.audio_location("r1"), b"not a flac at all")

    assert store.get_intermediate("r1", tmp_path / "a.wav") is None
    assert not (tmp_path / "a.wav").exists()


def test_video_fallback_result_is_not_cached(storage, store):
    """Разметка откатилась на голос из-за временного отказа видео — тот же ключ
    в следующий раз должен получить шанс на разметку по видео."""
    doc = {"schema": blocks.SCHEMA, "doc_id": SHA, "source_uri": None, "speakers": {}, "blocks": []}

    assert store.put_result("k1", doc, "текст", video_error="MemoryError") is False
    assert storage.writes == []
    assert store.get_result("k1") is None


def test_failed_tile_search_is_cached_as_not_searched(storage, store):
    """tiles == [] с причиной запретили бы искать плитки снова навсегда."""
    failed = dataclasses.replace(_intermediate(), tiles=[], labels=None, step=None, video_error="MemoryError")

    store.put_intermediate("r1", failed, None)

    cached = pipeline.Intermediate.from_dict(json.loads(storage.objects[store.intermediate_location("r1")]))
    assert (cached.tiles, cached.labels, cached.step, cached.video_error) == (None, None, None, None)
    assert cached.segments == failed.segments  # распознавание при этом сохранено
    assert failed.video_error == "MemoryError"  # объект вызывающего не тронут


def test_found_nothing_is_cached_as_is(storage, store):
    """Поиск прошёл и подсветки нет — это знание о записи, его искать заново незачем."""
    store.put_intermediate("r1", dataclasses.replace(_intermediate(), tiles=[], labels=None, step=None), None)

    assert json.loads(storage.objects[store.intermediate_location("r1")])["tiles"] == []


@pytest.mark.parametrize("raw", [b"{not json", b"\xff\xfe\x00garbage"])
def test_corrupt_json_is_a_miss(tmp_path, storage, store, raw):
    storage.put_bytes(store.intermediate_location("r1"), raw)
    storage.put_bytes(store.blocks_location("k1"), raw)
    storage.put_bytes(store.text_location("k1"), b"\xff\xfe")

    assert store.get_intermediate("r1", tmp_path / "a.wav") is None
    assert store.get_result("k1") is None
    assert store.get_text("k1") is None


def test_result_meta_is_written_before_blocks(storage, store):
    doc = {"schema": blocks.SCHEMA, "doc_id": SHA, "source_uri": None, "speakers": {}, "blocks": []}
    meta = {"speaker_source": "голос", "tiles": 0, "speakers": 1}

    assert store.put_result("k1", doc, "", video_error=None, meta=meta)

    assert storage.writes == [store.text_location("k1"), store.meta_location("k1"), store.blocks_location("k1")]
    assert store.get_meta("k1") == meta
    assert store.get_meta("нет") is None
