"""Интеграционные тесты: кэш звука на настоящем ffmpeg.

В обычный запуск и в CI не входят, там ffmpeg подменён. Запуск на машине с ffmpeg:

    pytest tests/integration    # из каталога service/
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
from fakes import write_tone

from stenogram import audio, pipeline
from stenogram.asr import Segment, Word
from stenogram.video import Tile
from stenogram_service import cache
from stenogram_service.storage import Location


def _intermediate() -> pipeline.Intermediate:
    words = [Word(0.5, 0.9, " Добрый"), Word(1.0, 1.4, " день")]
    return pipeline.Intermediate([Segment(0.5, 1.4, "Добрый день", words)], "ru", [Tile(10, 0, 3, 20)], np.array([0, 0, -1]), 0.2)


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


@pytest.mark.ffmpeg
def test_corrupt_cached_audio_is_a_miss(tmp_path, storage, store):
    storage.put_json(store.intermediate_location("r1"), _intermediate().to_dict())
    storage.put_bytes(store.audio_location("r1"), b"not a flac at all")

    assert store.get_intermediate("r1", tmp_path / "a.wav") is None
    assert not (tmp_path / "a.wav").exists()
