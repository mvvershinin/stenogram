"""Общие фикстуры для тестов stenogram.

Ничего из настоящих записей: WAV-фикстуры генерируются на лету средствами
stdlib `wave` (без ffmpeg вообще), видео — через ffmpeg lavfi (`testsrc`,
цветные прямоугольники), если бинарник найден. Правило зафиксировано
в `.docs/privacy.md`: тестовые фикстуры — синтетические.

Тесты, помеченные `@pytest.mark.ffmpeg`, автоматически пропускаются, если
`stenogram.audio.find_ffmpeg()` не может найти ни системный ffmpeg, ни
пакет `static-ffmpeg`. Такие тесты лежат в tests/integration и в CI не входят.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from stenogram.align import Utterance
from stenogram.asr import Segment, Word
from stenogram.audio import find_ffmpeg
from stenogram.diarization import Turn

# ---------------------------------------------------------------------------
# Фабрики dataclass'ов — короткие обёртки с удобными умолчаниями, чтобы в
# тестах не переписывать позиционные аргументы каждый раз.
# ---------------------------------------------------------------------------


def make_word(start: float, end: float, text: str = " слово") -> Word:
    """Slovo с потайминговкой. text — с ведущим пробелом по умолчанию,
    как отдаёт whisper (см. .docs/domain.md)."""
    return Word(start=start, end=end, text=text)


def make_segment(start: float, end: float, text: str = "", words: list[Word] | None = None) -> Segment:
    words = list(words) if words is not None else []
    if not text and words:
        text = "".join(w.text for w in words).strip()
    return Segment(start=start, end=end, text=text, words=words)


def make_turn(start: float, end: float, speaker: int = 0) -> Turn:
    return Turn(start=start, end=end, speaker=speaker)


def make_utterance(start: float, end: float, speaker: int = 0, text: str = "", words: list[Word] | None = None) -> Utterance:
    return Utterance(start=start, end=end, speaker=speaker, text=text, words=list(words) if words is not None else [])


@pytest.fixture
def word_factory():
    return make_word


@pytest.fixture
def segment_factory():
    return make_segment


@pytest.fixture
def turn_factory():
    return make_turn


@pytest.fixture
def utterance_factory():
    return make_utterance


# ---------------------------------------------------------------------------
# ffmpeg: автоскип маркера `ffmpeg`, если бинарник недоступен ни в системе,
# ни через static-ffmpeg. Проверяется один раз за сессию.
# ---------------------------------------------------------------------------


def _ffmpeg_available() -> bool:
    try:
        find_ffmpeg()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _ffmpeg_available():
        return
    skip_no_ffmpeg = pytest.mark.skip(reason="ffmpeg не найден (ни системный, ни static-ffmpeg)")
    for item in items:
        if "ffmpeg" in item.keywords:
            item.add_marker(skip_no_ffmpeg)


@pytest.fixture(scope="session")
def ffmpeg_bin() -> tuple[str, str]:
    """Пути к ffmpeg/ffprobe. Тест, использующий эту фикстуру, обязан быть
    помечен `@pytest.mark.ffmpeg` — иначе он упадёт вместо того, чтобы
    аккуратно пропуститься на машине без ffmpeg."""
    return find_ffmpeg()


# ---------------------------------------------------------------------------
# Синтетический WAV средствами stdlib `wave` — без ffmpeg и без сети.
# ---------------------------------------------------------------------------


def write_sine_wav(path: Path, duration: float = 1.0, freq: float = 440.0, sample_rate: int = 16000, amplitude: float = 0.5) -> Path:
    """16-битный моно WAV с синусоидой заданной частоты — синтетика по
    рецепту .docs/privacy.md, никаких настоящих записей."""
    n_samples = int(duration * sample_rate)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            value = int(32767 * amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
            frames += struct.pack("<h", value)
        f.writeframes(bytes(frames))
    return path


@pytest.fixture
def sine_wav_factory(tmp_path):
    """Фабрика: sine_wav_factory(name, duration=1.0, freq=440.0) -> Path."""

    def factory(name: str = "sine.wav", duration: float = 1.0, freq: float = 440.0) -> Path:
        return write_sine_wav(tmp_path / name, duration=duration, freq=freq)

    return factory


@pytest.fixture
def synthetic_wav(sine_wav_factory) -> Path:
    """Один готовый синтетический WAV на 1 секунду, 16 кГц моно — для тестов,
    которым нужен просто «какой-нибудь настоящий файл на диске»."""
    return sine_wav_factory()
