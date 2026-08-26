"""Извлечение аудиодорожки из любого медиафайла в 16 кГц моно."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def find_ffmpeg() -> tuple[str, str]:
    """Возвращает пути к ffmpeg и ffprobe: системные либо из пакета static-ffmpeg."""
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe

    try:
        import static_ffmpeg  # noqa: PLC0415

        static_ffmpeg.add_paths()
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise RuntimeError("ffmpeg не найден. Установите его в систему (apt install ffmpeg) или выполните: pip install static-ffmpeg") from exc

    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not (ffmpeg and ffprobe):  # pragma: no cover
        raise RuntimeError("ffmpeg не найден даже после static_ffmpeg.add_paths()")
    return ffmpeg, ffprobe


def probe_duration(path: Path) -> float:
    _, ffprobe = find_ffmpeg()
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def extract_wav(src: Path, dst: Path, denoise: bool = False) -> Path:
    """Достаёт моно-дорожку 16 кГц. denoise включает мягкий шумодав + нормализацию."""
    ffmpeg, _ = find_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src), "-vn"]
    if denoise:
        cmd += ["-af", "highpass=f=80,afftdn=nf=-25,dynaudnorm=f=200:g=5"]
    cmd += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le", str(dst)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg не смог извлечь аудио из {src}:\n{proc.stderr.strip()}")
    return dst


def load_wav(path: Path) -> np.ndarray:
    """Читает 16-битный моно WAV в float32 [-1, 1]."""
    with wave.open(str(path)) as f:
        if f.getframerate() != SAMPLE_RATE or f.getnchannels() != 1:
            raise ValueError(f"Ожидался моно WAV {SAMPLE_RATE} Гц, получено {f.getnchannels()} канал(ов) @ {f.getframerate()} Гц")
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
