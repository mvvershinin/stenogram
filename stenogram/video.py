"""Детектор активного говорящего по видео.

Сервисы видеоконференций (Zoom, Teams, TrueConf, Telemost, ВКС судов) подсвечивают
рамкой плитку того, кто сейчас говорит. Это гораздо более надёжный признак, чем
голосовые отпечатки, поэтому используется как основной, а звук — как дополнение.

Плитки ищутся автоматически: пиксели рамки то цветные, то серые, поэтому у них
аномально высокая дисперсия насыщенности по времени.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import find_ffmpeg

PROBE_FPS = 5.0


@dataclass
class Tile:
    """Плитка участника, заданная полоской кадра, по которой видно подсветку."""
    x: int
    y: int
    width: int
    height: int

    @property
    def probe(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


def _frames(path: Path, fps: float, width: int, height: int, count: int | None = None) -> np.ndarray:
    ffmpeg, _ = find_ffmpeg()
    cmd = [ffmpeg, "-v", "error", "-i", str(path), "-vf", f"fps={fps}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    if count:
        cmd[-1:-1] = ["-frames:v", str(count)]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def video_size(path: Path) -> tuple[int, int]:
    _, ffprobe = find_ffmpeg()
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


def _saturation(frames: np.ndarray) -> np.ndarray:
    f = frames.astype(np.float32)
    return f.max(axis=-1) - f.min(axis=-1)


def find_tiles(path: Path, samples: int = 120, progress: bool = True) -> list[Tile]:
    """Ищет плитки участников по мерцающей рамке подсветки.

    Рамки одной плитки (левая и правая) загораются одновременно, рамки разных плиток —
    в разное время. Поэтому границы группируются по совпадению их «мигания» во времени:
    это работает при любой раскладке и не зависит от сервиса.
    """
    width, height = video_size(path)
    duration = _duration(path)
    fps = max(samples / max(duration, 1.0), 0.02)
    if progress:
        print(f"      ищу плитки участников ({width}x{height}) ...", flush=True)

    frames = _frames(path, fps, width, height)
    if len(frames) < 8:
        return []

    sat = _saturation(frames)                       # насыщенность: рамка цветная, фон серый
    swing = np.percentile(sat, 90, axis=0) - np.percentile(sat, 10, axis=0)
    mask = swing > max(40.0, np.percentile(swing, 99.0))
    counts = mask.sum(axis=0)
    if counts.max() < 20:
        return []

    columns = np.where(counts >= max(10, 0.25 * counts.max()))[0]
    if len(columns) < 2:
        return []

    # Соседние столбцы — одна и та же рамка толщиной в несколько пикселей.
    groups: list[list[int]] = [[int(columns[0])]]
    for c in columns[1:]:
        if c - groups[-1][-1] > 3:
            groups.append([])
        groups[-1].append(int(c))

    edges, signals = [], []
    for group in groups:
        x = int(max(group, key=lambda c: counts[c]))
        rows = np.where(mask[:, x])[0]
        if len(rows) < 8:
            continue
        edges.append((x, int(rows.min()), int(rows.max())))
        signals.append(sat[:, rows.min():rows.max() + 1, x].mean(axis=1))

    if not edges:
        return []

    # Границы с одинаковым рисунком мигания принадлежат одной плитке.
    S = np.stack(signals)
    S = (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-9)
    corr = S @ S.T / S.shape[1]

    assigned: list[int] = [-1] * len(edges)
    tiles: list[Tile] = []
    for i in range(len(edges)):
        if assigned[i] >= 0:
            continue
        members = [j for j in range(len(edges)) if assigned[j] < 0 and corr[i, j] > 0.8]
        for j in members:
            assigned[j] = len(tiles)
        x, top, bottom = edges[members[0]]
        margin = (bottom - top) // 6
        tiles.append(Tile(x, top + margin, 3, max(8, bottom - top - 2 * margin)))

    tiles.sort(key=lambda t: t.x)
    if progress:
        print(f"      плиток найдено: {len(tiles)}", flush=True)
    return tiles


def _duration(path: Path) -> float:
    _, ffprobe = find_ffmpeg()
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def active_speaker(path: Path, tiles: list[Tile], fps: float = PROBE_FPS,
                   progress: bool = True) -> tuple[np.ndarray, float]:
    """Для каждого момента времени — индекс подсвеченной плитки (или -1).

    Возвращает (метки по кадрам, шаг по времени в секундах).
    """
    if not tiles:
        return np.empty(0, dtype=int), 1.0 / fps

    ffmpeg, _ = find_ffmpeg()
    # Полоску каждой плитки усредняем в один пиксель на кадр — данных выходит мало,
    # поэтому час видео обрабатывается секунды.
    signals = []
    for i, tile in enumerate(tiles):
        x, y, w, h = tile.probe
        cmd = [ffmpeg, "-v", "error", "-i", str(path), "-vf",
               f"fps={fps},crop={w}:{h}:{x}:{y},scale=1:1:flags=area",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        px = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.float32)
        # «Цветность» рамки: насколько пиксель далёк от серого.
        signals.append(px.max(axis=1) - px.min(axis=1))
        if progress:
            print(f"      подсветка плитки {i+1}/{len(tiles)}", flush=True)

    n = min(len(s) for s in signals)
    matrix = np.stack([s[:n] for s in signals])
    best = matrix.argmax(axis=0)
    strength = matrix.max(axis=0) - np.median(matrix, axis=0)
    labels = np.where(strength > 12.0, best, -1)
    return labels, 1.0 / fps


def to_turns(labels: np.ndarray, step: float, word_gaps: list[tuple[float, float]] | None = None,
             lag: float = 1.2, snap_window: float = 2.0):
    """Превращает покадровые метки в интервалы речи.

    Подсветка включается позже начала реплики, поэтому границы сдвигаются на lag назад
    и притягиваются к ближайшей паузе в речи — иначе фразы режутся посередине.
    """
    from .diarization import Turn

    if len(labels) == 0:
        return []

    changes = [0] + [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]] + [len(labels)]
    bounds = [c * step for c in changes]
    bounds[1:-1] = [max(0.0, t - lag) for t in bounds[1:-1]]

    if word_gaps:
        centres = np.array([(a + b) / 2 for a, b in word_gaps])
        lengths = np.array([b - a for a, b in word_gaps])
        for i in range(1, len(bounds) - 1):
            near = np.where(np.abs(centres - bounds[i]) <= snap_window)[0]
            if len(near):
                j = near[np.argmax(lengths[near] - 0.15 * np.abs(centres[near] - bounds[i]))]
                bounds[i] = float(centres[j])
        bounds = list(np.maximum.accumulate(bounds))

    turns = []
    for i in range(len(changes) - 1):
        speaker = int(labels[changes[i]])
        if speaker >= 0 and bounds[i + 1] - bounds[i] > 0.05:
            turns.append(Turn(bounds[i], bounds[i + 1], speaker))
    return turns


def word_gaps(segments, min_gap: float = 0.15) -> list[tuple[float, float]]:
    """Паузы между словами — кандидаты на настоящую границу реплики."""
    words = [(w.start, w.end) for s in segments for w in s.words]
    words.sort()
    return [(words[i][1], words[i + 1][0]) for i in range(len(words) - 1)
            if words[i + 1][0] - words[i][1] >= min_gap]


def has_video(path: Path) -> bool:
    """Есть ли в файле видеодорожка (у аудиофайлов её нет)."""
    _, ffprobe = find_ffmpeg()
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    return "video" in out.stdout
