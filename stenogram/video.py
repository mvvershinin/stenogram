"""Детектор активного говорящего по видео.

Сервисы видеоконференций (Zoom, Teams, TrueConf, Telemost, ВКС судов) подсвечивают
рамкой плитку того, кто сейчас говорит. Это гораздо более надёжный признак, чем
голосовые отпечатки, поэтому используется как основной, а звук — как дополнение.

Плитки ищутся автоматически: пиксели рамки то цветные, то серые, поэтому у них
аномально высокая дисперсия насыщенности по времени.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import find_ffmpeg

PROBE_FPS = 5.0

# Кадры поиска плиток читаются из ffmpeg кусками такого размера (в байтах «сырых»
# кадров), а не одним массивом: перевод всех кадров в float32 разом даёт ×4
# к размеру уже декодированного видео — на 1080p это единицы гигабайт ещё до
# загрузки моделей. Кусок обрабатывается независимо от соседних (поэлементно),
# поэтому результат совпадает с вычислением по цельному массиву вплоть до бита.
_CHUNK_BYTES = 32 * 1024 * 1024

# Перцентили размаха считаются полосами кадра такого размера (в байтах): np.percentile
# переставляет элементы, поэтому копирует вход целиком, а на 1080p это лишний гигабайт
# рядом с самим массивом насыщенности. Перцентиль по оси времени считается для каждого
# пикселя независимо от соседей, поэтому счёт полосами даёт тот же результат вплоть
# до бита — проверено поэлементно на нескольких формах массива.
_BAND_BYTES = 32 * 1024 * 1024


def _report(progress: bool | Callable[[str, float | None], None], message: str, fraction: float | None = None) -> None:
    """Сообщение о ходе работы: в stdout, в функцию вызывающего или никуда."""
    if callable(progress):
        progress(message, fraction)
    elif progress:
        print(message, flush=True)


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


def _frame_chunks(path: Path, fps: float, width: int, height: int, count: int):
    """Читает кадры поиска плиток из ffmpeg кусками — не больше count кадров всего.

    Каждый кусок — маленький np.uint8 массив формы (n, height, width, 3). Это тот
    же самый поток байт, который раньше читался одним `subprocess.run(capture_output=True)`
    и материализовался в один большой массив; здесь тот же вывод ffmpeg просто
    нарезается на части по мере чтения, побайтно без изменений.

    count — жёсткий предел (`-frames:v`), а не пожелание: всё, что сверх него, ffmpeg
    не отдаст. Значит вызывающий обязан передавать верхнюю оценку числа кадров
    (`_probe_frame_limit`), иначе выборка кадров молча поредеет.
    """
    ffmpeg, _ = find_ffmpeg()
    cmd = [ffmpeg, "-v", "error", "-i", str(path), "-vf", f"fps={fps}", "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    frame_size = height * width * 3
    chunk_frames = max(1, _CHUNK_BYTES // frame_size)
    chunk_bytes = chunk_frames * frame_size

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr: list[bytes] = []
    # stderr дренится отдельным потоком, иначе при достаточно длинном сообщении
    # об ошибке ffmpeg и наш процесс могут заблокировать друг друга: он — на
    # записи в переполненный пайп stderr, мы — на чтении stdout.
    reader = threading.Thread(target=lambda: stderr.append(proc.stderr.read()), daemon=True)
    reader.start()
    try:
        while True:
            raw = proc.stdout.read(chunk_bytes)
            if not raw:
                break
            if len(raw) % frame_size:
                raise ValueError(f"ffmpeg вернул {len(raw)} байт — не кратно размеру кадра {frame_size}")
            yield np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    finally:
        proc.stdout.close()
        reader.join()
        returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd, output=b"", stderr=stderr[0] if stderr else b"")


def video_size(path: Path) -> tuple[int, int]:
    _, ffprobe = find_ffmpeg()
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)], capture_output=True, text=True, check=True
    ).stdout.strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


def _saturation(frames: np.ndarray) -> np.ndarray:
    f = frames.astype(np.float32)
    return f.max(axis=-1) - f.min(axis=-1)


def _probe_frame_limit(duration: float, fps: float, samples: int) -> int:
    """Верхняя оценка числа кадров, которые ffmpeg отдаст при `fps=` на этой записи.

    Нужна ровно для того, чтобы заранее выделить буфер и ограничить чтение, не тронув
    саму выборку. Поэтому оценка обязана быть НЕ МЕНЬШЕ фактического числа кадров.

    samples на эту роль не годится. Фильтр `fps` выдаёт floor(t_посл · fps) + 1 кадр,
    где t_посл ≤ duration, то есть не больше floor(duration · fps) + 1. Пока fps не
    упёрся в нижний пол, duration · fps = samples и кадров выходит ровно samples. Но
    на записи длиннее samples / 0.02 (при samples = 120 — 100 минут) пол срабатывает,
    fps = 0.02, и кадров становится больше: у двухчасовой записи — 144, а не 120.
    Ограничить такую выборку числом samples значит проредить её на четверть.

    Запас в два кадра покрывает и погрешность произведения duration · fps в плавающей
    точке, и единицу модели; стоит он два кадра буфера. max(samples, ...) защищает от
    файла с нулевой или отсутствующей длительностью в заголовке.
    """
    return max(samples, int(duration * fps) + 2)


def _probe_saturation(path: Path, fps: float, width: int, height: int, count: int) -> np.ndarray:
    """Насыщенность по кадрам поиска плиток, посчитанная кусками (см. _frame_chunks).

    Результат — тот же массив (n, height, width) float32, что раньше собирался
    через _saturation(frames) на всём массиве разом; здесь он заполняется по мере
    чтения кусков в заранее выделенный буфер, поэтому одновременно в памяти нет ни
    полного набора кадров в uint8, ни тем более его копии в float32.

    count — верхняя оценка числа кадров (см. _probe_frame_limit), а не желаемая длина
    выборки: лишние строки буфера просто отрезаются в конце.
    """
    sat = np.empty((count, height, width), dtype=np.float32)
    n = 0
    for block in _frame_chunks(path, fps, width, height, count):
        m = len(block)
        sat[n : n + m] = _saturation(block)
        n += m
    return sat[:n]


def _swing(sat: np.ndarray) -> np.ndarray:
    """Размах насыщенности во времени по каждому пикселю: перцентиль_90 − перцентиль_10.

    Считается горизонтальными полосами кадра (см. _BAND_BYTES). Значение перцентиля по
    оси времени зависит только от «своего» пикселя, поэтому полосы дают тот же массив,
    что и один вызов на всём sat, вплоть до бита.

    Два отдельных вызова, а не один `np.percentile(band, [90, 10], axis=0)`: список
    квантилей numpy считает другим путём и даёт другие последние биты — проверено.
    """
    n, height, width = sat.shape
    rows = max(1, _BAND_BYTES // (n * width * sat.itemsize))
    swing = np.empty((height, width), dtype=sat.dtype)
    for y in range(0, height, rows):
        band = sat[:, y : y + rows, :]
        swing[y : y + rows] = np.percentile(band, 90, axis=0) - np.percentile(band, 10, axis=0)
    return swing


def find_tiles(path: Path, samples: int = 120, progress: bool | Callable[[str, float | None], None] = True) -> list[Tile]:
    """Ищет плитки участников по мерцающей рамке подсветки.

    Рамки одной плитки (левая и правая) загораются одновременно, рамки разных плиток —
    в разное время. Поэтому границы группируются по совпадению их «мигания» во времени:
    это работает при любой раскладке и не зависит от сервиса.
    """
    width, height = video_size(path)
    duration = _duration(path)
    fps = max(samples / max(duration, 1.0), 0.02)
    if progress:
        _report(progress, f"      ищу плитки участников ({width}x{height}) ...")

    # рамка цветная, фон серый; предел чтения — из фактической формулы, а не из samples
    sat = _probe_saturation(path, fps, width, height, _probe_frame_limit(duration, fps, samples))
    if len(sat) < 8:
        return []

    swing = _swing(sat)
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
        signals.append(sat[:, rows.min() : rows.max() + 1, x].mean(axis=1))

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
        _report(progress, f"      плиток найдено: {len(tiles)}")
    return tiles


def _duration(path: Path) -> float:
    _, ffprobe = find_ffmpeg()
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], capture_output=True, text=True, check=True
    )
    return float(out.stdout.strip())


def active_speaker(path: Path, tiles: list[Tile], fps: float = PROBE_FPS, progress: bool | Callable[[str, float | None], None] = True) -> tuple[np.ndarray, float]:
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
        cmd = [ffmpeg, "-v", "error", "-i", str(path), "-vf", f"fps={fps},crop={w}:{h}:{x}:{y},scale=1:1:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        px = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.float32)
        # «Цветность» рамки: насколько пиксель далёк от серого.
        signals.append(px.max(axis=1) - px.min(axis=1))
        if progress:
            _report(progress, f"      подсветка плитки {i + 1}/{len(tiles)}", (i + 1) / len(tiles))

    n = min(len(s) for s in signals)
    matrix = np.stack([s[:n] for s in signals])
    best = matrix.argmax(axis=0)
    strength = matrix.max(axis=0) - np.median(matrix, axis=0)
    labels = np.where(strength > 12.0, best, -1)
    return labels, 1.0 / fps


def to_turns(labels: np.ndarray, step: float, word_gaps: list[tuple[float, float]] | None = None, lag: float = 1.2, snap_window: float = 2.0):
    """Превращает покадровые метки в интервалы речи.

    Подсветка включается позже начала реплики, поэтому границы сдвигаются на lag назад
    и притягиваются к ближайшей паузе в речи — иначе фразы режутся посередине.
    """
    from .diarization import Turn  # noqa: PLC0415 -- намеренно локальный импорт, см. docs/архитектура.md

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
    return [(words[i][1], words[i + 1][0]) for i in range(len(words) - 1) if words[i + 1][0] - words[i][1] >= min_gap]


def has_video(path: Path) -> bool:
    """Есть ли в файле видеодорожка (у аудиофайлов её нет)."""
    _, ffprobe = find_ffmpeg()
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return "video" in out.stdout
