"""Тесты video.py вокруг падения/риска по памяти, исправленного перед
публикацией (см. .docs/plans/2026-08-25_публикация.md, раздел 2, пункт
«Понизить пиковую память video.find_tiles»), плюс _probe_frame_limit —
чистая арифметика, появившаяся уже после первого захода на тесты.

Не полное покрытие модуля (to_turns, word_gaps и find_tiles на мигающих
плитках — в разделе «хорошо бы» плана).

Честно о памяти: раньше здесь были только два теста, считающие КОЛИЧЕСТВО
кадров (то, что ограничитель count не мёртв), а не память — то есть
заявленный в шапке файла регресс по памяти этими двумя тестами вообще не
охранялся, ревью это верно нашло. Ниже это исправлено тестом
test_probe_saturation_peak_memory_scales_with_output_not_with_full_materialization,
который измеряет пиковую трассируемую tracemalloc память по-настоящему —
подробности и границы честности метода в его собственном docstring.

_probe_frame_limit — чистая арифметика (max/int/сравнение), без ffmpeg и без
видео: тесты на неё не помечены `ffmpeg` и идут в быстрый прогон.
"""

from __future__ import annotations

import io
import subprocess

import numpy as np
import pytest

from stenogram import video
from stenogram.video import _frame_chunks, _probe_frame_limit, _probe_saturation

# ---------------------------------------------------------------------------
# _probe_frame_limit(): пол 0.02 на длинных записях — чистая арифметика,
# без ffmpeg
# ---------------------------------------------------------------------------


def test_probe_frame_limit_zero_duration_falls_back_to_samples():
    """max(samples, ...) — защита от файла с нулевой/отсутствующей
    длительностью в заголовке: не должно давать 2 (int(0*fps)+2) вместо
    разумных samples."""
    assert _probe_frame_limit(0.0, 0.02, samples=120) == 120


def test_probe_frame_limit_below_floor_uses_duration_times_fps_plus_margin():
    """Обычная запись (fps ещё не упёрся в пол 0.02): duration*fps == samples
    с точностью до плавающей точки, поэтому предел — samples + запас в 2
    кадра, а не голый samples."""
    # duration=60, fps=2.0 -> samples/duration=120/60=2.0 >= 0.02, пол не сработал
    assert _probe_frame_limit(60.0, 2.0, samples=120) == 122


def test_probe_frame_limit_floor_triggered_on_long_recording_exceeds_samples():
    """Регресс: запись длиннее samples/0.02 (при samples=120 — 100 минут)
    упирает fps в пол 0.02, и реальных кадров у ffmpeg получается заметно
    больше, чем samples, — раньше (`-frames:v samples`) выборка от этого
    редела на четверть, а не просто теряла пару кадров на округлении.
    Двухчасовая запись — ровно пример из docstring _probe_frame_limit."""
    duration = 7200.0  # 2 часа
    fps = max(120 / duration, 0.02)
    assert fps == 0.02  # пол сработал

    limit = _probe_frame_limit(duration, fps, samples=120)

    assert limit == 146
    assert limit > 120 * 1.2  # выборка не должна была бы поредеть на четверть


def test_probe_frame_limit_exactly_at_floor_boundary_still_covers_samples():
    """Ровно на границе (samples/0.02 = 6000 с = 100 минут) — устойчивость
    возле границы, а не конкретное число по одну её сторону.

    Регресс, найденный ревью: `>= 120` здесь — тавтология max(samples, x)
    >= samples, верная при ЛЮБОМ x, поэтому даже сломанная арифметика
    второй ветки формулы (int(duration*fps)+2) осталась бы незамеченной,
    пока не проваливалась совсем в отрицательные числа. На границе
    duration*fps == samples == 120 ровно, поэтому точное ожидаемое
    значение — 120 + 2 (запас) = 122; только равенство ловит поломку самой
    формулы, а не только «не провалился ниже samples»."""
    duration = 100 * 60.0  # 6000 c
    fps = max(120 / duration, 0.02)

    assert _probe_frame_limit(duration, fps, samples=120) == 122


def test_probe_frame_limit_never_below_samples():
    """Свойство по построению (max(samples, ...)): предел никогда не бывает
    меньше запрошенного числа сэмплов — иначе _probe_saturation получила бы
    буфер меньше, чем сама просит заполнить.

    Регресс, найденный ревью: прежняя проверка `>= 120` по каждому случаю —
    та же тавтология max(samples, x) >= samples, верная независимо от
    того, права ли арифметика x = int(duration*fps)+2. Теперь для каждой
    пары (duration, fps) сверяется КОНКРЕТНОЕ ожидаемое значение: свойство
    «не меньше samples» по-прежнему выполняется (все числа справа — 120 или
    больше), но заодно зафиксирована и сама арифметика второй ветки."""
    cases = [
        (0.0, 0.02, 120),  # int(0*0.02)+2=2 < samples -> сработал max(), предел 120
        (1.0, 120.0, 122),  # int(1*120)+2=122 -> уже выше samples сам по себе
        (3600.0, 0.02, 120),  # int(3600*0.02)+2=74 < samples -> сработал max(), предел 120
        (5999.0, 0.02, 121),  # int(5999*0.02)+2=121 -> чуть выше samples без всякого max()
    ]
    for duration, fps, expected in cases:
        assert _probe_frame_limit(duration, fps, samples=120) == expected


# ---------------------------------------------------------------------------
# _frame_chunks / _probe_saturation: ffmpeg подменён, поток кадров задан в тесте.
# То же на настоящем ffmpeg, включая пик памяти, — tests/integration/test_ffmpeg.py.
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 4, 2
FRAME_SIZE = WIDTH * HEIGHT * 3


class FakeFfmpeg:
    """Подмена subprocess.Popen: отдаёт заданный поток байт и код возврата."""

    def __init__(self, stream: bytes, returncode: int = 0, stderr: bytes = b""):
        self.stream, self.returncode, self.errors = stream, returncode, stderr
        self.cmd: list[str] = []

    def __call__(self, cmd, **kwargs):
        self.cmd = cmd
        self.stdout, self.stderr = io.BytesIO(self.stream), io.BytesIO(self.errors)
        return self

    def wait(self) -> int:
        return self.returncode


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    def install(stream: bytes, returncode: int = 0, stderr: bytes = b"") -> FakeFfmpeg:
        fake = FakeFfmpeg(stream, returncode, stderr)
        monkeypatch.setattr(video, "find_ffmpeg", lambda: ("ffmpeg", "ffprobe"))
        monkeypatch.setattr(video.subprocess, "Popen", fake)
        return fake

    return install


def test_frame_chunks_limits_ffmpeg_by_count(fake_ffmpeg):
    """count обязан дойти до ffmpeg: иначе он декодирует запись целиком."""
    fake = fake_ffmpeg(bytes(10 * FRAME_SIZE))

    total = sum(len(chunk) for chunk in _frame_chunks("запись.mp4", 30.0, WIDTH, HEIGHT, 10))

    assert fake.cmd[fake.cmd.index("-frames:v") + 1] == "10"
    assert total == 10


def test_frame_chunks_splits_stream_into_chunks(fake_ffmpeg, monkeypatch):
    """Чтение кусками держит пик памяти: весь поток одним массивом не материализуется."""
    monkeypatch.setattr(video, "_CHUNK_BYTES", 3 * FRAME_SIZE)
    fake_ffmpeg(bytes(7 * FRAME_SIZE))

    sizes = [len(chunk) for chunk in _frame_chunks("запись.mp4", 30.0, WIDTH, HEIGHT, 7)]

    assert sizes == [3, 3, 1]


def test_frame_chunks_rejects_partial_frame(fake_ffmpeg):
    fake_ffmpeg(bytes(FRAME_SIZE + 1))

    with pytest.raises(ValueError, match="не кратно"):
        list(_frame_chunks("запись.mp4", 30.0, WIDTH, HEIGHT, 2))


def test_frame_chunks_reports_ffmpeg_failure(fake_ffmpeg):
    fake_ffmpeg(b"", returncode=1, stderr=b"moov atom not found")

    with pytest.raises(subprocess.CalledProcessError) as caught:
        list(_frame_chunks("запись.mp4", 30.0, WIDTH, HEIGHT, 2))

    assert caught.value.stderr == b"moov atom not found"


def test_probe_saturation_returns_frames_actually_received(fake_ffmpeg):
    """Буфер рассчитан на count кадров, лишние строки отрезаются. Насыщенность — max − min по каналам."""
    fake_ffmpeg(bytes([10, 20, 30]) * (WIDTH * HEIGHT) * 3)

    sat = _probe_saturation("запись.mp4", 30.0, WIDTH, HEIGHT, 10)

    assert sat.shape == (3, HEIGHT, WIDTH)
    assert np.all(sat == 20)
