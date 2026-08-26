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

import subprocess
import tracemalloc

import pytest

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
# _frame_chunks / _probe_saturation: ограничитель count не мёртв (регресс)
# ---------------------------------------------------------------------------


@pytest.fixture
def longer_than_count_video(tmp_path, ffmpeg_bin):
    """Синтетическое видео заведомо длиннее count кадров на запрошенном fps:
    3 с при 30 кадрах/с = 90 кадров источника, что намного больше count=10
    в тестах ниже. Ничего похожего на настоящую запись — просто тестовый
    паттерн ffmpeg (см. .docs/privacy.md, раздел о фикстурах)."""
    ffmpeg, _ = ffmpeg_bin
    path = tmp_path / "synthetic.mp4"

    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30:duration=3", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@pytest.mark.ffmpeg
def test_probe_saturation_respects_count_limit_on_longer_video(longer_than_count_video):
    """Регресс: раньше ограничитель count у чтения кадров не использовался
    («мёртвый» параметр) — ffmpeg декодировал видео целиком в полном
    разрешении, что для 1080p давало по расчётам единицы гигабайт пиковой
    памяти ещё до загрузки моделей. Теперь count передаётся в ffmpeg как
    `-frames:v`, и _probe_saturation обязана вернуть ровно count кадров,
    а не все кадры источника (в фикстуре их 90)."""
    sat = _probe_saturation(longer_than_count_video, fps=30.0, width=64, height=64, count=10)

    assert sat.shape == (10, 64, 64)


@pytest.mark.ffmpeg
def test_frame_chunks_total_frames_equals_count_not_source_length(longer_than_count_video):
    """То же самое на уровне генератора кусков: сумма кадров по всем кускам
    обязана равняться count (10), а не количеству кадров источника (90)."""
    total = sum(len(chunk) for chunk in _frame_chunks(longer_than_count_video, 30.0, 64, 64, 10))

    assert total == 10


# ---------------------------------------------------------------------------
# _probe_saturation: честный тест на пик памяти
# ---------------------------------------------------------------------------


@pytest.fixture
def frame_count_video(tmp_path, ffmpeg_bin):
    """7 с синтетического видео на 30 к/с = 210 кадров источника — заведомо
    больше обоих count из теста ниже (40 и 160). Разрешение 640x360 выбрано
    так, чтобы разница между «читать кусками» и «материализовать всё разом»
    была на порядок больше шума измерения (см. расчёт в docstring теста
    ниже), но видео при этом кодировалось и обрабатывалось за секунды, а не
    минуты. Ничего похожего на настоящую запись — тестовый паттерн ffmpeg."""
    ffmpeg, _ = ffmpeg_bin
    path = tmp_path / "synthetic_long.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=7", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def _peak_traced_bytes(fn, *args, **kwargs) -> int:
    """Пиковая прослеженная tracemalloc память за время вызова fn.

    numpy регистрирует свои аллокации в домене tracemalloc начиная с 1.15,
    поэтому массивы numpy (а не только «чистые» Python-объекты) учитываются
    в get_traced_memory() — проверено эмпирически в этом окружении перед
    тем, как полагаться на это в тесте: выделение массива (1000,1000,3)
    uint8 и его .astype(float32) корректно отражаются в traced-памяти как
    ~3 МБ и ~12 МБ соответственно.

    reset_peak() (доступен с Python 3.9) обнуляет только счётчик пика, не
    останавливая трассировку, — так вызов измеряется изолированно, даже
    если tracemalloc уже был включён снаружи (например, другим инструментом
    в CI)."""
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        fn(*args, **kwargs)
        return tracemalloc.get_traced_memory()[1]
    finally:
        if not was_tracing:
            tracemalloc.stop()


@pytest.mark.ffmpeg
def test_probe_saturation_peak_memory_scales_with_output_not_with_full_materialization(frame_count_video):
    """Честный тест на регресс по памяти (раздел 2 плана «Понизить пиковую
    память video.find_tiles»): раньше _saturation получала на вход ВЕСЬ
    декодированный клип разом (`frames.astype(np.float32)` на полном
    массиве) — то есть держала одновременно исходный uint8-массив и его
    четырёхкратно более тяжёлую float32-копию, каждая размером с ЧИСЛО
    ЗАПРОШЕННЫХ КАДРОВ. После правки _probe_saturation читает и обрабатывает
    поток кусками (_CHUNK_BYTES), поэтому пиковая память растёт с числом
    кадров ТОЛЬКО за счёт неизбежного выходного массива (height·width·4
    байт/кадр), а не за счёт временной полноразмерной float32-копии всего
    декодированного клипа.

    Тест не сравнивает абсолютную память с фиксированным порогом (она
    зависит от окружения и версии numpy) — он измеряет НАКЛОН
    (Δpeak / Δcount) на двух разных count одного и того же видео и
    сравнивает его с неизбежной ценой ОДНОГО кадра выходного массива
    (height*width*4 байт). Проверено эмпирически перед записью теста: на
    текущем (чункованном) коде наклон стабильно ~1.1–1.3×
    ожидаемого на нескольких повторах и парах (N1, N2), включая пару ниже;
    на копии кода со старым поведением (весь клип одним массивом, без
    чтения кусками) — стабильно ~5.7× ожидаемого на той же паре. Порог
    ниже (3×) отделяет одно от другого с запасом в обе стороны, не будучи
    завязан на абсолютные мегабайты.

    N1/N2 подобраны как компромисс между надёжностью сигнала (разница между
    чанкованным и материализующим всё разом поведением должна быть на
    порядок больше шума измерения) и временем прогона: большие N1/N2 дают
    чуть более чистый сигнал, но на этой машине счёт идёт на секунды за
    каждый лишний десяток кадров — 40/160 достаточно для уверенного
    разделения (~1.2× против ~5.7×) и укладывается в единицы секунд.

    Требует настоящего ffmpeg и настоящего числа кадров в источнике (иначе
    сравнение двух count выродилось бы в сравнение двух одинаково урезанных
    до конца файла выборок) — поэтому помечен @pytest.mark.ffmpeg и занимает
    несколько секунд, а не миллисекунды."""
    width, height, fps = 640, 360, 30.0
    small, large = 40, 160

    peak_small = _peak_traced_bytes(_probe_saturation, frame_count_video, fps, width, height, small)
    peak_large = _peak_traced_bytes(_probe_saturation, frame_count_video, fps, width, height, large)

    slope = (peak_large - peak_small) / (large - small)
    per_frame_output = height * width * 4  # неизбежный выходной float32-массив (n, h, w)

    assert slope < 3 * per_frame_output, (
        f"{slope:.0f} Б/кадр вместо ожидаемых ~{per_frame_output} Б/кадр — похоже, что весь клип снова материализуется в один массив, а не читается кусками"
    )
