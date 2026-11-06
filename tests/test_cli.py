"""Тесты командного интерфейса (stenogram.cli).

parse_names / parse_tile_speakers — разбор аргументов на границе CLI;
build_parser — дефолты; main() — ранние выходы и работа общего перехватчика
исключений.

Регрессы этого файла (см. .docs/plans/2026-08-25_публикация.md, раздел 2 и 5):
  - общий перехватчик исключений в main() переводит RuntimeError/ValueError/
    OSError/URLError/tarfile.TarError/subprocess.SubprocessError/wave.Error
    в сообщение в stderr + код 1 (сообщение НЕ обязано быть одной строкой —
    см. отдельный тест на настоящий четырёхстрочный триггер), а
    KeyboardInterrupt — в код 130 отдельной веткой, BrokenPipeError — в код
    0 без всякого сообщения отдельной веткой ВЫШЕ общего перехватчика;
  - STENOGRAM_TRACEBACK=1 возвращает полный трейсбек вместо одной строки:
    без него баг-репорт от постороннего состоит из одной непонятной фразы
    (RecursionError — подкласс RuntimeError, ошибки numpy — ValueError);
  - stdout сбрасывается ВНУТРИ try, до `return 0`: иначе остаток буфера
    уходит в закрытую трубу уже при завершении интерпретатора, вне всякого
    перехватчика, и `stenogram … | head` даёт «Exception ignored» и код 120;
  - каталог вывода создаётся при разборе аргументов, до первого «[1/4]»,
    а не как побочный эффект --keep-wav на последнем шаге;
  - --max-speakers < 1 и --speakers < 1 отвергаются на границе CLI: первый
    ронял diarization.estimate_speakers, второй молча давал стенограмму на
    одного участника после полного прохода;
  - parse_names требует N >= 1 в форме "N=имя";
  - parse_tile_speakers разбирает и валидирует --tile-speakers (числа >= 1)
    на границе CLI; длина сверяется отдельно, уже после того как найдены
    плитки (шаг [2/4]), — раньше просто нельзя, число плиток ещё не известно.
    Сверяется именно с НАЙДЕННЫМИ плитками: по ним числа и нумеруются.

main() тестируется без единого настоящего медиафайла и без сети: тяжёлые
шаги (audio.extract_wav и дальше) подменяются моками, потому что нас
интересует только код вокруг них — разбор аргументов и обработка ошибок.
Исключений два. Фикстура `fake_pipeline` прогоняет main() насквозь, подменяя
только то, что требует ffmpeg/ONNX/сети, — середина (video.to_turns,
diarization.diarize_with_video, align.assign, render.write) работает
по-настоящему, иначе тупик с --tile-speakers не воспроизвести. Настоящий audio.extract_wav на синтетическом
видео без звуковой дорожки прогоняет tests/integration/test_ffmpeg.py — реальный первый триггер общего перехватчика.
"""

from __future__ import annotations

import subprocess
import tarfile
import wave
from urllib.error import URLError

import numpy as np
import pytest

from stenogram import asr, audio, cli, diarization, video
from stenogram.asr import Segment, Word
from stenogram.render import WRITERS

# ---------------------------------------------------------------------------
# Фикстура полного прогона main() без ffmpeg, ONNX и сети
# ---------------------------------------------------------------------------
#
# Подменяется ровно то, что ходит наружу: извлечение звука, поиск плиток,
# покадровые метки подсветки, распознавание речи и сегментатор sherpa.
# Середина пайплайна работает по-настоящему — именно она и интересна:
# video.to_turns решает, чья подсветка доживёт до разметки участников.

_STEP = 0.2  # 5 к/с, как PROBE_FPS


def _labels_three_tiles_short_middle() -> np.ndarray:
    """0–10 с — плитка 0, 10–11.2 с — плитка 1, 11.2–20 с — плитка 2.

    Подсветка средней плитки длится 1.2 с — ровно столько, сколько составляет
    сдвиг --highlight-lag. После сдвига обе её границы (8.8 и 10.0) попадают
    в окно притяжения одной и той же паузы, схлопываются монотонизацией в
    video.to_turns и отсекаются порогом длительности."""
    labels = np.zeros(100, dtype=int)
    labels[50:56] = 1
    labels[56:] = 2
    return labels


def _dense_speech_with_one_pause() -> list[Segment]:
    """Слитная речь с единственной настоящей паузой 8.75–10.25 с (центр 9.5).

    Остальные промежутки между словами — 0.05 с, ниже порога min_gap=0.15,
    и в video.word_gaps не попадают: притягивать границы больше не к чему."""
    starts = [round(i * 0.4, 2) for i in range(22)] + [round(10.25 + i * 0.4, 2) for i in range(22)]
    words = [Word(start=t, end=round(t + 0.35, 2), text=" слово") for t in starts]
    return [Segment(start=words[0].start, end=words[-1].end, text="".join(w.text for w in words).strip(), words=words)]


@pytest.fixture
def fake_pipeline(tmp_path, monkeypatch):
    """Прогоняет cli.main() насквозь. Возвращает объект с .run(*argv) и .video_turns."""

    class Pipeline:
        def __init__(self):
            self.input = tmp_path / "запись.mp4"
            self.input.write_bytes(b"")
            self.tiles = [video.Tile(10, 0, 3, 20), video.Tile(50, 0, 3, 20), video.Tile(90, 0, 3, 20)]
            self.segments = _dense_speech_with_one_pause()
            self.video_turns: list = []

        def run(self, *extra: str) -> int:
            return cli.main([str(self.input), "-o", str(tmp_path / "вывод"), "-f", "txt", *extra])

    pipeline = Pipeline()
    real_to_turns = video.to_turns

    def recording_to_turns(*args, **kwargs):
        pipeline.video_turns = real_to_turns(*args, **kwargs)
        return pipeline.video_turns

    monkeypatch.setattr(audio, "extract_wav", lambda src, dst, denoise=False: dst)
    monkeypatch.setattr(audio, "probe_duration", lambda path: 20.0)
    monkeypatch.setattr(audio, "load_wav", lambda path: np.zeros(16000 * 20, dtype=np.float32))
    monkeypatch.setattr(video, "has_video", lambda path: True)
    monkeypatch.setattr(video, "find_tiles", lambda path, **kw: pipeline.tiles)
    monkeypatch.setattr(video, "active_speaker", lambda path, tiles, **kw: (_labels_three_tiles_short_middle(), _STEP))
    monkeypatch.setattr(video, "to_turns", recording_to_turns)
    monkeypatch.setattr(asr, "transcribe", lambda *a, **k: (pipeline.segments, "ru"))
    monkeypatch.setattr(diarization, "segment", lambda audio, num_threads=4, progress=True: [diarization.Turn(i + 0.05, i + 0.95, 0) for i in range(20)])
    return pipeline


# ---------------------------------------------------------------------------
# parse_names(): позиционный, через "=", смешанный, границы
# ---------------------------------------------------------------------------


def test_parse_names_none_returns_none():
    assert cli.parse_names(None) is None


def test_parse_names_empty_string_returns_none():
    assert cli.parse_names("") is None


def test_parse_names_only_commas_returns_empty_dict():
    """Раскладка вся из пустых элементов — не то же самое, что None:
    пользователь передал непустую строку, значит явно намеревался что-то
    задать, но там нечего разбирать; результат — пустой словарь, а не None."""
    assert cli.parse_names(",,,") == {}


def test_parse_names_positional_simple():
    assert cli.parse_names("Судья,Эксперт") == {0: "Судья", 1: "Эксперт"}


def test_parse_names_single_positional():
    assert cli.parse_names("Судья") == {0: "Судья"}


def test_parse_names_via_equals_one_indexed():
    assert cli.parse_names("3=Эксперт") == {2: "Эксперт"}


def test_parse_names_mixed_positional_and_equals():
    """Позиция считается по номеру элемента в исходной раскладке через
    запятую (enumerate по всему списку), а не отдельным счётчиком только
    позиционных элементов — поэтому "Истец" (третий по счёту элемент,
    позиция 2) получает ключ 2, а не 1."""
    assert cli.parse_names("Судья,4=Эксперт,Истец") == {0: "Судья", 3: "Эксперт", 2: "Истец"}


def test_parse_names_empty_element_between_commas_is_skipped_but_consumes_position():
    """Пустой элемент пропускается (continue), но позиция всё равно
    расходуется — за ним следующий позиционный элемент получает ключ 2,
    а не 1: индекс 1 просто отсутствует в результате."""
    assert cli.parse_names("Судья,,Эксперт") == {0: "Судья", 2: "Эксперт"}


def test_parse_names_spaces_around_equals_and_value_are_stripped():
    assert cli.parse_names(" 2 = Эксперт ") == {1: "Эксперт"}


def test_parse_names_spaces_around_positional_value_are_stripped():
    assert cli.parse_names(" Судья , Эксперт ") == {0: "Судья", 1: "Эксперт"}


def test_parse_names_zero_index_raises_value_error_with_hint():
    """Нумерация в --names с единицы: "0=..." обязано явно упасть, а не
    молча уйти в ключ -1 и потерять имя."""
    with pytest.raises(ValueError, match="нумерация с 1"):
        cli.parse_names("0=Эксперт")


def test_parse_names_negative_index_raises_value_error():
    with pytest.raises(ValueError, match="нумерация с 1"):
        cli.parse_names("-1=Эксперт")


def test_parse_names_non_numeric_key_raises_value_error():
    """int() на нечисловом ключе бросает свой ValueError без явной
    обработки — сообщение менее дружелюбное, чем у проверки "< 1", но это
    всё ещё ValueError, и main() поймает его тем же общим перехватчиком."""
    with pytest.raises(ValueError):
        cli.parse_names("abc=Эксперт")


# ---------------------------------------------------------------------------
# parse_tile_speakers(): список чисел, границы, значения < 1
# ---------------------------------------------------------------------------


def test_parse_tile_speakers_none_returns_none():
    assert cli.parse_tile_speakers(None) is None


def test_parse_tile_speakers_empty_string_returns_none():
    assert cli.parse_tile_speakers("") is None


def test_parse_tile_speakers_basic_list():
    assert cli.parse_tile_speakers("7,1") == [7, 1]


def test_parse_tile_speakers_single_value():
    assert cli.parse_tile_speakers("3") == [3]


def test_parse_tile_speakers_spaces_around_values_are_stripped():
    assert cli.parse_tile_speakers(" 7 , 1 ") == [7, 1]


def test_parse_tile_speakers_empty_item_between_commas_raises_value_error():
    with pytest.raises(ValueError, match="пустое значение"):
        cli.parse_tile_speakers("7,,1")


def test_parse_tile_speakers_non_integer_item_raises_value_error():
    with pytest.raises(ValueError, match="не целое число"):
        cli.parse_tile_speakers("abc")


def test_parse_tile_speakers_zero_raises_value_error():
    """Регресс: за плиткой не может быть меньше одного человека — 0 (как и
    отрицательные числа) отвергается на границе CLI, той же формулировкой,
    что и в diarization._tile_counts (см. test_diarization.py), а не только
    там."""
    with pytest.raises(ValueError, match="не может быть меньше одного человека"):
        cli.parse_tile_speakers("0")


def test_parse_tile_speakers_negative_value_raises_value_error():
    with pytest.raises(ValueError, match="не может быть меньше одного человека"):
        cli.parse_tile_speakers("7,-1")


# ---------------------------------------------------------------------------
# build_parser(): дефолты
# ---------------------------------------------------------------------------


def test_build_parser_prog_is_stenogram():
    assert cli.build_parser().prog == "stenogram"


def test_build_parser_defaults():
    args = cli.build_parser().parse_args(["запись.mp4"])

    assert str(args.input) == "запись.mp4"
    assert args.output is None
    assert args.formats == "txt,md,srt,json"
    assert args.speakers is None
    assert args.max_speakers == 12
    assert args.names is None
    assert args.no_diarization is False
    assert args.no_video is False
    assert args.tile_speakers is None
    assert args.highlight_lag == 1.2
    assert args.model == "large-v3-turbo"
    assert args.language is None
    assert args.prompt is None
    assert args.beam_size == 5
    assert args.device == "cpu"
    assert args.compute_type == "int8"
    assert args.threads == 0
    assert args.denoise is False
    assert args.keep_wav is False


def test_formats_module_constant_matches_render_writers():
    """FORMATS используется и в валидации --formats, и в --help — обязан
    совпадать с реальным набором WRITERS, а не быть отдельным списком,
    который можно забыть обновить."""
    assert set(cli.FORMATS) == set(WRITERS)


# ---------------------------------------------------------------------------
# main(): ранние выходы до тяжёлого пайплайна
# ---------------------------------------------------------------------------


def test_main_file_not_found_returns_1(tmp_path, capsys):
    missing = tmp_path / "нет-такого-файла.mp4"

    code = cli.main([str(missing)])

    assert code == 1
    assert "Файл не найден" in capsys.readouterr().err


def test_main_unknown_format_returns_1(tmp_path, capsys):
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")

    code = cli.main([str(existing), "-f", "bogus"])

    assert code == 1
    assert "Неизвестный формат" in capsys.readouterr().err


def test_main_max_speakers_zero_returns_1_before_pipeline(tmp_path, capsys):
    """Регресс: раньше --max-speakers 0 доходил до
    diarization.estimate_speakers и падал там на argmax пустой
    последовательности. Теперь отвергается на границе CLI."""
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")

    code = cli.main([str(existing), "--max-speakers", "0"])

    assert code == 1
    assert "--max-speakers" in capsys.readouterr().err


def test_main_max_speakers_negative_returns_1(tmp_path, capsys):
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")

    code = cli.main([str(existing), "--max-speakers", "-1"])

    assert code == 1
    assert "--max-speakers" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-3"])
def test_main_speakers_below_one_returns_1_before_pipeline(tmp_path, monkeypatch, capsys, value):
    """Регресс: --speakers 0 и --speakers -3 раньше не падали, а молча давали
    стенограмму на ОДНОГО участника — cluster() отвечает нулями на любое
    num_speakers <= 1. Воспроизведено на синтетическом звуке: 3 участника при
    отсутствии флага и 1 при обоих этих значениях, причём узнать об этом можно
    было только по готовому файлу, после полного прохода. Теперь отказ идёт на
    границе CLI: audio.extract_wav замокан так, чтобы падать, если пайплайн
    всё-таки начнётся."""

    def boom(*args, **kwargs):
        raise RuntimeError("сюда пайплайн специально не должен дойти")

    monkeypatch.setattr(audio, "extract_wav", boom)
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")

    code = cli.main([str(existing), "--speakers", value])
    captured = capsys.readouterr()

    assert code == 1
    assert "--speakers" in captured.err
    assert value in captured.err
    assert "[1/4]" not in captured.out


def test_main_speakers_non_integer_is_rejected_by_argparse(tmp_path, capsys):
    """--speakers 0.5 отсекает сам argparse (type=int) — до тела main(), с
    кодом 2 и своим сообщением. Отдельной проверки в main() для этого не
    нужно, и SystemExit здесь не ошибка пайплайна: тест фиксирует, что
    дробное значение не долетает до валидации «не меньше 1» и не выглядит
    как отказ пайплайна."""
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")

    with pytest.raises(SystemExit) as exit_info:
        cli.main([str(existing), "--speakers", "0.5"])

    assert exit_info.value.code == 2
    assert "invalid int value" in capsys.readouterr().err


def test_main_speakers_one_is_accepted(fake_pipeline):
    """Граница проверки: 1 — допустимое значение, а не отвергнутое заодно
    с 0 и -3."""
    assert fake_pipeline.run("--speakers", "1") == 0


def test_main_bad_names_value_error_caught_by_generic_handler(tmp_path, monkeypatch, capsys):
    """parse_names("0=...") бросает ValueError уже ПОСЛЕ проверок формата
    и max-speakers, но ДО тяжёлого пайплайна — этот путь идёт через общий
    перехватчик, а не через отдельную ветку в main().

    Регресс, найденный ревью: прежняя версия теста не различала «ValueError
    из parse_names поймана до пайплайна» и «пайплайн честно провалился на
    пустом .mp4» — оба дают код 1 и stderr, начинающийся с «Ошибка:».
    Мутация (parse_names вызывается ПОСЛЕ align.assign, как раньше) набор
    не красила: audio.extract_wav на пустом .mp4 тоже падает RuntimeError
    с тем же форматом сообщения. Здесь audio.extract_wav замокан так, чтобы
    падать с ДРУГИМ, узнаваемым текстом, если код всё-таки дойдёт до
    пайплайна, — тест ловит источник ошибки, а не только код возврата и
    префикс сообщения."""
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")

    def boom(*args, **kwargs):
        raise RuntimeError("сюда пайплайн специально не должен дойти")

    monkeypatch.setattr(audio, "extract_wav", boom)

    code = cli.main([str(existing), "--names", "0=Судья"])
    captured = capsys.readouterr()

    assert code == 1
    assert captured.err.startswith("Ошибка:")
    assert "нумерация с 1" in captured.err  # это именно ошибка parse_names, а не боль от boom
    assert "[1/4]" not in captured.out  # тяжёлый пайплайн не должен был даже начаться


# ---------------------------------------------------------------------------
# main(): --tile-speakers — на границе CLI и после шага [2/4]
# ---------------------------------------------------------------------------


def test_main_tile_speakers_invalid_value_returns_1_before_pipeline(tmp_path, monkeypatch, capsys):
    """parse_tile_speakers валидирует значения (>= 1) сразу при разборе
    аргументов — опечатка в --tile-speakers не должна стоить пользователю
    самого долгого этапа пайплайна. audio.extract_wav замокан так, чтобы
    падать, если код всё-таки туда дойдёт по-настоящему."""
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")

    def boom(*args, **kwargs):
        raise RuntimeError("сюда пайплайн специально не должен дойти")

    monkeypatch.setattr(audio, "extract_wav", boom)

    code = cli.main([str(existing), "--tile-speakers", "0"])
    err = capsys.readouterr().err

    assert code == 1
    assert "--tile-speakers" in err


def test_main_tile_speakers_length_mismatch_after_step_2_returns_1(tmp_path, monkeypatch, capsys):
    """Регресс: число плиток известно только ПОСЛЕ шага [2/4]
    (video.find_tiles), поэтому длину --tile-speakers сверяют именно
    здесь — раньше несовпадение длины проявлялось намного позже, внутри
    diarization.diarize_with_video, потратив время на распознавание
    (диаризация здесь вообще не должна запуститься — asr.transcribe
    замокан так, чтобы упасть, если код всё-таки дойдёт до шага [3/4]).
    video.has_video/find_tiles замокан, чтобы не тянуть настоящий ffmpeg
    и видео с настоящей подсветкой — нас интересует только сверка длины."""
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")

    monkeypatch.setattr(audio, "extract_wav", lambda src, dst, denoise=False: dst)
    monkeypatch.setattr(audio, "probe_duration", lambda path: 1.0)
    monkeypatch.setattr(video, "has_video", lambda *_: True)
    monkeypatch.setattr(video, "find_tiles", lambda *_, **__: [video.Tile(0, 0, 3, 10), video.Tile(20, 0, 3, 10)])

    def boom(*args, **kwargs):
        raise RuntimeError("сюда пайплайн специально не должен дойти")

    monkeypatch.setattr(asr, "transcribe", boom)

    code = cli.main([str(existing), "--tile-speakers", "1,1,1"])
    err = capsys.readouterr().err

    assert code == 1
    assert "--tile-speakers" in err
    assert "передано — 3" in err
    assert "найдено — 2" in err


def test_main_tile_speakers_by_found_tiles_passes_through_when_highlight_of_one_tile_dies(fake_pipeline, capsys):
    """БЛОКЕР ПУБЛИКАЦИИ, воспроизведённый целиком: в кадре найдено 3 плитки,
    до разметки участников дожила подсветка только двух — и раньше флаг
    --tile-speakers нельзя было удовлетворить ни одним значением:

        --tile-speakers 1,1    → [2/4] «передано 2, а плиток найдено 3»
        --tile-speakers 1,1,1  → [2/4] ок → [3/4] распознавание (35 минут
                                 на часовой записи) → «передано 3, а плиток
                                 с пойманной подсветкой 2»

    Расхождение штатное: подсветка средней плитки живёт 1.2 с, после сдвига
    на --highlight-lag её границы схлопываются притяжением к паузе, и в
    video.to_turns плитка исчезает. Ни ffmpeg, ни ONNX для этого не нужны —
    to_turns здесь работает по-настоящему, что и проверяется ассертами про
    3 найденных и 2 доживших.

    Правильное поведение: числа нумеруются по НАЙДЕННЫМ плиткам, поэтому
    трёх чисел достаточно, а число за не дожившей плиткой просто не
    используется — прогон доходит до конца и пишет файл."""
    code = fake_pipeline.run("--tile-speakers", "1,1,1")
    out = capsys.readouterr().out

    assert len(fake_pipeline.tiles) == 3
    assert sorted({t.speaker for t in fake_pipeline.video_turns}) == [0, 2]  # подсветка плитки 2 не дожила
    assert code == 0
    assert "Готово за" in out


def test_main_tile_speakers_counted_by_surviving_tiles_is_rejected_before_transcription(fake_pipeline, capsys):
    """Обратная сторона той же семантики: счёт по ДОЖИВШИМ плиткам (два числа
    на три найденные) отвергается — и отвергается на шаге [2/4], до
    распознавания, то есть без потери 35 минут. Пользователю остаётся ровно
    один правильный ответ: по числу плиток, которое напечатал шаг [2/4]."""
    code = fake_pipeline.run("--tile-speakers", "1,1")
    captured = capsys.readouterr()

    assert code == 1
    assert "передано — 2" in captured.err
    assert "найдено — 3" in captured.err
    assert "[3/4]" not in captured.out  # до распознавания дело не дошло


def test_main_tile_speakers_without_tiles_says_so_instead_of_silently_ignoring(fake_pipeline, monkeypatch, capsys):
    """Плиток нет вовсе (аудиофайл, --no-video, подсветка не найдена) — делить
    по плиткам нечего, и --tile-speakers не используется. Раньше об этом
    нельзя было догадаться иначе как по числу участников в итоговой сводке."""
    monkeypatch.setattr(video, "has_video", lambda path: False)
    monkeypatch.setattr(diarization, "diarize", lambda *a, **k: [diarization.Turn(0.0, 20.0, 0)])

    code = fake_pipeline.run("--tile-speakers", "3,1,1")
    out = capsys.readouterr().out

    assert code == 0
    assert "--tile-speakers не пригодился" in out


def test_main_tile_speakers_number_of_dead_tile_is_ignored_not_shifted(fake_pipeline, monkeypatch, capsys):
    """Число за плиткой, чья подсветка не дожила, обязано именно НЕ
    ИСПОЛЬЗОВАТЬСЯ, а не сдвигаться на соседку. Второе число здесь — 9;
    если бы числа брались по позиции среди доживших, плитку 2 попытались бы
    разделить на девятерых — то есть дошли бы до голосовых отпечатков.
    diarization.embed замокан так, чтобы это было видно сразу и чтобы тест
    ни при каком исходе не потянул настоящий ONNX."""

    def boom(*args, **kwargs):
        raise AssertionError("k=1 у обеих доживших плиток: до отпечатков дело доходить не должно")

    monkeypatch.setattr(diarization, "embed", boom)

    code = fake_pipeline.run("--tile-speakers", "1,9,1")
    out = capsys.readouterr().out

    assert code == 0
    assert "участников: 2" in out


# ---------------------------------------------------------------------------
# main(): каталог вывода создаётся при разборе аргументов (регресс)
# ---------------------------------------------------------------------------


def test_main_creates_output_directory_before_pipeline_starts(tmp_path, monkeypatch, capsys):
    """Регресс: раньше `-o несуществующий/каталог/файл.расширение` падал на
    последнем шаге (render.write) с ENOENT, потратив на часовой записи
    полчаса впустую — mkdir стоял только в ветке «-o без расширения»
    (stem = output / input.stem), а в ветке «-o с расширением»
    (stem = output.with_suffix("")) каталог заранее не создавался вовсе.

    Регресс, найденный ревью: прежняя версия теста передавала в -o каталог
    БЕЗ расширения — а это ровно та ветка, которая и в сломанном коде
    создавала каталог как побочный эффект (`stem = output / input.stem` в
    паре с mkdir внутри той же ветки else); настоящий баг жил в ветке С
    расширением, и старый тест её не задевал вовсе. Мутация (mkdir внутрь
    ветки else, а не отдельной строкой после всего if/elif/else) набор не
    красила.

    Ниже -o указывает на путь С расширением: если mkdir снова окажется
    только в ветке без расширения (или переедет к концу пайплайна),
    каталог не будет создан до того, как замоканный audio.extract_wav
    оборвёт прогон, — и assert output_dir.is_dir() провалится."""
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")
    output_dir = tmp_path / "нет" / "такого" / "каталога"
    output_path = output_dir / "результат.txt"

    def boom(*args, **kwargs):
        raise RuntimeError("сюда пайплайн специально не должен дойти по-настоящему")

    monkeypatch.setattr(audio, "extract_wav", boom)

    code = cli.main([str(existing), "-o", str(output_path)])

    assert output_dir.is_dir()
    assert code == 1
    assert capsys.readouterr().err.startswith("Ошибка:")


# ---------------------------------------------------------------------------
# main(): общий перехватчик исключений (регресс)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("ffmpeg не смог извлечь аудио"),
        ValueError("неверный код языка"),
        OSError("диск переполнен"),
        URLError("нет сети для скачивания модели"),
        tarfile.ReadError("прокси отдал HTML вместо архива модели"),
        subprocess.CalledProcessError(1, ["ffmpeg"]),
        wave.Error("WAV повреждён или нечитаем"),
        # Подкласс RuntimeError с прежним текстом: вывод CLI для тишины не изменился.
        diarization.NoSpeech("В записи не найдено речи достаточной длительности"),
    ],
    ids=["RuntimeError", "ValueError", "OSError", "URLError_subclass_of_OSError", "tarfile.TarError", "subprocess.SubprocessError", "wave.Error", "diarization.NoSpeech"],
)
def test_main_generic_handler_catches_expected_exceptions_without_traceback(tmp_path, monkeypatch, capsys, exc):
    """Регресс: раньше перехватчик стоял только вокруг video.find_tiles,
    остальное падало сырым трейсбеком. Теперь любое из этих исключений на
    любом шаге пайплайна даёт сообщение в stderr и код 1 — включая три
    класса, которые НЕ наследуются от OSError и потому раньше были бы вне
    перехватчика (tarfile.TarError, subprocess.SubprocessError, wave.Error).

    Раньше здесь стояла проверка `err.strip().count("\\n") == 0` («одна
    строка, не трейсбек») — она зеленела ТОЛЬКО потому, что все
    параметризованные исключения были однострочными: тест проверял не то,
    что заявлено в названии. Настоящий первый триггер общего перехватчика
    (см. test_main_keeps_multiline_ffmpeg_error_whole ниже) даёт четырёхстрочное сообщение — audio.extract_wav вкладывает
    stderr ffmpeg целиком. Перехватчик не обязан схлопывать сообщение в
    одну строку — обязан только не превращать его в трейсбек и не терять
    исходный текст. Здесь проверяется именно это: сообщение сохранено
    целиком (сколько бы строк в нём ни было) и "Traceback" не выводится.

    Вторая строка вывода — подсказка про STENOGRAM_TRACEBACK. Она часть
    сообщения об ошибке, а не отдельный случай: без неё баг-репорт от
    постороннего человека состоит из одной непонятной фразы."""
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")

    def raiser(*args, **kwargs):
        raise exc

    monkeypatch.setattr(audio, "extract_wav", raiser)

    code = cli.main([str(existing)])
    err = capsys.readouterr().err

    assert code == 1
    assert err.startswith(f"Ошибка: {exc}\n")
    assert "STENOGRAM_TRACEBACK=1" in err
    assert "Traceback" not in err


def test_url_error_is_already_covered_by_os_error_in_generic_handler():
    """URLError — подкласс OSError из стандартной библиотеки, поэтому в
    кортеже перехватчика main() он не открывает отдельный путь исполнения:
    любой URLError и так ловится веткой OSError (перехват идёт через
    isinstance по всему кортежу, а не по порядку). Параметризованный случай
    URLError выше сохранён не ради отдельной ветки except (её нет и не
    нужно), а ради характерного вида __str__ (`<urlopen error ...>`) —
    этот тест фиксирует, ПОЧЕМУ так, чтобы не создавалось впечатления,
    что проверяется путь, отличный от OSError."""
    assert issubclass(URLError, OSError)


def test_main_broken_pipe_error_returns_0_without_error_message(tmp_path, monkeypatch, capsys):
    """BrokenPipeError — не отказ, а штатное завершение конвейера (например,
    `stenogram запись.mp4 | head` перестаёт читать раньше времени). Ветка
    стоит ВЫШЕ общего перехватчика и обязана вернуть код 0, ничего не
    печатая в stderr, — в отличие от любого другого OSError (подклассом
    которого BrokenPipeError и является), который ушёл бы в "Ошибка: ..."
    и код 1, если бы стоял ниже. sys.stdout под capsys не имеет настоящего
    файлового дескриптора (fileno() бросает io.UnsupportedOperation —
    подкласс и OSError, и ValueError сразу), поэтому код идёт по ветке
    "перехват в тестах" и возвращает 0 сразу, не пытаясь дублировать fd."""
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")

    def raiser(*args, **kwargs):
        raise BrokenPipeError

    monkeypatch.setattr(audio, "extract_wav", raiser)

    code = cli.main([str(existing)])
    err = capsys.readouterr().err

    assert code == 0
    assert err == ""


def test_main_keyboard_interrupt_is_not_caught_by_generic_handler(tmp_path, monkeypatch, capsys):
    """Регресс/защита от будущей поломки: KeyboardInterrupt обязан давать
    код 130 и своё сообщение — а не тихо проглатываться общим перехватчиком
    (RuntimeError/ValueError/OSError/URLError), что вернуло бы код 1."""
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")

    def raiser(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(audio, "extract_wav", raiser)

    code = cli.main([str(existing)])
    err = capsys.readouterr().err

    assert code == 130
    assert "Прервано пользователем" in err
    assert "Ошибка:" not in err


# ---------------------------------------------------------------------------
# main(): STENOGRAM_TRACEBACK — способ получить трейсбек для баг-репорта
# ---------------------------------------------------------------------------


def test_main_traceback_env_var_lets_exception_through(tmp_path, monkeypatch):
    """Перехватчик покрывает всё тело main(), поэтому без такого выключателя
    у постороннего человека нет никакого способа увидеть, ГДЕ упало:
    RecursionError (подкласс RuntimeError) и ошибки numpy (ValueError)
    превращаются в одну строку. С переменной исключение выходит наружу как
    есть — трейсбек печатает сам интерпретатор."""
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")
    monkeypatch.setenv(cli._TRACEBACK_ENV, "1")

    def raiser(*args, **kwargs):
        raise RuntimeError("maximum recursion depth exceeded")

    monkeypatch.setattr(audio, "extract_wav", raiser)

    with pytest.raises(RuntimeError, match="maximum recursion depth"):
        cli.main([str(existing)])


@pytest.mark.parametrize("value", ["", "0"])
def test_main_traceback_env_var_off_values_keep_one_line_message(tmp_path, monkeypatch, capsys, value):
    """Пустая строка и «0» — это выключено. Иначе `STENOGRAM_TRACEBACK=0`,
    написанное по привычке из других инструментов, включало бы режим."""
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")
    monkeypatch.setenv(cli._TRACEBACK_ENV, value)

    def raiser(*args, **kwargs):
        raise RuntimeError("что-то пошло не так")

    monkeypatch.setattr(audio, "extract_wav", raiser)

    code = cli.main([str(existing)])

    assert code == 1
    assert capsys.readouterr().err.startswith("Ошибка: что-то пошло не так\n")


def test_main_traceback_env_var_does_not_change_keyboard_interrupt(tmp_path, monkeypatch, capsys):
    """Прерывание пользователем — не отказ и не предмет баг-репорта: режим
    отладки не должен превращать Ctrl+C в трейсбек и менять код 130."""
    existing = tmp_path / "запись.mp4"
    existing.write_bytes(b"")
    monkeypatch.setenv(cli._TRACEBACK_ENV, "1")

    def raiser(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(audio, "extract_wav", raiser)

    assert cli.main([str(existing)]) == 130
    assert "Прервано пользователем" in capsys.readouterr().err


def test_main_traceback_env_var_does_not_make_video_failure_fatal(fake_pipeline, monkeypatch, capsys):
    """Поиск плиток падать не имеет права: не нашлось — переходим на голос.
    Режим отладки это правило не отменяет, а только объясняет — трейсбек
    печатается, прогон продолжается и заканчивается кодом 0."""
    monkeypatch.setenv(cli._TRACEBACK_ENV, "1")

    def raiser(*args, **kwargs):
        raise RuntimeError("видео нестандартное")

    monkeypatch.setattr(video, "find_tiles", raiser)
    monkeypatch.setattr(diarization, "diarize", lambda *a, **k: [diarization.Turn(0.0, 20.0, 0)])

    code = fake_pipeline.run()
    captured = capsys.readouterr()

    assert code == 0
    assert "останусь на звуке" in captured.out
    assert "Traceback" in captured.err


# ---------------------------------------------------------------------------
# main(): stdout сброшен до return 0 (регресс на код возврата 120)
# ---------------------------------------------------------------------------


class _RecordingStdout:
    """stdout, который запоминает порядок операций write/flush.

    Настоящую трубу в тесте не воспроизвести (capsys подменяет stdout без
    файлового дескриптора), но проверить надо не саму трубу, а инвариант,
    из которого росла проблема: к моменту выхода из main() в буфере stdout
    не должно оставаться ничего ненапечатанного."""

    def __init__(self):
        self.ops: list[str] = []
        self.text = ""

    def write(self, text: str) -> int:
        self.ops.append("write")
        self.text += text
        return len(text)

    def flush(self) -> None:
        self.ops.append("flush")

    def fileno(self):
        raise OSError("нет файлового дескриптора, как под capsys")


def test_main_flushes_stdout_inside_try_before_returning_zero(fake_pipeline, monkeypatch):
    """Регресс на код возврата 120. Финальные print (сводка по участникам и
    список файлов) шли без flush, поэтому остаток буфера уходил в трубу уже
    при завершении интерпретатора — вне try, вне ветки BrokenPipeError.
    Прогон `stenogram … | head -1` давал «Exception ignored» в stderr и код
    120 в 30 случаях из 30; после правки — ни одного из 30.

    Здесь проверяется причина, а не следствие: последняя операция над
    stdout — flush, то есть ничего ненапечатанного не остаётся."""
    recorder = _RecordingStdout()
    monkeypatch.setattr(cli.sys, "stdout", recorder)

    code = fake_pipeline.run()

    assert code == 0
    assert recorder.ops[-1] == "flush"


def test_main_broken_pipe_on_final_block_is_handled_not_left_to_interpreter(fake_pipeline, monkeypatch):
    """Вторая половина того же регресса, ближе к настоящей трубе: получатель
    (`head`) уходит ровно на финальном блоке — сброс буфера с этого момента
    бросает BrokenPipeError.

    Ожидается, что EPIPE случится ВНУТРИ main() (последняя операция над
    stdout — flush, а не write) и будет обработан веткой BrokenPipeError с
    кодом 0. Раньше сбрасывать было нечего и некому: финальный блок оставался
    в буфере, а EPIPE прилетал уже интерпретатору при выходе — «Exception
    ignored» и код возврата 120."""

    class _ClosedPipeStdout(_RecordingStdout):
        def flush(self):
            super().flush()
            if "Готово за" in self.text:
                raise BrokenPipeError(32, "Broken pipe")

    recorder = _ClosedPipeStdout()
    monkeypatch.setattr(cli.sys, "stdout", recorder)

    code = fake_pipeline.run()

    assert code == 0
    assert recorder.ops[-1] == "flush"


# ---------------------------------------------------------------------------
# main(): настоящий (не смоделированный) четырёхстрочный триггер перехватчика
# ---------------------------------------------------------------------------


def test_main_keeps_multiline_ffmpeg_error_whole(tmp_path, monkeypatch, capsys):
    """audio.extract_wav вкладывает stderr ffmpeg целиком, и он многострочный. Перехватчик
    обязан выдать его как есть, без трейсбека. ffmpeg подменён; то же на настоящем ffmpeg —
    tests/integration/test_ffmpeg.py."""
    source = tmp_path / "без_звука.mp4"
    source.write_bytes(b"")
    stderr = "Output file does not contain any stream\nError opening output files\nConversion failed!"
    monkeypatch.setattr(audio, "find_ffmpeg", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(audio.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, "", stderr))

    code = cli.main([str(source)])
    err = capsys.readouterr().err

    assert code == 1
    assert err.startswith("Ошибка: ffmpeg не смог извлечь аудио")
    assert stderr in err
    assert "Traceback" not in err
