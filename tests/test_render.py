"""Тесты форматов выгрузки (stenogram.render).

Приоритет — таймкоды на границах, переход индекса с нуля на единицу, и
нарезка длинных реплик для субтитров: правила зафиксированы в коде, а не
где-либо ещё, и легко ломаются при рефакторинге.
"""

from __future__ import annotations

import json

import pytest

from stenogram.align import Utterance
from stenogram.asr import Word
from stenogram.render import WRITERS, _split, hhmmss, speaker_name, to_json, to_markdown, to_srt, to_text, to_vtt, write

# ---------------------------------------------------------------------------
# hhmmss(): границы
# ---------------------------------------------------------------------------


def test_hhmmss_zero():
    assert hhmmss(0.0) == "00:00:00"


def test_hhmmss_negative_clamped_to_zero():
    """max(seconds, 0.0) — отрицательное время не должно давать «минус» ни
    в каком виде и не должно бросать исключение."""
    assert hhmmss(-5.0) == "00:00:00"


def test_hhmmss_over_one_hour():
    assert hhmmss(3725.0) == "01:02:05"


def test_hhmmss_exactly_one_hour():
    assert hhmmss(3600.0) == "01:00:00"


def test_hhmmss_millis_dot_separator_by_default():
    assert hhmmss(1.234, millis=True) == "00:00:01.234"


def test_hhmmss_millis_comma_separator_for_srt():
    assert hhmmss(1.234, millis=True, comma=True) == "00:00:01,234"


def test_hhmmss_millis_truncates_not_rounds():
    """int() усечение, а не round(): 0.6 мс должны уйти в 0, а не
    округлиться вверх до 1.

    Регресс, найденный ревью: прежнее значение — 0.0005 (0.5 мс) — не
    различает int() и round(): int(0.5) == 0, и round(0.5) == 0 тоже —
    банковское округление (round-half-to-even) в самой Python округляет
    0.5 к ближайшему чётному, то есть вниз. Мутация `int(secs % 1 * 1000)`
    -> `round(secs % 1 * 1000)` на прежнем значении не красила набор.
    0.0006 различает честно: int(0.6) == 0, round(0.6) == 1."""
    assert hhmmss(0.0006, millis=True) == "00:00:00.000"


def test_hhmmss_millis_over_hour_with_fraction():
    assert hhmmss(3725.25, millis=True, comma=True) == "01:02:05,250"


# ---------------------------------------------------------------------------
# speaker_name(): индекс с нуля -> показ с единицы
# ---------------------------------------------------------------------------


def test_speaker_name_uses_provided_name():
    assert speaker_name(0, {0: "Судья"}) == "Судья"


def test_speaker_name_zero_indexed_becomes_one_indexed_default():
    assert speaker_name(0, None) == "Говорящий 1"


def test_speaker_name_index_missing_from_names_falls_back_to_default():
    assert speaker_name(2, {0: "Судья"}) == "Говорящий 3"


def test_speaker_name_empty_names_dict_falls_back_to_default():
    assert speaker_name(0, {}) == "Говорящий 1"


# ---------------------------------------------------------------------------
# to_text()
# ---------------------------------------------------------------------------


def test_to_text_empty_utterances_returns_just_newline():
    assert to_text([]) == "\n"


def test_to_text_basic_with_timestamp():
    u = Utterance(65.0, 70.0, 0, "привет")
    assert to_text([u]) == "[00:01:05] Говорящий 1: привет\n"


def test_to_text_without_timestamp():
    u = Utterance(65.0, 70.0, 0, "привет")
    assert to_text([u], timestamps=False) == "Говорящий 1: привет\n"


def test_to_text_joins_multiple_utterances_with_blank_line():
    u1 = Utterance(0.0, 1.0, 0, "раз")
    u2 = Utterance(1.0, 2.0, 1, "два")
    text = to_text([u1, u2])
    assert text == "[00:00:00] Говорящий 1: раз\n\n[00:00:01] Говорящий 2: два\n"


# ---------------------------------------------------------------------------
# to_markdown()
# ---------------------------------------------------------------------------


def test_to_markdown_includes_meta_and_participant_table():
    u = Utterance(0.0, 60.0, 0, "привет")
    text = to_markdown([u], meta={"Источник": "запись.mp4"})
    assert "- **Источник:** запись.mp4" in text
    assert "## Участники" in text
    assert "Говорящий 1" in text
    assert "## Диалог" in text
    assert "привет" in text


def test_to_markdown_without_meta_has_no_meta_lines():
    u = Utterance(0.0, 1.0, 0, "привет")
    text = to_markdown([u], meta=None)
    assert "**Источник" not in text
    assert "## Участники" in text


def test_to_markdown_empty_utterances_does_not_crash():
    text = to_markdown([])
    assert "## Участники" in text
    assert "## Диалог" in text


# ---------------------------------------------------------------------------
# _split(): без слов не режет, короткий текст не режет, накопление порога
# ---------------------------------------------------------------------------


def test_split_without_words_yields_single_unsplit_block_even_if_long():
    """`not utterance.words` — реплика без потайминга по словам не режется,
    даже если текст длиннее max_chars: резать было бы нечем."""
    long_text = "а" * 200
    u = Utterance(0.0, 5.0, 0, long_text, words=[])

    chunks = list(_split(u, max_chars=90))

    assert len(chunks) == 1
    assert chunks[0] == (0.0, 5.0, long_text)


def test_split_short_text_not_split():
    u = Utterance(0.0, 1.0, 0, "коротко", words=[Word(0.0, 1.0, " коротко")])

    chunks = list(_split(u, max_chars=90))

    assert chunks == [(0.0, 1.0, "коротко")]


def test_split_threshold_is_accumulation_stop_not_hard_ceiling():
    """max_chars — точка остановки накопления, а не потолок: второй кусок
    может ПРЕВЫСИТЬ max_chars, если последнее добавленное слово длинное."""
    w1 = Word(0.0, 1.0, " aaaaaaaaaa")  # 10 непробельных символов, триггерит сразу
    w2 = Word(1.0, 2.0, " bb")  # короткое, накапливается
    w3 = Word(2.0, 3.0, " cccccccccccc")  # длинное — толкает второй кусок за порог
    full_text = "".join(w.text for w in (w1, w2, w3)).strip()
    u = Utterance(0.0, 3.0, 0, full_text, words=[w1, w2, w3])

    chunks = list(_split(u, max_chars=10))

    assert len(chunks) == 2
    assert chunks[0] == (0.0, 1.0, "aaaaaaaaaa")
    assert chunks[1] == (1.0, 3.0, "bb cccccccccccc")
    assert len(chunks[1][2]) > 10  # порог превышен, а не соблюдён как потолок


def test_split_flushes_exactly_when_buffer_reaches_threshold():
    """Гонка `>=` против `>` в условии переноса. Ревью нашло, что тест выше
    (test_split_threshold_is_accumulation_stop_not_hard_ceiling) не ловит
    подмену `>=` на `>`: там первое же слово (11 символов с пробелом) уже
    ПРЕВЫШАЕТ max_chars=10 на единицу, и `>=`/`>` в этом случае неотличимы.

    Здесь буфер после первого слова достигает max_chars РОВНО (10 из 10) —
    перенос обязан случиться уже в этой точке. При `>` вместо `>=` слово
    осталось бы в буфере, оба слова склеились бы в один кусок вместо двух."""
    w1 = Word(0.0, 1.0, " aaaaaaaaa")  # пробел + 9 «a» = 10 символов — ровно max_chars
    w2 = Word(1.0, 2.0, " b")
    full_text = "".join(w.text for w in (w1, w2)).strip()
    u = Utterance(0.0, 2.0, 0, full_text, words=[w1, w2])

    chunks = list(_split(u, max_chars=10))

    assert chunks == [(0.0, 1.0, "aaaaaaaaa"), (1.0, 2.0, "b")]


def test_split_flushes_trailing_buffer_below_threshold():
    """Слова, не добравшие до порога к концу реплики, обязаны попасть в
    финальный кусок через `if buffer: yield ...` после цикла."""
    w1 = Word(0.0, 1.0, " aaaaaaaaaa")  # триггерит первый кусок
    w2 = Word(1.0, 2.0, " хвост")  # остаётся в буфере после цикла
    full_text = "".join(w.text for w in (w1, w2)).strip()
    u = Utterance(0.0, 2.0, 0, full_text, words=[w1, w2])

    chunks = list(_split(u, max_chars=10))

    assert chunks[-1] == (1.0, 2.0, "хвост")


# ---------------------------------------------------------------------------
# to_srt() / to_vtt(): сквозная нумерация, заголовок, теги говорящего
# ---------------------------------------------------------------------------


def test_to_srt_sequential_numbering_across_split_and_next_utterance():
    w1 = Word(0.0, 1.0, " aaaaaaaaaa")
    w2 = Word(1.0, 2.0, " bb")
    w3 = Word(2.0, 3.0, " cccccccccccc")
    full_text = "".join(w.text for w in (w1, w2, w3)).strip()
    long_utterance = Utterance(0.0, 3.0, 0, full_text, words=[w1, w2, w3])
    next_utterance = Utterance(3.5, 4.0, 1, "ок", words=[Word(3.5, 4.0, " ок")])

    srt = to_srt([long_utterance, next_utterance], max_chars=10)
    blocks = srt.split("\n\n")

    assert blocks[0].startswith("1\n")
    assert blocks[1].startswith("2\n")
    assert blocks[2].startswith("3\n")
    assert "Говорящий 2: ок" in blocks[2]


def test_to_srt_uses_comma_millis_separator():
    u = Utterance(1.5, 2.5, 0, "текст", words=[Word(1.5, 2.5, " текст")])
    srt = to_srt([u])
    assert "00:00:01,500 --> 00:00:02,500" in srt


def test_to_vtt_has_webvtt_header():
    vtt = to_vtt([])
    assert vtt.startswith("WEBVTT\n")


def test_to_vtt_uses_dot_millis_separator_and_voice_tag():
    u = Utterance(1.5, 2.5, 0, "текст", words=[Word(1.5, 2.5, " текст")])
    vtt = to_vtt([u])
    assert "00:00:01.500 --> 00:00:02.500" in vtt
    assert "<v Говорящий 1>текст" in vtt


# ---------------------------------------------------------------------------
# to_json(): ведущий пробел в словах, отсутствие ascii-эскейпа
# ---------------------------------------------------------------------------


def test_to_json_strips_leading_space_only_at_word_level():
    u = Utterance(0.0, 1.0, 0, "привет", words=[Word(0.0, 1.0, " привет")])
    payload = json.loads(to_json([u]))

    assert payload["utterances"][0]["words"][0]["word"] == "привет"


def test_to_json_does_not_escape_cyrillic():
    u = Utterance(0.0, 1.0, 0, "привет", words=[Word(0.0, 1.0, " привет")])
    raw = to_json([u])

    assert "привет" in raw
    assert "\\u" not in raw


def test_to_json_includes_meta_and_speaker_stats():
    u = Utterance(0.0, 2.0, 0, "привет", words=[])
    payload = json.loads(to_json([u], meta={"Модель": "tiny"}))

    assert payload["meta"] == {"Модель": "tiny"}
    assert payload["speakers"][0]["id"] == 0
    assert payload["speakers"][0]["name"] == "Говорящий 1"
    assert payload["utterances"][0]["speaker_id"] == 0


def test_to_json_empty_utterances_has_empty_sections():
    payload = json.loads(to_json([]))
    assert payload["speakers"] == []
    assert payload["utterances"] == []


def test_to_json_rounds_timestamps():
    u = Utterance(1.23456, 2.34567, 0, "x", words=[Word(1.23456, 2.34567, " x")])
    payload = json.loads(to_json([u]))
    assert payload["utterances"][0]["start"] == 1.23
    assert payload["utterances"][0]["end"] == 2.35


# ---------------------------------------------------------------------------
# WRITERS / write(): неизвестный формат, реально пишущиеся файлы
# ---------------------------------------------------------------------------


def test_writers_dict_has_all_five_known_formats():
    assert set(WRITERS) == {"txt", "md", "srt", "vtt", "json"}


def test_write_unknown_format_raises_value_error(tmp_path):
    u = Utterance(0.0, 1.0, 0, "x")
    with pytest.raises(ValueError, match="Неизвестный формат"):
        write([u], tmp_path / "out", ["bogus"])


def test_write_unknown_format_mentions_available_formats(tmp_path):
    u = Utterance(0.0, 1.0, 0, "x")
    with pytest.raises(ValueError, match="txt"):
        write([u], tmp_path / "out", ["bogus"])


def test_write_creates_files_with_correct_suffixes(tmp_path):
    u = Utterance(0.0, 1.0, 0, "привет")
    written = write([u], tmp_path / "стенограмма", ["txt", "json"])

    assert written == [tmp_path / "стенограмма.txt", tmp_path / "стенограмма.json"]
    for path in written:
        assert path.exists()
    assert "привет" in written[0].read_text(encoding="utf-8")


def test_write_stops_before_unknown_format_but_keeps_earlier_files(tmp_path):
    """write() не откатывает уже записанные файлы, если один из последующих
    форматов неизвестен, — документируем реальное поведение, не идеальное."""
    u = Utterance(0.0, 1.0, 0, "привет")
    with pytest.raises(ValueError):
        write([u], tmp_path / "стенограмма", ["txt", "bogus"])

    assert (tmp_path / "стенограмма.txt").exists()
    assert not (tmp_path / "стенограмма.bogus").exists()
