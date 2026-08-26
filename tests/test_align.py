"""Тесты сшивки текста с разметкой говорящих (stenogram.align).

align.assign — сердце сшивки, чистая функция без внешних зависимостей.
Каждый случай ниже воспроизведён и сверен вручную на текущем коде перед
записью (см. .docs/plans/2026-08-25_публикация.md, раздел 5).
"""

from __future__ import annotations

import random

from stenogram.align import Utterance, _absorb_short, _smooth, _speaker_at, assign, speaker_stats
from stenogram.asr import Segment, Word
from stenogram.diarization import Turn

# ---------------------------------------------------------------------------
# assign(): пустые входы
# ---------------------------------------------------------------------------


def test_assign_empty_turns_returns_one_utterance_per_segment(segment_factory, word_factory):
    """Без турнов assign() не разметку строит, а просто заворачивает каждый
    сегмент в Utterance говорящего 0 — ветка `if not turns` в коде."""
    seg1 = segment_factory(0.0, 1.0, words=[word_factory(0.0, 1.0, " раз")])
    seg2 = segment_factory(1.0, 2.0, words=[word_factory(1.0, 2.0, " два")])
    seg1.text, seg2.text = " раз", " два"

    result = assign([seg1, seg2], [])

    assert [u.speaker for u in result] == [0, 0]
    assert [u.text for u in result] == [" раз", " два"]
    assert result[0].words == seg1.words


def test_assign_empty_segments_returns_empty_list(turn_factory):
    """Турны есть, сегментов нет: цикл по segments не выполняется ни разу,
    tagged остаётся пустым — assign обязан вернуть [], а не упасть."""
    result = assign([], [turn_factory(0.0, 5.0, 0)])
    assert result == []


def test_assign_both_empty_returns_empty_list():
    assert assign([], []) == []


# ---------------------------------------------------------------------------
# _speaker_at(): перекрытие, ничья, слова вне всех реплик
# ---------------------------------------------------------------------------


def test_speaker_at_tie_goes_to_first_turn(word_factory, turn_factory):
    """Слово 0.5-1.5 перекрывает Turn(0,1) и Turn(1,2) ровно по 0.5 с каждый —
    строгое `>` в сравнении оставляет первого найденного победителем."""
    turns = [turn_factory(0.0, 1.0, speaker=0), turn_factory(1.0, 2.0, speaker=1)]
    starts = [t.start for t in turns]
    word = word_factory(0.5, 1.5)

    assert _speaker_at(turns, starts, word) == 0


def test_speaker_at_word_entirely_before_all_turns(word_factory, turn_factory):
    """Слово заканчивается до начала первой реплики: перекрытий нет вовсе,
    код обязан упасть на «ближайший по времени», а не бросить исключение."""
    turns = [turn_factory(5.0, 6.0, speaker=0), turn_factory(7.0, 8.0, speaker=1)]
    starts = [t.start for t in turns]
    word = word_factory(0.0, 1.0)

    assert _speaker_at(turns, starts, word) == 0


def test_speaker_at_word_entirely_after_all_turns(word_factory, turn_factory):
    turns = [turn_factory(5.0, 6.0, speaker=0), turn_factory(7.0, 8.0, speaker=1)]
    starts = [t.start for t in turns]
    word = word_factory(20.0, 21.0)

    assert _speaker_at(turns, starts, word) == 1


def test_speaker_at_zero_duration_word_falls_back_to_nearest(word_factory, turn_factory):
    """Слово нулевой длительности: overlap = min(end,t.end) - max(start,t.start)
    всегда получается 0.0 для любой реплики, что НЕ строго больше стартового
    best_overlap=0.0 — код обязан пройти через ветку «ближайший», а не
    случайно вернуть -1 (несуществующего говорящего)."""
    turns = [turn_factory(0.0, 2.0, speaker=7)]
    starts = [0.0]
    word = word_factory(1.0, 1.0)

    assert _speaker_at(turns, starts, word) == 7


# ---------------------------------------------------------------------------
# assign(): сегмент без слов, ведущий пробел
# ---------------------------------------------------------------------------


def test_assign_segment_with_text_no_words_synthesizes_single_word(turn_factory):
    """segment.words пуст, но text не пуст: assign обязан собрать одно
    синтетическое Word на весь сегмент (`words = segment.words or [...]`)."""
    turns = [turn_factory(0.0, 10.0, speaker=0)]
    seg = Segment(start=1.0, end=2.0, text="привет мир", words=[])

    result = assign([seg], turns)

    assert len(result) == 1
    assert result[0].text == "привет мир"
    assert len(result[0].words) == 1
    assert result[0].words[0].start == 1.0 and result[0].words[0].end == 2.0


def test_assign_strips_only_leading_space_of_whole_utterance(turn_factory):
    """whisper отдаёт текст слова с ведущим пробелом (.docs/domain.md). При
    склейке слова просто конкатенируются, и лишь итоговая строка целиком
    проходит .strip() — пробел между словами (из второго слова) остаётся."""
    turns = [turn_factory(0.0, 10.0, speaker=0)]
    seg = Segment(0.0, 1.0, "", [Word(0.0, 0.5, " привет"), Word(0.5, 1.0, " мир")])

    result = assign([seg], turns)

    assert len(result) == 1
    assert result[0].text == "привет мир"


# ---------------------------------------------------------------------------
# assign(): граница max_pause
# ---------------------------------------------------------------------------


def test_assign_pause_exactly_max_pause_stays_in_same_utterance(turn_factory):
    """Условие в коде — `<=`: пауза, РОВНО равная max_pause, ещё не рвёт реплику."""
    turns = [turn_factory(0.0, 10.0, speaker=0)]
    seg = Segment(0.0, 4.0, "", [Word(0.0, 1.0, " привет"), Word(3.0, 4.0, " мир")])

    result = assign([seg], turns, max_pause=2.0)

    assert len(result) == 1
    assert result[0].text == "привет мир"


def test_assign_pause_max_pause_plus_epsilon_splits_utterance(turn_factory):
    """Пауза чуть больше max_pause (на epsilon) обязана разорвать реплику —
    это и есть смысл строгого `<=` в паре с предыдущим тестом на границе."""
    turns = [turn_factory(0.0, 10.0, speaker=0)]
    seg = Segment(0.0, 4.0, "", [Word(0.0, 1.0, " привет"), Word(3.0001, 4.0, " мир")])

    result = assign([seg], turns, max_pause=2.0)

    assert len(result) == 2
    assert result[0].text == "привет"
    assert result[1].text == "мир"


# ---------------------------------------------------------------------------
# _smooth(): одиночные выбросы и выбросы длиннее окна
# ---------------------------------------------------------------------------


def test_smooth_single_outlier_is_flipped(word_factory):
    tagged = [(0, word_factory(0, 1)), (0, word_factory(1, 2)), (1, word_factory(2, 3)), (0, word_factory(3, 4)), (0, word_factory(4, 5))]

    result = _smooth(tagged)

    assert [s for s, _ in result] == [0, 0, 0, 0, 0]


def test_smooth_outlier_longer_than_window_is_not_flipped(word_factory):
    """Окно сглаживания по умолчанию — 2. Выброс длиной 3 не подпадает под
    правило «слева и справа одинаковая метка»: внутри самого выброса окно
    захватывает соседние элементы того же выброса, а не окружения."""
    tagged = [
        (0, word_factory(0, 1)),
        (0, word_factory(1, 2)),
        (1, word_factory(2, 3)),
        (1, word_factory(3, 4)),
        (1, word_factory(4, 5)),
        (0, word_factory(5, 6)),
        (0, word_factory(6, 7)),
    ]

    result = _smooth(tagged)

    assert [s for s, _ in result] == [0, 0, 1, 1, 1, 0, 0]


def test_smooth_default_window_is_two_not_one(word_factory):
    """Дефолт `window=2` в сигнатуре `_smooth` не закреплён явно ни одним
    тестом рядом. Ревью нашло, что test_smooth_outlier_longer_than_window_is_not_flipped
    даёт ОДИНАКОВЫЙ результат что при window=2, что при мутации на
    window=1 — выброс длиной 3 не сглаживается в обоих случаях, разницу не
    видно (проверено перебором вручную: для той последовательности result
    совпадает при window in {0,2,3,4,5} и отличается только при window=1,
    но именно от window=1 сам тест не отличается по итоговому списку).

    Чередование 0,1,0,1 — минимальный вход, который РЕАЛЬНО различает
    window=2 и window=1: при window=2 ни одна точка не сглаживается (окну
    длины 2 по обе стороны не хватает однородности — сосед по одну сторону
    всегда часть соседнего «переключения»), при window=1 чередование
    стирается в 0,0,1,1. window ниже не передаётся явно — тест бьёт именно
    по значению аргумента по умолчанию в сигнатуре _smooth."""
    tagged = [(0, word_factory(0, 1)), (1, word_factory(1, 2)), (0, word_factory(2, 3)), (1, word_factory(3, 4))]

    result = _smooth(tagged)

    assert [s for s, _ in result] == [0, 1, 0, 1]


def test_smooth_no_outliers_unchanged(word_factory):
    tagged = [(0, word_factory(0, 1)), (0, word_factory(1, 2)), (1, word_factory(2, 3)), (1, word_factory(3, 4))]

    result = _smooth(tagged)

    assert [s for s, _ in result] == [0, 0, 1, 1]


def test_smooth_boundary_first_element_no_left_neighbours(word_factory):
    """У первого элемента нет «левого» окна — код не должен упасть и не
    должен ничего сгладить (нечего сравнивать)."""
    tagged = [(1, word_factory(0, 1)), (0, word_factory(1, 2)), (0, word_factory(2, 3))]

    result = _smooth(tagged)

    assert [s for s, _ in result] == [1, 0, 0]


def test_smooth_boundary_last_element_no_right_neighbours(word_factory):
    tagged = [(0, word_factory(0, 1)), (0, word_factory(1, 2)), (1, word_factory(2, 3))]

    result = _smooth(tagged)

    assert [s for s, _ in result] == [0, 0, 1]


def test_smooth_empty_input():
    assert _smooth([]) == []


# ---------------------------------------------------------------------------
# _absorb_short(): микрореплики и пустой текст после обработки
# ---------------------------------------------------------------------------


def test_absorb_short_merges_microreplique_into_previous_and_chains_next(utterance_factory):
    """Микрореплика ("да", 0.1с — короче min_duration и <=3 символов) от
    ДРУГОГО говорящего склеивается с предыдущей репликой, а не остаётся
    самостоятельной. После этого следующая реплика того же (первого)
    говорящего с паузой <=2с тоже склеивается — итог: одна реплика."""
    u1 = utterance_factory(0.0, 1.0, speaker=0, text="привет", words=[Word(0.0, 1.0, " привет")])
    u2 = utterance_factory(1.0, 1.1, speaker=1, text="да", words=[Word(1.0, 1.1, " да")])
    u3 = utterance_factory(1.2, 2.0, speaker=0, text="пока", words=[Word(1.2, 2.0, " пока")])

    result = _absorb_short([u1, u2, u3], min_duration=0.35)

    assert len(result) == 1
    assert result[0].speaker == 0
    assert result[0].text == "привет да пока"
    assert result[0].start == 0.0 and result[0].end == 2.0


def test_absorb_short_microreplique_same_speaker_not_absorbed_by_first_branch(utterance_factory):
    """Первая ветка склейки требует РАЗНОГО говорящего (result[-1].speaker !=
    utterance.speaker). Если микрореплика — того же говорящего, что и
    предыдущая, первая ветка не сработает; но сработает вторая (соседняя
    реплика одного говорящего с паузой <=2с), и результат всё равно один
    объединённый Utterance — только через другой путь кода."""
    u1 = utterance_factory(0.0, 1.0, speaker=0, text="привет", words=[Word(0.0, 1.0, " привет")])
    u2 = utterance_factory(1.0, 1.1, speaker=0, text="да", words=[Word(1.0, 1.1, " да")])

    result = _absorb_short([u1, u2], min_duration=0.35)

    assert len(result) == 1
    assert result[0].text == "привет да"


def test_absorb_short_drops_utterances_with_empty_text(utterance_factory):
    """Финальный фильтр `[u for u in result if u.text]` обязан убрать реплику
    с пустым текстом (например, из одних лишь пробельных слов)."""
    empty = utterance_factory(0.0, 1.0, speaker=0, text="", words=[])

    result = _absorb_short([empty], min_duration=0.35)

    assert result == []


def test_absorb_short_empty_input():
    assert _absorb_short([], min_duration=0.35) == []


# ---------------------------------------------------------------------------
# assign(): сортировка turns не мутирует список вызывающего
# ---------------------------------------------------------------------------


def test_assign_does_not_mutate_callers_turns_list(turn_factory):
    """assign() обязана сортировать turns через sorted(), а НЕ через
    turns.sort(): первое создаёт новый список и не трогает список
    вызывающего, второе отсортировало бы его на месте. Это тест именно на
    ЭТУ, отдельную гарантию (неизменность объекта вызывающего) — он НЕ
    проверяет, что сортировка вообще нужна и что без неё код ломается: это
    отдельное свойство, для него ниже есть свой тест
    (test_assign_result_is_independent_of_turns_input_order), который
    красит перестановки без внутренней сортировки. Сам по себе этот тест
    прошёл бы и при полностью убранной сортировке (turns = pass) — оба
    варианта одинаково не мутируют исходный список, — поэтому он не
    заменяет тест на перестановки, а дополняет его."""
    original = [turn_factory(5.0, 6.0, speaker=1), turn_factory(0.0, 1.0, speaker=0)]
    before = list(original)
    seg = Segment(0.0, 10.0, "", [Word(0.5, 1.0, " a"), Word(5.5, 6.0, " b")])

    assign([seg], original, max_pause=10.0)

    assert original == before


def _adjacent_labelled_turns(n: int) -> list[Turn]:
    """n смежных (без зазора) реплик с уникальным speaker на каждую (0..n-1).
    Смежность важна: слово, охватывающее границу двух реплик, получает
    положительное перекрытие сразу с обеими — именно в этой неоднозначности
    порядок сканирования (а значит и то, отсортирован ли список) может
    повлиять на результат. С зазором между репликами (как в остальных тестах
    файла) слово однозначно попадает в одну реплику, и код всегда находит
    верный ответ через ветку «ближайший», независимо от сортировки — то есть
    та форма входа для этого конкретного регресса не годится."""
    return [Turn(float(2 * i), float(2 * i + 2), speaker=i) for i in range(n)]


def _words_straddling_each_boundary(turns: list[Turn]) -> Segment:
    """По одному слову на каждую внутреннюю границу между репликами, наполовину
    заходящему в обе смежные реплики (перекрытие 0.4 с с каждой стороны)."""
    words = [Word(turns[i].end - 0.4, turns[i].end + 0.4, f" w{i}") for i in range(len(turns) - 1)]
    return Segment(words[0].start, words[-1].end, "", words)


def test_assign_result_is_independent_of_turns_input_order():
    """Регресс на docs/plans/2026-08-25_публикация.md, раздел 3 («Закрыть
    скрытую предпосылку сортировки в align.assign»): assign() обязана
    сортировать turns по .start первой строкой — bisect_left в _speaker_at
    по несортированному списку молча даёт другой ответ. Один случайный
    порядок может случайно совпасть с правильным ответом (ровно это и было
    найдено ревью в прежней версии этого теста, использовавшей всего одну
    перестановку двух реплик с зазором между ними — см. историю файла и
    test_assign_does_not_mutate_callers_turns_list выше), поэтому здесь
    проверяется большая выборка перестановок сразу, а реплики специально
    смежные (см. _adjacent_labelled_turns), чтобы неоднозначность
    перекрытия была настоящей, а не маскировалась веткой «ближайший».

    Проверено вручную перед записью (сломать align.py:54 на `pass` в копии
    пакета вне репозитория): на этих 30 репликах и 200 перестановках их
    порядка без внутренней сортировки воспроизводится 200/200 расхождений
    с базовым (отсортированным) результатом — на текущем коде их обязано
    быть 0/200."""
    turns = _adjacent_labelled_turns(30)
    segment = _words_straddling_each_boundary(turns)
    baseline = [(u.speaker, u.text) for u in assign([segment], list(turns), max_pause=1000.0)]

    rng = random.Random(20260825)
    mismatches = 0
    trials = 200
    for _ in range(trials):
        shuffled = list(turns)
        rng.shuffle(shuffled)
        result = [(u.speaker, u.text) for u in assign([segment], shuffled, max_pause=1000.0)]
        if result != baseline:
            mismatches += 1

    assert mismatches == 0, f"{mismatches}/{trials} перестановок порядка turns дали другую разметку — turns не были отсортированы перед bisect"


# ---------------------------------------------------------------------------
# assign(): базовый happy-path с несколькими говорящими
# ---------------------------------------------------------------------------


def test_assign_basic_two_speakers_produces_two_utterances(turn_factory):
    turns = [turn_factory(0.0, 2.0, speaker=0), turn_factory(2.0, 4.0, speaker=1)]
    seg = Segment(0.0, 4.0, "", [Word(0.0, 1.0, " раз"), Word(1.0, 2.0, " два"), Word(2.5, 3.5, " три")])

    result = assign([seg], turns, max_pause=2.0)

    assert [(u.speaker, u.text) for u in result] == [(0, "раз два"), (1, "три")]


# ---------------------------------------------------------------------------
# speaker_stats(): пустой вход без деления на ноль
# ---------------------------------------------------------------------------


def test_speaker_stats_empty_input_no_zero_division():
    """total = sum(...) or 1.0 — страховка от деления на ноль на пустом входе."""
    assert speaker_stats([]) == {}


def test_speaker_stats_basic_share_and_counts(utterance_factory):
    utterances = [
        utterance_factory(0.0, 6.0, speaker=0, text="привет как дела", words=[]),
        utterance_factory(6.0, 8.0, speaker=1, text="да", words=[]),
    ]

    stats = speaker_stats(utterances)

    assert stats[0]["seconds"] == 6.0
    assert stats[0]["turns"] == 1
    assert stats[0]["words"] == 3
    assert stats[0]["share"] == 0.75
    assert stats[1]["share"] == 0.25


def test_utterance_dataclass_words_default_is_independent_list():
    """field(default_factory=list) обязан давать НОВЫЙ список каждому
    экземпляру — иначе разные Utterance делили бы один и тот же список слов."""
    a = Utterance(0.0, 1.0, 0, "a")
    b = Utterance(0.0, 1.0, 0, "b")
    a.words.append(Word(0.0, 1.0, " x"))

    assert b.words == []
