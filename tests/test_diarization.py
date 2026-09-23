"""Регресс-тесты на падения diarization.py, исправленные перед публикацией
(см. .docs/plans/2026-08-25_публикация.md, раздел 2), плюс тесты на
_tile_counts — единое (словарь по id плитки) представление «сколько человек
за плиткой», появившееся уже после первого захода на тесты.

Отдельная группа тестов — на семантику нумерации tile_speakers. Числа идут по
плиткам, НАЙДЕННЫМ в кадре (find_tiles, слева направо), потому что метка
плитки — это индекс в списке find_tiles. Требование равенства длин здесь было
блокером публикации: плитка, найденная в кадре, но с не дожившей до разметки
подсветкой, делала --tile-speakers неудовлетворимым ни при какой длине списка.
Полный воспроизведённый тупик — в test_cli.py.

Не полное покрытие модуля. Полная таблица (cluster, _viterbi, _centroids,
_gaps_between, _merge_adjacent, _covered, ...) — в разделе «хорошо бы»
плана, в этот заход не пишется.

Все тесты здесь — чистая арифметика на numpy-массивах или на списках Turn:
ни ONNX, ни sherpa_onnx, ни сеть не задействуются. segment/embed/cluster
там, где нужно исполнить код ПОСЛЕ них (см. тест на использование per-tile
числа говорящих), замоканы, а не вызваны по-настоящему.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from stenogram import diarization as diarization_module
from stenogram.diarization import Turn, _allocate, _tile_counts, diarize_with_video, estimate_speakers

# ---------------------------------------------------------------------------
# estimate_speakers: пустой np.diff при len(eigenvalues) < 2
# ---------------------------------------------------------------------------


def test_estimate_speakers_single_embedding_returns_one_speaker():
    """Один-единственный надёжный отрезок речи (например, короткий монолог
    или диктовка без --speakers): матрица эмбеддингов 1xD даёт лапласиан
    1x1 и, соответственно, ровно одно собственное значение. np.diff на
    массиве длины 1 даёт ПУСТОЙ массив, и np.argmax на нём раньше падал с
    `ValueError: attempt to get argmax of an empty sequence`. Теперь —
    безопасный ответ «один говорящий»."""
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    assert estimate_speakers(embeddings) == 1


def test_estimate_speakers_max_speakers_zero_returns_one_speaker():
    """--max-speakers 0 роняет ЛЮБОЙ вход независимо от числа реальных
    говорящих: eigenvalues[: 0 + 1] обрезает спектр до одного значения,
    и снова пустой np.diff. Это особенно коварно, потому что --threads 0
    в том же CLI означает «авто» — пользователь по аналогии может решить,
    что --max-speakers 0 тоже безопасное значение."""
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(5, 8)).astype(np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)

    assert estimate_speakers(rows, max_speakers=0) == 1


def test_estimate_speakers_identical_single_cluster_returns_one():
    """Механический (не качественный) тривиальный случай: все эмбеддинги
    ПОБИТОВО одинаковы — идеальный, безошибочный вход для эвристики разрыва
    спектра лапласиана. Это не тест качества оценки на реальном материале
    (docs/алгоритмы.md прямо называет автооценку числа участников самым
    хрупким местом пайплайна — на шумных данных даже 1 говорящий с небольшим
    гауссовым шумом на этих же векторах даёт 2, что и ожидаемо для эвристики
    на грани применимости и не является тестируемым инвариантом); здесь же
    единственно возможный ответ гарантирован построением входа, а не
    подобран под наблюдаемое поведение — проверено эмпирически."""
    embeddings = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), (6, 1))

    assert estimate_speakers(embeddings, max_speakers=12) == 1


def test_estimate_speakers_two_orthogonal_clusters_returns_two():
    """Тот же принцип: два строго ортогональных (нулевое сходство между
    группами, единичное — внутри группы) кластера эмбеддингов — эвристика
    обязана увидеть ровно один разрыв в спектре."""
    dim = 8
    e1, e2 = np.eye(dim)[0], np.eye(dim)[1]
    embeddings = np.vstack([np.tile(e1, (4, 1)), np.tile(e2, (4, 1))]).astype(np.float32)

    assert estimate_speakers(embeddings, max_speakers=12) == 2


def test_estimate_speakers_is_invariant_to_input_row_order():
    """Механическое свойство метода (не оценка качества): спектр лапласиана
    строится из симметричной матрицы сходства и не зависит от порядка строк
    входной матрицы эмбеддингов — перестановка строк переставляет affinity
    симметрично (P·A·Pᵀ) и не меняет её собственные значения. Если бы в
    вычисление закралась зависимость от позиции вместо содержимого (например,
    случайно перепутанный индекс), она проявилась бы здесь как расхождение
    с базовым результатом на переставленном входе."""
    rng = np.random.default_rng(7)
    dim = 8
    e1, e2, e3 = np.eye(dim)[0], np.eye(dim)[1], np.eye(dim)[2]
    rows = []
    for center, n in [(e1, 4), (e2, 4), (e3, 4)]:
        v = np.tile(center, (n, 1)).astype(np.float64) + rng.normal(scale=0.02, size=(n, dim))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rows.append(v)
    embeddings = np.vstack(rows).astype(np.float32)
    baseline = estimate_speakers(embeddings, max_speakers=12)

    permute = np.random.default_rng(123)
    for _ in range(20):
        idx = permute.permutation(len(embeddings))
        assert estimate_speakers(embeddings[idx], max_speakers=12) == baseline


# ---------------------------------------------------------------------------
# diarize_with_video: чисел должно хватать на индекс плитки, а не на счёт
# ---------------------------------------------------------------------------
#
# Числа tile_speakers нумеруются по плиткам, НАЙДЕННЫМ в кадре (find_tiles,
# слева направо), потому что метка плитки — это индекс в том самом списке.
# Отсюда проверка «max(tiles) < len(tile_speakers)» вместо равенства длин:
# требовать равенства нельзя, иначе плитка, найденная в кадре, но с не
# пойманной подсветкой, делает флаг неудовлетворимым (см. тест на тупик в
# test_cli.py).


def test_diarize_with_video_tile_speakers_too_short_for_tile_index_raises_value_error():
    """Раньше tile_speakers короче числа плиток приводил к IndexError на
    `counts[tile]` внутри цикла по плиткам. Теперь — понятный ValueError
    сразу на входе, до вызова сегментатора: числу негде взяться для плитки
    с индексом 1, если в списке всего одно число."""
    video_turns = [Turn(0.0, 1.0, speaker=0), Turn(1.0, 2.0, speaker=1)]  # плитки 0 и 1

    with pytest.raises(ValueError, match="tile_speakers"):
        diarize_with_video(np.zeros(1600, dtype=np.float32), video_turns, num_speakers=3, tile_speakers=[3], progress=False)


def test_diarize_with_video_tile_speakers_longer_than_surviving_tiles_is_fine(monkeypatch):
    """Регресс на блокер публикации: лишние числа — это НОРМА, а не ошибка.

    Раньше здесь требовалось равенство длин, и `--tile-speakers` с числами по
    всем найденным плиткам падал, если хоть одна из них не дожила до
    video_turns. Теперь число за не дожившей плиткой просто не используется."""
    video_turns = [Turn(0.0, 1.0, speaker=0)]  # подсветка поймана только на плитке 0
    monkeypatch.setattr(diarization_module, "segment", lambda *a, **k: [])

    turns, counts = diarize_with_video(np.zeros(1600, dtype=np.float32), video_turns, num_speakers=3, tile_speakers=[2, 3, 4], progress=False)

    assert turns == []
    assert counts == {0: 2}  # числа 3 и 4 — за плитками 2 и 3, их подсветка не дожила


def test_diarize_with_video_tile_speakers_error_message_names_count_and_tile_index():
    """Сообщение обязано называть обе величины: сколько чисел передано и
    какая плитка в них не помещается. Оно адресовано тому, кто зовёт API
    напрямую, минуя CLI, — у CLI своя проверка, по числу найденных плиток."""
    video_turns = [Turn(0.0, 1.0, speaker=0), Turn(1.0, 2.0, speaker=1)]

    with pytest.raises(ValueError, match=r"чисел передано — 1.*плитке № 2.*индекс 1"):
        diarize_with_video(np.zeros(1600, dtype=np.float32), video_turns, num_speakers=3, tile_speakers=[3], progress=False)


def test_diarize_with_video_tile_speakers_value_below_one_raises_value_error():
    """Значение --tile-speakers < 1 (0 или отрицательное) отвергается той же
    проверкой, что и несовпадение длины, и тоже до вызова сегментатора.
    CLI отсекает такие значения раньше, на этапе parse_tile_speakers (см.
    test_cli.py), но diarize_with_video вызывается и напрямую (не только из
    cli.main), поэтому обязана валидировать сама, не полагаясь на вызывающий
    код."""
    video_turns = [Turn(0.0, 1.0, speaker=0), Turn(1.0, 2.0, speaker=1)]

    with pytest.raises(ValueError, match="не может быть меньше одного человека"):
        diarize_with_video(np.zeros(1600, dtype=np.float32), video_turns, tile_speakers=[1, 0], progress=False)


def test_diarize_with_video_number_for_every_tile_does_not_raise_before_segment(monkeypatch):
    """Когда число нашлось для каждой плитки, проверка обязана промолчать и
    передать управление дальше (к сегментатору) — здесь сегментатор замокан,
    чтобы тест оставался офлайновым и быстрым, а не проверял sherpa_onnx.

    Регресс: раньше второй элемент результата был списком (`counts == [1,
    2]`) — этот же ассерт с списком закреплял как ожидаемое поведение ровно
    ту рассинхронизацию «список против словаря», из которой рос IndexError
    на несмежных id плиток (см. _tile_counts и тесты на tiles=[0, 2] ниже).
    diarize_with_video теперь всегда возвращает словарь по id плитки."""
    video_turns = [Turn(0.0, 1.0, speaker=0), Turn(1.0, 2.0, speaker=1)]
    monkeypatch.setattr(diarization_module, "segment", lambda *a, **k: [])

    turns, counts = diarize_with_video(np.zeros(1600, dtype=np.float32), video_turns, num_speakers=3, tile_speakers=[1, 2], progress=False)

    assert turns == []
    assert counts == {0: 1, 1: 2}


def test_diarize_with_video_uses_per_tile_speaker_count_to_split_clusters(monkeypatch):
    """Регресс: замена segment() на [] (как в тесте выше) делает `inside`
    пустым для каждой плитки, поэтому ветка кластеризации внутри плитки
    (k > 1 и len(inside) >= 4) не выполняется НИ РАЗУ — тест выше проверяет
    только то, что проверка длины не мешает дойти до сегментатора, а не то,
    что per-tile-число говорящих действительно используется. Здесь segment,
    embed и cluster замоканы так, чтобы это исполнилось по-настоящему: у
    плитки 0 один говорящий (k=1, проходит без кластеризации), у плитки 1 —
    два (k=2, четыре «реплики внутри» — заведомо больше порога len(inside)
    >= 4, поэтому код обязан дойти до embed()+cluster())."""
    video_turns = [Turn(0.0, 1.0, speaker=0), Turn(1.0, 2.0, speaker=1)]
    tile0_turn = Turn(0.1, 0.9, speaker=-1)
    tile1_turns = [Turn(1.05, 1.25, speaker=-1), Turn(1.3, 1.5, speaker=-1), Turn(1.55, 1.75, speaker=-1), Turn(1.8, 1.95, speaker=-1)]

    monkeypatch.setattr(diarization_module, "segment", lambda *a, **k: [tile0_turn, *tile1_turns])
    monkeypatch.setattr(diarization_module, "embed", lambda audio, turns, num_threads=4: (np.zeros((len(turns), 2), dtype=np.float32), list(range(len(turns)))))
    monkeypatch.setattr(diarization_module, "cluster", lambda embeddings, weights, k, gaps=None: np.array([0, 0, 1, 1]))

    turns, counts = diarize_with_video(np.zeros(1600, dtype=np.float32), video_turns, tile_speakers=[1, 2], progress=False)

    assert counts == {0: 1, 1: 2}
    # плитка 0 (k=1): проходит без кластеризации, offset=0 — один говорящий.
    assert {t.speaker for t in turns if t.start < 1.0} == {0}
    # плитка 1 (k=2): offset=1 (после плитки 0), метки кластера {0,1}
    # сдвинуты на offset -> {1,2} — то есть k реально использован, а не
    # проигнорирован.
    assert {t.speaker for t in turns if t.start >= 1.0} == {1, 2}


# ---------------------------------------------------------------------------
# _tile_counts / diarize_with_video: несмежные id плиток (tiles = [0, 2])
# ---------------------------------------------------------------------------
#
# Плитка, найденная в кадре, но ни разу не подсвеченная уверенно, выпадает
# из video_turns — id оставшихся плиток становятся несмежными. Числа при
# этом нумеруются по НАЙДЕННЫМ плиткам, то есть берутся по индексу плитки:
# tile_speakers[tile], а не tile_speakers[позиция среди доживших].


def test_tile_counts_takes_number_by_tile_index_not_by_position_among_survivors():
    """tiles=[0, 2] (подсветка плитки 2 не дожила). Числа [3, 9, 1] заданы по
    трём НАЙДЕННЫМ плиткам, поэтому плитке 2 достаётся третье число (1), а не
    второе (9): второе принадлежит плитке, которая до разметки не дошла.

    Это и есть смысл нумерации «по найденным плиткам»: пользователь считает
    плитки в кадре слева направо и не обязан знать, чья подсветка выжила."""
    result = _tile_counts([3, 9, 1], tiles=[0, 2], speech={0: 10.0, 2: 5.0}, num_speakers=None)

    assert result == {0: 3, 2: 1}
    assert 1 not in result  # плитки 1 нет в разметке — и числа для неё в словаре нет


def test_tile_counts_message_names_count_and_offending_tile_index():
    """Сообщение объясняет ровно то, чего не хватило: сколько чисел передано
    и какой индекс плитки в них не помещается. Номер плитки — id+1, как в
    выводе шага [4/4] («плитка N: ... мин речи»); на шаге [2/4] печатается
    только количество плиток, поэтому ссылаться на него нельзя."""
    with pytest.raises(ValueError, match=r"чисел передано — 1.*плитке № 3.*индекс 2.*не меньше 3"):
        _tile_counts([5], tiles=[0, 2], speech={0: 1.0, 2: 1.0}, num_speakers=None)


def test_tile_counts_negative_tile_label_raises_instead_of_indexing_from_the_end():
    """Обратная сторона индексации по метке плитки: отрицательная метка молча
    взяла бы число с конца списка (tile_speakers[-1]) и разметила бы плитку
    чужим числом. Встроенные источники таких меток не дают — video.to_turns
    отбрасывает -1, — но diarize_with_video принимает разметку и извне."""
    with pytest.raises(ValueError, match="не может быть отрицательной"):
        _tile_counts([3, 1], tiles=[-1, 0], speech={-1: 1.0, 0: 1.0}, num_speakers=None)


def test_tile_counts_empty_tiles_with_tile_speakers_returns_empty_dict():
    """Подсветка не поймана ни на одной плитке, а числа переданы: max() по
    пустому списку — ValueError, поэтому проверка обязана начинаться с
    `if tiles`. Разметка по плиткам просто не состоится, ошибки нет."""
    assert _tile_counts([2, 3], tiles=[], speech={}, num_speakers=None) == {}


def test_diarize_with_video_handles_non_contiguous_tile_ids_end_to_end(monkeypatch):
    """Полный путь: video_turns содержит только id 0 и 2 (плитка 1 не
    поймана), чисел передано три — по числу найденных в кадре плиток.
    Итоговая нумерация говорящих в результате обязана быть НЕПРЕРЫВНОЙ с
    нуля (0, 1), а не «дырявой» вслед за id плиток (0, 2) — потребители
    align.assign ожидают плотную нумерацию участников."""
    video_turns = [Turn(0.0, 1.0, speaker=0), Turn(1.0, 2.0, speaker=2)]
    monkeypatch.setattr(diarization_module, "segment", lambda *a, **k: [Turn(0.1, 0.9, speaker=-1), Turn(1.1, 1.9, speaker=-1)])

    turns, counts = diarize_with_video(np.zeros(1600, dtype=np.float32), video_turns, tile_speakers=[1, 1, 1], progress=False)

    assert counts == {0: 1, 2: 1}
    assert {t.speaker for t in turns} == {0, 1}


def test_diarize_with_video_splits_tile_by_number_taken_from_its_own_index(monkeypatch):
    """Число говорящих действительно берётся по индексу плитки, а не по
    позиции среди доживших, — и это видно по результату, а не только по
    словарю counts.

    tiles=[0, 2], числа [1, 9, 2]: плитка 2 обязана разделиться на ДВОИХ
    (третье число), а не на девятерых (второе). Проверяется через offset
    итоговых меток: 1 человек за плиткой 0 → метки плитки 2 начинаются с 1
    и дают {1, 2}."""
    video_turns = [Turn(0.0, 1.0, speaker=0), Turn(1.0, 2.0, speaker=2)]
    inside_tile2 = [Turn(1.05, 1.25, speaker=-1), Turn(1.3, 1.5, speaker=-1), Turn(1.55, 1.75, speaker=-1), Turn(1.8, 1.95, speaker=-1)]
    seen_k: list[int] = []

    def fake_cluster(embeddings, weights, k, gaps=None):
        seen_k.append(k)
        return np.array([0, 0, 1, 1])

    monkeypatch.setattr(diarization_module, "segment", lambda *a, **k: [Turn(0.1, 0.9, speaker=-1), *inside_tile2])
    monkeypatch.setattr(diarization_module, "embed", lambda audio, turns, num_threads=4: (np.zeros((len(turns), 2), dtype=np.float32), list(range(len(turns)))))
    monkeypatch.setattr(diarization_module, "cluster", fake_cluster)

    turns, counts = diarize_with_video(np.zeros(1600, dtype=np.float32), video_turns, tile_speakers=[1, 9, 2], progress=False)

    assert counts == {0: 1, 2: 2}
    assert seen_k == [2]  # кластеризация запрошена на двоих, а не на девятерых
    assert {t.speaker for t in turns if t.start >= 1.0} == {1, 2}


# ---------------------------------------------------------------------------
# _allocate: не падает на пустом tiles
# ---------------------------------------------------------------------------


def test_allocate_empty_tiles_returns_empty_dict_without_crashing():
    """Сценарий: find_tiles нашёл плитки физически, но подсветка ни разу не
    поймана уверенно (active_speaker/to_turns не дали ни одного Turn),
    поэтому список плиток на входе в _allocate пуст, а --speakers задан
    явно. Раньше это падало на `max(tiles, key=...)` —
    `ValueError: max() iterable argument is empty`, потому что проверка
    `not tiles` отсутствовала в условии раннего выхода."""
    result = _allocate({}, [], num_speakers=5)

    assert result == {}


def test_allocate_normal_input_gives_extra_speakers_to_busiest_tile():
    speech = {0: 10.0, 1: 2.0}
    result = _allocate(speech, [0, 1], num_speakers=5)

    assert result == {0: 4, 1: 1}


def test_allocate_num_speakers_none_gives_one_per_tile():
    result = _allocate({0: 10.0, 1: 2.0}, [0, 1], num_speakers=None)

    assert result == {0: 1, 1: 1}


def test_embed_without_speech_raises_no_speech_as_runtime_error(monkeypatch):
    """Тишина или музыка: ни одной реплики длиннее порога отпечатка. Сервис отличает
    это по классу (код no_speech), а для CLI и старых вызывающих это тот же RuntimeError
    с тем же текстом."""
    fake = types.SimpleNamespace(SpeakerEmbeddingExtractor=lambda config: object(), SpeakerEmbeddingExtractorConfig=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)
    monkeypatch.setattr(diarization_module, "ensure_all", lambda: ("seg.onnx", "emb.onnx"))

    with pytest.raises(diarization_module.NoSpeech) as err:
        diarization_module.embed(np.zeros(16000, dtype=np.float32), [Turn(0.0, 0.1, 0), Turn(0.5, 0.7, 0)])

    assert isinstance(err.value, RuntimeError)
    assert str(err.value) == "В записи не найдено речи достаточной длительности"
