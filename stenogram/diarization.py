"""Диаризация: кто и когда говорит.

Пайплайн:
  1. pyannote-segmentation-3.0 (ONNX) размечает границы реплик;
  2. CAM++ считает голосовой эмбеддинг каждой реплики;
  3. кластеризация эмбеддингов объединяет реплики одного человека.

Число говорящих задаётся явно (--speakers) либо оценивается по eigengap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE
from .models import ensure_all

# Внутренний порог сегментатора: намеренно низкий, чтобы реплики дробились мелко —
# итоговые кластеры всё равно строятся заново на шаге 3.
_SEG_THRESHOLD = 0.5
_MIN_EMBED_SEC = 0.4


@dataclass
class Turn:
    start: float
    end: float
    speaker: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def segment(audio: np.ndarray, num_threads: int = 4, progress: bool = True) -> list[Turn]:
    """Границы реплик. Метки говорящих на этом шаге черновые и позже заменяются."""
    import sherpa_onnx

    seg_model, emb_model = ensure_all()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(seg_model)
            ),
            num_threads=num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_model), num_threads=num_threads
        ),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=_SEG_THRESHOLD),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    last = [-10]

    def report(done: int, total: int, _=None) -> int:
        pct = int(done / total * 100)
        if progress and pct >= last[0] + 10:
            last[0] = pct
            print(f"  сегментация {pct}%", flush=True)
        return 0

    result = sd.process(audio, callback=report if progress else None).sort_by_start_time()
    return [Turn(r.start, r.end, r.speaker) for r in result]


def embed(audio: np.ndarray, turns: list[Turn], num_threads: int = 4) -> tuple[np.ndarray, list[int]]:
    """Эмбеддинг на реплику. Возвращает (матрица, индексы озвученных реплик)."""
    import sherpa_onnx

    _, emb_model = ensure_all()
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_model), num_threads=num_threads)
    )

    vectors, kept = [], []
    for i, turn in enumerate(turns):
        if turn.duration < _MIN_EMBED_SEC:
            continue
        chunk = audio[int(turn.start * SAMPLE_RATE): int(turn.end * SAMPLE_RATE)]
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, chunk)
        stream.input_finished()
        vectors.append(np.array(extractor.compute(stream), dtype=np.float32))
        kept.append(i)

    if not vectors:
        raise RuntimeError("В записи не найдено речи достаточной длительности")

    matrix = np.stack(vectors)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    return matrix, kept


def _refine_affinity(similarity: np.ndarray, prune: float = 0.90) -> np.ndarray:
    """Классическое сглаживание матрицы сходства перед спектральной кластеризацией."""
    affinity = (similarity + 1.0) / 2.0
    np.fill_diagonal(affinity, 0.0)
    np.fill_diagonal(affinity, affinity.max(axis=1))
    threshold = np.quantile(affinity, prune, axis=1, keepdims=True)
    affinity = np.where(affinity < threshold, affinity * 0.01, affinity)
    affinity = np.maximum(affinity, affinity.T)
    affinity = affinity @ affinity
    affinity /= affinity.max(axis=1, keepdims=True) + 1e-9
    return np.maximum(affinity, affinity.T)


def estimate_speakers(embeddings: np.ndarray, max_speakers: int = 12) -> int:
    """Оценка числа говорящих по наибольшему разрыву в спектре лапласиана."""
    affinity = _refine_affinity(embeddings @ embeddings.T)
    degree = affinity.sum(axis=1)
    inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-9))
    laplacian = np.eye(len(affinity)) - inv_sqrt @ affinity @ inv_sqrt
    eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))[: max_speakers + 1]
    gaps = np.diff(eigenvalues)
    return int(np.argmax(gaps) + 1)


def _centroids(embeddings: np.ndarray, weights: np.ndarray, labels: np.ndarray,
               num_speakers: int) -> np.ndarray:
    rows = []
    for c in range(num_speakers):
        sel = labels == c
        rows.append(np.average(embeddings[sel], axis=0, weights=weights[sel])
                    if sel.any() else embeddings[c % len(embeddings)])
    centroids = np.stack(rows)
    return centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)


def _viterbi(embeddings: np.ndarray, weights: np.ndarray, gaps: np.ndarray,
             centroids: np.ndarray, stay: float, scale: float) -> np.ndarray:
    """Последовательность говорящих со штрафом за смену (упрощённый VBx).

    Без этого монолог рассыпается: соседние реплики одного человека получают разные метки,
    потому что голосовой отпечаток короткой реплики шумит.
    """
    k = len(centroids)
    emission = scale * (embeddings @ centroids.T) * np.sqrt(np.clip(weights, 0.3, 6.0))[:, None]
    emission -= emission.max(axis=1, keepdims=True)

    # Чем длиннее пауза перед репликой, тем вероятнее, что говорящий сменился.
    p_stay = np.clip(stay - 0.3 * np.tanh(np.clip(gaps, 0, None) / 1.0), 0.30, 0.99)
    log_stay, log_switch = np.log(p_stay), np.log((1 - p_stay) / max(k - 1, 1))

    delta = emission[0].copy()
    back = np.zeros((len(embeddings), k), dtype=np.int16)
    for t in range(1, len(embeddings)):
        same = delta + log_stay[t]
        other = delta.max() + log_switch[t]
        back[t] = np.where(same > other, np.arange(k), int(delta.argmax()))
        delta = np.maximum(same, other) + emission[t]
        delta -= delta.max()

    path = np.zeros(len(embeddings), dtype=int)
    path[-1] = int(delta.argmax())
    for t in range(len(embeddings) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def cluster(embeddings: np.ndarray, weights: np.ndarray, num_speakers: int,
            gaps: np.ndarray | None = None, stay: float = 0.80, scale: float = 10.0,
            iterations: int = 6) -> np.ndarray:
    """Метки говорящих: кластеризация надёжных реплик + сглаживание по времени."""
    from sklearn.cluster import KMeans

    if num_speakers <= 1 or len(embeddings) <= num_speakers:
        return np.zeros(len(embeddings), dtype=int)

    # Короткие реплики не участвуют в поиске центроидов: их отпечаток ненадёжен.
    solid = weights >= 1.5
    if solid.sum() < num_speakers:
        solid = np.ones(len(weights), dtype=bool)

    km = KMeans(n_clusters=num_speakers, n_init=40, random_state=0).fit(
        embeddings[solid], sample_weight=weights[solid])
    centroids = km.cluster_centers_
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
    labels = np.asarray((embeddings @ centroids.T).argmax(axis=1), dtype=int)

    if gaps is None:
        return labels

    for _ in range(iterations):
        labels = _viterbi(embeddings, weights, gaps, centroids, stay, scale)
        centroids = _centroids(embeddings, weights, labels, num_speakers)
    return labels


def _gaps_between(turns: list[Turn], kept: list[int]) -> np.ndarray:
    """Паузы перед каждой репликой: длинная пауза — вероятная смена говорящего."""
    starts = np.array([turns[i].start for i in kept])
    ends = np.array([turns[i].end for i in kept])
    return np.concatenate([[10.0], starts[1:] - ends[:-1]])


def _fill_short(turns: list[Turn], kept: list[int], labels: np.ndarray) -> None:
    """Реплики без эмбеддинга (слишком короткие) отдаём ближайшей размеченной соседке."""
    for turn, label in zip((turns[i] for i in kept), labels):
        turn.speaker = int(label)
    kept_set = set(kept)
    for i, turn in enumerate(turns):
        if i in kept_set:
            continue
        near = [j for j in (i - 1, i + 1) if 0 <= j < len(turns) and j in kept_set]
        turn.speaker = turns[min(near, key=lambda j: abs(turns[j].start - turn.start))].speaker \
            if near else 0


def diarize(audio: np.ndarray, num_speakers: int | None = None, max_speakers: int = 12,
            num_threads: int = 4, progress: bool = True) -> list[Turn]:
    """Только по звуку: сегментация → эмбеддинги → кластеризация → сглаживание."""
    turns = segment(audio, num_threads=num_threads, progress=progress)
    if progress:
        print(f"      реплик размечено: {len(turns)}", flush=True)

    embeddings, kept = embed(audio, turns, num_threads=num_threads)
    weights = np.array([turns[i].duration for i in kept], dtype=np.float64)

    if num_speakers is None:
        num_speakers = estimate_speakers(embeddings, max_speakers=max_speakers)
        if progress:
            print(f"      оценка числа участников: {num_speakers}", flush=True)

    labels = cluster(embeddings, weights, num_speakers, gaps=_gaps_between(turns, kept))
    _fill_short(turns, kept, labels)
    return _merge_adjacent(turns)


def diarize_with_video(audio: np.ndarray, video_turns: list[Turn], num_speakers: int | None = None,
                       tile_speakers: list[int] | None = None, num_threads: int = 4,
                       progress: bool = True) -> tuple[list[Turn], list[int]]:
    """Гибрид: видео говорит, какая плитка активна, звук делит людей внутри плитки.

    Возвращает (реплики, сколько участников пришлось на каждую плитку).
    """
    tiles = sorted({t.speaker for t in video_turns})
    speech = {tile: sum(t.duration for t in video_turns if t.speaker == tile) for tile in tiles}
    if progress:
        for tile in tiles:
            print(f"      плитка {tile+1}: {speech[tile]/60:.1f} мин речи", flush=True)

    turns = segment(audio, num_threads=num_threads, progress=progress)
    counts = tile_speakers or _allocate(speech, tiles, num_speakers)

    result: list[Turn] = []
    offset = 0
    for tile in tiles:
        intervals = [(t.start, t.end) for t in video_turns if t.speaker == tile]
        inside = [t for t in turns if _covered(t, intervals) > 0.7 * t.duration]
        if not inside:
            offset += counts[tile]
            continue

        k = max(1, counts[tile])
        if k == 1 or len(inside) < 4:
            for turn in inside:
                result.append(Turn(turn.start, turn.end, offset))
        else:
            embeddings, kept = embed(audio, inside, num_threads=num_threads)
            weights = np.array([inside[i].duration for i in kept], dtype=np.float64)
            labels = cluster(embeddings, weights, k, gaps=_gaps_between(inside, kept))
            _fill_short(inside, kept, labels)
            for turn in inside:
                result.append(Turn(turn.start, turn.end, offset + turn.speaker))
            if progress:
                print(f"      плитка {tile+1}: разделена на {len(set(t.speaker for t in inside))}",
                      flush=True)
        offset += k

    result.sort(key=lambda t: t.start)
    return _merge_adjacent(result), counts


def _covered(turn: Turn, intervals: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(turn.end, b) - max(turn.start, a)) for a, b in intervals)


def _allocate(speech: dict[int, float], tiles: list[int], num_speakers: int | None) -> dict[int, int]:
    """Сколько человек за какой плиткой. Лишних сажаем к самой говорливой плитке.

    Так устроены записи ВКС: несколько человек в одном зале делят общую камеру,
    остальные подключены по одному.
    """
    counts = {tile: 1 for tile in tiles}
    if num_speakers is None or num_speakers <= len(tiles):
        return counts
    busiest = max(tiles, key=lambda t: speech[t])
    counts[busiest] += num_speakers - len(tiles)
    return counts


def _merge_adjacent(turns: list[Turn], gap: float = 0.6) -> list[Turn]:
    """Склеивает соседние реплики одного говорящего, разделённые короткой паузой."""
    merged: list[Turn] = []
    for turn in turns:
        if merged and merged[-1].speaker == turn.speaker and turn.start - merged[-1].end <= gap:
            merged[-1].end = max(merged[-1].end, turn.end)
        else:
            merged.append(Turn(turn.start, turn.end, turn.speaker))
    return merged
