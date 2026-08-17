"""Сшивка распознанного текста с разметкой говорящих."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

from .asr import Segment, Word
from .diarization import Turn


@dataclass
class Utterance:
    start: float
    end: float
    speaker: int
    text: str
    words: list[Word] = field(default_factory=list)


def _speaker_at(turns: list[Turn], starts: list[float], word: Word) -> int:
    """Говорящий с наибольшим перекрытием по времени; при отсутствии — ближайший по времени."""
    best_speaker, best_overlap = -1, 0.0
    i = max(0, bisect_left(starts, word.start) - 1)
    while i < len(turns) and turns[i].start < word.end:
        overlap = min(word.end, turns[i].end) - max(word.start, turns[i].start)
        if overlap > best_overlap:
            best_speaker, best_overlap = turns[i].speaker, overlap
        i += 1
    if best_speaker >= 0:
        return best_speaker

    middle = (word.start + word.end) / 2
    nearest = min(turns, key=lambda t: 0 if t.start <= middle <= t.end
                  else min(abs(t.start - middle), abs(t.end - middle)))
    return nearest.speaker


def assign(segments: list[Segment], turns: list[Turn], max_pause: float = 2.0,
           min_utterance: float = 0.35) -> list[Utterance]:
    """Размечает каждое слово говорящим и собирает слова в реплики.

    max_pause — пауза внутри реплики одного человека, после которой начинается новая.
    min_utterance — реплики короче отдаются соседям, чтобы убрать дребезг разметки.
    """
    if not turns:
        return [Utterance(s.start, s.end, 0, s.text, s.words) for s in segments]

    starts = [t.start for t in turns]
    tagged: list[tuple[int, Word]] = []
    for segment in segments:
        words = segment.words or [Word(segment.start, segment.end, segment.text)]
        for word in words:
            tagged.append((_speaker_at(turns, starts, word), word))

    if not tagged:
        return []

    tagged = _smooth(tagged)

    utterances: list[Utterance] = []
    for speaker, word in tagged:
        last = utterances[-1] if utterances else None
        if last and last.speaker == speaker and word.start - last.end <= max_pause:
            last.words.append(word)
            last.end = word.end
        else:
            utterances.append(Utterance(word.start, word.end, speaker, "", [word]))

    for utterance in utterances:
        utterance.text = "".join(w.text for w in utterance.words).strip()

    return _absorb_short(utterances, min_utterance)


def _smooth(tagged: list[tuple[int, Word]], window: int = 2) -> list[tuple[int, Word]]:
    """Убирает одиночные выбросы: слово, окружённое другим говорящим, отдаётся ему."""
    labels = [s for s, _ in tagged]
    fixed = labels[:]
    for i in range(len(labels)):
        left = labels[max(0, i - window): i]
        right = labels[i + 1: i + 1 + window]
        if left and right and left[-1] == right[0] and left[-1] != labels[i]:
            if all(x == left[-1] for x in left) and all(x == right[0] for x in right):
                fixed[i] = left[-1]
    return [(fixed[i], word) for i, (_, word) in enumerate(tagged)]


def _absorb_short(utterances: list[Utterance], min_duration: float) -> list[Utterance]:
    """Склеивает микрореплики с соседней, если та принадлежит одному говорящему."""
    result: list[Utterance] = []
    for utterance in utterances:
        if (result and utterance.end - utterance.start < min_duration
                and len(utterance.text) <= 3 and result[-1].speaker != utterance.speaker):
            result[-1].words.extend(utterance.words)
            result[-1].end = utterance.end
            result[-1].text = "".join(w.text for w in result[-1].words).strip()
            continue
        if (result and result[-1].speaker == utterance.speaker
                and utterance.start - result[-1].end <= 2.0):
            result[-1].words.extend(utterance.words)
            result[-1].end = utterance.end
            result[-1].text = "".join(w.text for w in result[-1].words).strip()
            continue
        result.append(utterance)
    return [u for u in result if u.text]


def speaker_stats(utterances: list[Utterance]) -> dict[int, dict]:
    """Сводка по каждому говорящему: время, доля, число реплик, число слов."""
    stats: dict[int, dict] = {}
    total = sum(u.end - u.start for u in utterances) or 1.0
    for utterance in utterances:
        entry = stats.setdefault(utterance.speaker, {"seconds": 0.0, "turns": 0, "words": 0})
        entry["seconds"] += utterance.end - utterance.start
        entry["turns"] += 1
        entry["words"] += len(utterance.text.split())
    for entry in stats.values():
        entry["share"] = entry["seconds"] / total
    return dict(sorted(stats.items(), key=lambda kv: -kv[1]["seconds"]))
