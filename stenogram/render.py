"""Форматы выгрузки стенограммы."""
from __future__ import annotations

import json
from pathlib import Path

from .align import Utterance, speaker_stats


def hhmmss(seconds: float, millis: bool = False, comma: bool = False) -> str:
    hours, rest = divmod(max(seconds, 0.0), 3600)
    minutes, secs = divmod(rest, 60)
    if not millis:
        return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
    sep = "," if comma else "."
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}{sep}{int(secs % 1 * 1000):03d}"


def speaker_name(index: int, names: dict[int, str] | None) -> str:
    if names and index in names:
        return names[index]
    return f"Говорящий {index + 1}"


def to_text(utterances: list[Utterance], names: dict[int, str] | None = None,
            timestamps: bool = True) -> str:
    lines = []
    for utterance in utterances:
        stamp = f"[{hhmmss(utterance.start)}] " if timestamps else ""
        lines.append(f"{stamp}{speaker_name(utterance.speaker, names)}: {utterance.text}")
    return "\n\n".join(lines) + "\n"


def to_markdown(utterances: list[Utterance], names: dict[int, str] | None = None,
                title: str = "Стенограмма", meta: dict | None = None) -> str:
    out = [f"# {title}", ""]
    if meta:
        out += [f"- **{k}:** {v}" for k, v in meta.items()] + [""]

    stats = speaker_stats(utterances)
    out += ["## Участники", "",
            "| Участник | Время | Доля | Реплик | Слов |",
            "|---|---:|---:|---:|---:|"]
    for index, entry in stats.items():
        out.append(f"| {speaker_name(index, names)} | {entry['seconds']/60:.1f} мин "
                   f"| {entry['share']:.0%} | {entry['turns']} | {entry['words']} |")

    out += ["", "## Диалог", ""]
    for utterance in utterances:
        out.append(f"**{speaker_name(utterance.speaker, names)}** "
                   f"`{hhmmss(utterance.start)}`  \n{utterance.text}")
        out.append("")
    return "\n".join(out)


def to_srt(utterances: list[Utterance], names: dict[int, str] | None = None,
           max_chars: int = 90) -> str:
    blocks, index = [], 1
    for utterance in utterances:
        for start, end, text in _split(utterance, max_chars):
            blocks.append(f"{index}\n{hhmmss(start, True, True)} --> {hhmmss(end, True, True)}\n"
                          f"{speaker_name(utterance.speaker, names)}: {text}\n")
            index += 1
    return "\n".join(blocks)


def to_vtt(utterances: list[Utterance], names: dict[int, str] | None = None,
           max_chars: int = 90) -> str:
    blocks = ["WEBVTT", ""]
    for utterance in utterances:
        for start, end, text in _split(utterance, max_chars):
            blocks.append(f"{hhmmss(start, True)} --> {hhmmss(end, True)}")
            blocks.append(f"<v {speaker_name(utterance.speaker, names)}>{text}")
            blocks.append("")
    return "\n".join(blocks)


def _split(utterance: Utterance, max_chars: int):
    """Режет длинную реплику на субтитровые куски по границам слов."""
    if not utterance.words or len(utterance.text) <= max_chars:
        yield utterance.start, utterance.end, utterance.text
        return
    buffer, start = [], utterance.words[0].start
    for word in utterance.words:
        buffer.append(word)
        if len("".join(w.text for w in buffer)) >= max_chars:
            yield start, word.end, "".join(w.text for w in buffer).strip()
            buffer, start = [], word.end
    if buffer:
        yield start, buffer[-1].end, "".join(w.text for w in buffer).strip()


def to_json(utterances: list[Utterance], names: dict[int, str] | None = None,
            meta: dict | None = None) -> str:
    payload = {
        "meta": meta or {},
        "speakers": [
            {"id": index, "name": speaker_name(index, names), **entry}
            for index, entry in speaker_stats(utterances).items()
        ],
        "utterances": [
            {
                "start": round(u.start, 2), "end": round(u.end, 2),
                "speaker_id": u.speaker, "speaker": speaker_name(u.speaker, names),
                "text": u.text,
                "words": [{"start": round(w.start, 2), "end": round(w.end, 2),
                           "word": w.text.strip()} for w in u.words],
            }
            for u in utterances
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


WRITERS = {
    "txt": lambda u, n, m: to_text(u, n),
    "md": lambda u, n, m: to_markdown(u, n, meta=m),
    "srt": lambda u, n, m: to_srt(u, n),
    "vtt": lambda u, n, m: to_vtt(u, n),
    "json": lambda u, n, m: to_json(u, n, m),
}


def write(utterances: list[Utterance], stem: Path, formats: list[str],
          names: dict[int, str] | None = None, meta: dict | None = None) -> list[Path]:
    written = []
    for fmt in formats:
        if fmt not in WRITERS:
            raise ValueError(f"Неизвестный формат: {fmt}. Доступны: {', '.join(WRITERS)}")
        path = stem.with_suffix(f".{fmt}")
        path.write_text(WRITERS[fmt](utterances, names, meta), encoding="utf-8")
        written.append(path)
    return written
