"""Распознавание речи через faster-whisper с потайминговкой по словам."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def transcribe(wav: Path, model_name: str = "large-v3-turbo", language: str | None = None,
               beam_size: int = 5, compute_type: str = "int8", device: str = "cpu",
               num_threads: int = 8, initial_prompt: str | None = None,
               progress: bool = True) -> tuple[list[Segment], str]:
    """Возвращает (сегменты с потайминговкой по словам, код языка)."""
    from faster_whisper import WhisperModel

    if progress:
        print(f"  загружаю модель {model_name} ({device}/{compute_type}) ...", flush=True)
    model = WhisperModel(model_name, device=device, compute_type=compute_type,
                         cpu_threads=num_threads)

    stream, info = model.transcribe(
        str(wav),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        # На длинных записях контекст предыдущего сегмента провоцирует зацикливание.
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 200},
        initial_prompt=initial_prompt,
    )
    if progress:
        print(f"  язык: {info.language} (уверенность {info.language_probability:.0%}), "
              f"длительность {info.duration/60:.1f} мин", flush=True)

    segments: list[Segment] = []
    started = time.time()
    for item in stream:
        segments.append(Segment(
            start=item.start, end=item.end, text=item.text.strip(),
            words=[Word(w.start, w.end, w.word) for w in (item.words or [])],
        ))
        if progress and len(segments) % 25 == 0:
            elapsed = time.time() - started
            speed = item.end / max(elapsed, 1e-9)
            print(f"  {item.end/60:5.1f} / {info.duration/60:.1f} мин "
                  f"(x{speed:.1f}, осталось ~{(info.duration-item.end)/speed/60:.0f} мин)", flush=True)

    return segments, info.language
