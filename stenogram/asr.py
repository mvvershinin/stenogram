"""Распознавание речи через faster-whisper с потайминговкой по словам."""

from __future__ import annotations

import time
from collections.abc import Callable
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


def _report(progress: bool | Callable[[str, float | None], None], message: str, fraction: float | None = None) -> None:
    """Сообщение о ходе работы: в stdout, в функцию вызывающего или никуда."""
    if callable(progress):
        progress(message, fraction)
    elif progress:
        print(message, flush=True)


def load_model(
    model_name: str = "large-v3-turbo",
    device: str = "cpu",
    compute_type: str = "int8",
    num_threads: int = 8,
    local_files_only: bool = False,
    download_root: str | None = None,
):
    """Загружает модель whisper. Долгоживущий процесс держит одну и передаёт её в transcribe.

    `model_name` — имя модели или путь к её каталогу. С `local_files_only=True` отсутствие
    модели на диске — ошибка, а не молчаливый поход в сеть: иначе опечатка в пути на машине
    без интернета проявляется только на первой задаче."""
    from faster_whisper import WhisperModel  # noqa: PLC0415 -- тяжёлый импорт: должен остаться внутри функции

    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=num_threads,
        local_files_only=local_files_only,
        download_root=download_root,
    )


def transcribe(
    wav: Path,
    model_name: str = "large-v3-turbo",
    language: str | None = None,
    beam_size: int = 5,
    compute_type: str = "int8",
    device: str = "cpu",
    num_threads: int = 8,
    initial_prompt: str | None = None,
    progress: bool | Callable[[str, float | None], None] = True,
    *,
    model=None,
    local_files_only: bool = False,
    download_root: str | None = None,
) -> tuple[list[Segment], str]:
    """Возвращает (сегменты с потайминговкой по словам, код языка).

    `model` — уже загруженная `load_model` модель; с ней параметры загрузки (`model_name`,
    `device`, `compute_type`, `num_threads`, `local_files_only`, `download_root`) не
    используются. `progress`: True — печатать ход работы, False — молчать, функция —
    получать (сообщение, долю распознанного или None)."""
    if model is None:
        _report(progress, f"  загружаю модель {model_name} ({device}/{compute_type}) ...")
        model = load_model(model_name, device, compute_type, num_threads, local_files_only, download_root)

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
    _report(progress, f"  язык: {info.language} (уверенность {info.language_probability:.0%}), длительность {info.duration / 60:.1f} мин", 0.0)

    segments: list[Segment] = []
    started = time.time()
    for item in stream:
        segments.append(
            Segment(
                start=item.start,
                end=item.end,
                text=item.text.strip(),
                words=[Word(w.start, w.end, w.word) for w in (item.words or [])],
            )
        )
        if progress and len(segments) % 25 == 0:
            elapsed = time.time() - started
            speed = item.end / max(elapsed, 1e-9)
            _report(
                progress,
                f"  {item.end / 60:5.1f} / {info.duration / 60:.1f} мин (x{speed:.1f}, осталось ~{(info.duration - item.end) / speed / 60:.0f} мин)",
                min(item.end / info.duration, 1.0) if info.duration else None,
            )

    return segments, info.language
