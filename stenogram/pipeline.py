"""Весь пайплайн одной функцией: медиафайл → реплики с участниками.

Порядок этапов записан только здесь. CLI и сервис вызывают `run`, а не повторяют
последовательность у себя: половина умолчаний стоит за замерами, и копия порядка
от них дрейфует. Почему распознавание идёт до разметки участников —
.docs/decisions/0005-порядок-этапов.md.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import align, asr, audio, diarization, video

# Пауза, после которой сшивка начинает новую реплику того же участника.
MAX_PAUSE = 3.0


@dataclass
class Options:
    """Параметры прогона. Умолчания общие для CLI и сервиса; за числами стоят замеры."""

    model: str = "large-v3-turbo"
    language: str | None = None
    prompt: str | None = None
    beam_size: int = 5
    device: str = "cpu"
    compute_type: str = "int8"
    threads: int = 0  # 0 — все ядра, кроме двух
    denoise: bool = False
    diarization: bool = True
    video: bool = True
    speakers: int | None = None
    max_speakers: int = 12
    tile_speakers: list[int] | None = None
    highlight_lag: float = 1.2


@dataclass
class Intermediate:
    """Дорогие промежуточные результаты — всё, что не зависит от параметров разметки.

    С ними повторный прогон с другим числом участников обходится без распознавания
    (32 минуты из 35 на часовой записи на процессоре) и без повторного чтения видео."""

    segments: list[asr.Segment]
    language: str
    tiles: list[video.Tile] | None = None  # None — плитки не искали, [] — искали и не нашли
    labels: np.ndarray | None = None  # метка подсвеченной плитки на каждый шаг, -1 — ничья
    step: float | None = None
    video_error: str | None = None  # поиск упал: tiles == [] значит «не вышло», а не «подсветки нет»

    def to_dict(self) -> dict:
        """JSON-совместимый словарь для внешнего кэша. Числа приводятся к int/float явно:
        numpy-скаляры `json.dumps` не принимает."""
        return {
            "segments": [
                {
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": s.text,
                    "words": [{"start": float(w.start), "end": float(w.end), "text": w.text} for w in s.words],
                }
                for s in self.segments
            ],
            "language": self.language,
            "tiles": None if self.tiles is None else [[int(t.x), int(t.y), int(t.width), int(t.height)] for t in self.tiles],
            "labels": None if self.labels is None else [int(x) for x in self.labels],
            "step": None if self.step is None else float(self.step),
            "video_error": self.video_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Intermediate:
        """Обратно из `to_dict`. `None` и пустой список у плиток и меток различаются."""
        return cls(
            segments=[asr.Segment(s["start"], s["end"], s["text"], [asr.Word(w["start"], w["end"], w["text"]) for w in s["words"]]) for s in data["segments"]],
            language=data["language"],
            tiles=None if data["tiles"] is None else [video.Tile(*t) for t in data["tiles"]],
            labels=None if data["labels"] is None else np.asarray(data["labels"], dtype=int),
            step=data["step"],
            video_error=data["video_error"],
        )


@dataclass
class Result:
    utterances: list[align.Utterance]
    turns: list[diarization.Turn]
    duration: float
    intermediate: Intermediate
    tile_counts: dict[int, int] | None = None  # сколько людей пришлось на каждую плитку
    # Поиск плиток упал (в этом прогоне или в том, что оставил `reuse`), а разметка
    # по видео в этом прогоне была нужна — поэтому откатилась на голос.
    video_error: str | None = None
    video_used: bool = False  # разметка участников в этом прогоне шла по видео


@dataclass
class Event:
    """Сообщение о ходе работы. `message` — готовая строка, ровно как её печатает CLI."""

    stage: str  # audio | video | asr | highlight | diarization | align
    message: str
    fraction: float | None = None  # доля пройденного внутри этапа, где она известна
    error: BaseException | None = None  # из-за чего разметка откатилась на голос


class TileSpeakersMismatch(ValueError):
    """Чисел в `tile_speakers` не столько, сколько плиток найдено в кадре."""


def run(
    source: Path | None,
    wav: Path,
    options: Options | None = None,
    *,
    reuse: Intermediate | None = None,
    whisper_model=None,
    progress: Callable[[Event], None] | None = None,
) -> Result:
    """Прогоняет пайплайн и возвращает данные; файлы пишет вызывающий.

    `wav` — куда извлечь звук из `source`. С `reuse` звук берётся из `wav` как есть,
    а распознавание и разбор видео — из кэша; заново считаются только разметка
    участников и сшивка. Плитки, которых в кэше нет, ищутся в `source`.

    С какими параметрами распознавания получен `reuse`, функция не знает и не
    проверяет: это забота ключа кэша у вызывающего (модель, язык, подсказка, beam,
    compute_type, denoise).

    `whisper_model` — модель от `asr.load_model`: долгоживущий процесс держит одну.
    Без `progress` функция молчит. Исключения из `progress` всплывают наружу как есть.
    """
    opts = options or Options()
    _check(opts)
    threads = opts.threads or max(1, (os.cpu_count() or 4) - 2)

    # Исключения колбэка вызывающего, прошедшие через модули: их нельзя принять
    # за отказ видео (например, BrokenPipeError от `stenogram … | head`).
    callback_errors: list[BaseException] = []

    def say(stage: str, message: str, error: BaseException | None = None) -> None:
        if progress is not None:
            progress(Event(stage, message, error=error))

    def reporter(stage: str) -> bool | Callable[[str, float | None], None]:
        if progress is None:
            return False

        def report(message: str, fraction: float | None = None) -> None:
            try:
                progress(Event(stage, message, fraction))
            except BaseException as exc:
                callback_errors.append(exc)
                raise

        return report

    if reuse is None:
        if source is None:
            raise ValueError("Нужен исходный файл или кэш промежуточных результатов")
        say("audio", f"[1/4] Извлекаю аудио из {source.name} ...")
        audio.extract_wav(source, wav, denoise=opts.denoise)
    else:
        say("audio", "[1/4] Звук взят из кэша")
    duration = audio.probe_duration(wav)
    say("audio", f"      {duration / 60:.1f} мин звука")

    # Что известно о плитках (идёт в кэш целиком) и по чему размечаем в этом прогоне —
    # разные вещи: повтор с выключенным видео плитки из кэша не использует, но и не теряет.
    tiles = reuse.tiles if reuse is not None else None
    video_error = reuse.video_error if reuse is not None else None
    want_video = opts.diarization and opts.video
    if want_video and tiles is None:
        if source is None:
            raise ValueError("Для разметки по видео нужен исходный файл: в кэше нет плиток")
        tiles = []
        video_error = None  # новый поиск заменяет причину старого отказа из кэша
        if video.has_video(source):
            say("video", "[2/4] Ищу подсветку активного говорящего в видео ...")
            try:
                tiles = video.find_tiles(source, progress=reporter("video"))
            except Exception as exc:  # видео может быть нестандартным — не повод падать
                if any(exc is own for own in callback_errors):
                    raise
                video_error = str(exc) or type(exc).__name__  # у MemoryError() str пустой
                try:
                    say("video", f"      не получилось ({exc}); останусь на звуке", error=exc)
                finally:
                    # Кадры find_tiles держат массив насыщенности (до ~1 ГБ на 1080p), а
                    # событие с исключением может жить в чужом процессе сколь угодно долго.
                    import traceback  # noqa: PLC0415 -- лениво: иначе модуль грузится ради каждого `--help`

                    traceback.clear_frames(exc.__traceback__)
    video_used = want_video and bool(tiles)
    if not video_used:
        say("video", "[2/4] Подсветка не найдена — участники только по голосу")
        if opts.tile_speakers is not None:
            # Иначе параметр молча ничего не делает: без плиток делить некого,
            # и пользователь узнаёт об этом только по числу участников в сводке.
            say("video", "      --tile-speakers не пригодился: плиток нет, помочь может --speakers")

    # Число плиток известно только сейчас — зато всё ещё до распознавания,
    # то есть несовпадение длины не стоит пользователю самого долгого этапа.
    # Сверяется именно с найденными в кадре плитками — по ним и нумеруются числа
    # (diarization._tile_counts). Дальше по пайплайну часть плиток может отсеяться,
    # но список от этого короче не становится: там числа берутся по индексу плитки.
    if video_used and opts.tile_speakers is not None and len(opts.tile_speakers) != len(tiles):
        raise TileSpeakersMismatch(
            f"--tile-speakers: чисел передано — {len(opts.tile_speakers)}, а плиток в видео найдено — {len(tiles)}; нужно по одному числу "
            "на каждую найденную плитку, слева направо — включая те, что за весь разговор ни разу не подсвечены"
        )

    if reuse is not None:
        segments, language = reuse.segments, reuse.language
        say("asr", "[3/4] Распознавание взято из кэша")
    else:
        say("asr", "[3/4] Распознаю речь ...")
        segments, language = asr.transcribe(
            wav,
            model_name=opts.model,
            language=opts.language,
            beam_size=opts.beam_size,
            compute_type=opts.compute_type,
            device=opts.device,
            num_threads=threads,
            initial_prompt=opts.prompt,
            progress=reporter("asr"),
            model=whisper_model,
        )

    say("diarization", "[4/4] Определяю участников ...")
    turns: list[diarization.Turn] = []
    counts = None
    labels = reuse.labels if reuse is not None else None
    step = reuse.step if reuse is not None else None
    if opts.diarization:
        samples = audio.load_wav(wav)
        if video_used:
            if labels is None:
                if source is None:
                    raise ValueError("Для разметки по видео нужен исходный файл: в кэше нет меток подсветки")
                labels, step = video.active_speaker(source, tiles, progress=reporter("highlight"))
            video_turns = video.to_turns(labels, step, video.word_gaps(segments), lag=opts.highlight_lag)
            turns, counts = diarization.diarize_with_video(
                samples,
                video_turns,
                num_speakers=opts.speakers,
                tile_speakers=opts.tile_speakers,
                num_threads=max(2, threads // 2),
                progress=reporter("diarization"),
            )
        else:
            turns = diarization.diarize(
                samples,
                num_speakers=opts.speakers,
                max_speakers=opts.max_speakers,
                num_threads=max(2, threads // 2),
                progress=reporter("diarization"),
            )
        say("diarization", f"      участников: {len({t.speaker for t in turns})}, реплик: {len(turns)}")
        del samples

    say("align", "Собираю стенограмму ...")
    utterances = align.assign(segments, turns, max_pause=MAX_PAUSE)
    return Result(
        utterances=utterances,
        turns=turns,
        duration=duration,
        intermediate=Intermediate(segments, language, tiles, labels, step, video_error),
        tile_counts=counts,
        video_error=video_error if want_video else None,
        video_used=video_used,
    )


def _check(opts: Options) -> None:
    """Числа участников, которые ничего не значат. Без проверки speakers=0 молча даёт
    одного участника, а tile_speakers с нулём падает уже после распознавания."""
    if opts.speakers is not None and opts.speakers < 1:
        raise ValueError(f"speakers должен быть не меньше 1 (получено {opts.speakers})")
    if opts.max_speakers < 1:
        raise ValueError(f"max_speakers должен быть не меньше 1 (получено {opts.max_speakers})")
    if opts.tile_speakers is not None and any(n < 1 for n in opts.tile_speakers):
        raise ValueError(f"tile_speakers: за каждой плиткой не меньше 1 человека (получено {opts.tile_speakers})")
