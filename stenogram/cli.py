"""Командный интерфейс: медиафайл → стенограмма с разбивкой по участникам."""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from . import align, asr, audio, diarization, render, video

FORMATS = list(render.WRITERS)


def parse_names(raw: str | None) -> dict[int, str] | None:
    """'Судья,Эксперт' → {0: 'Судья', 1: 'Эксперт'};  '2=Эксперт' → {1: 'Эксперт'}."""
    if not raw:
        return None
    names: dict[int, str] = {}
    for position, item in enumerate(raw.split(",")):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            names[int(key.strip()) - 1] = value.strip()
        else:
            names[position] = item
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stenogram",
        description="Стенограмма диалога из видео или аудио: распознавание речи + разбивка по участникам.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""примеры:
  stenogram запись.mp4
  stenogram запись.mp4 --speakers 8 --names "Судья,Эксперт,Истец"
  stenogram подкаст.mp3 -f txt,srt,json -o ./результаты
  stenogram интервью.wav --model large-v3 --language ru --threads 12
""",
    )
    parser.add_argument("input", type=Path, help="видео- или аудиофайл")
    parser.add_argument("-o", "--output", type=Path,
                        help="папка или путь без расширения (по умолчанию рядом с исходником)")
    parser.add_argument("-f", "--formats", default="txt,md,srt,json",
                        help=f"через запятую: {', '.join(FORMATS)} (по умолчанию txt,md,srt,json)")

    group = parser.add_argument_group("говорящие")
    group.add_argument("-s", "--speakers", type=int,
                       help="точное число участников; без него оценивается автоматически")
    group.add_argument("--max-speakers", type=int, default=12,
                       help="верхняя граница при автооценке (по умолчанию 12)")
    group.add_argument("--names", help='имена участников: "Судья,Эксперт" или "3=Эксперт"')
    group.add_argument("--no-diarization", action="store_true",
                       help="только текст, без разделения на участников")
    group.add_argument("--no-video", action="store_true",
                       help="не использовать подсветку активного говорящего из видео")
    group.add_argument("--tile-speakers",
                       help='сколько человек за каждой плиткой видео, например "7,1"')
    group.add_argument("--highlight-lag", type=float, default=1.2,
                       help="на сколько секунд подсветка отстаёт от речи (по умолчанию 1.2)")

    group = parser.add_argument_group("распознавание")
    group.add_argument("-m", "--model", default="large-v3-turbo",
                       help="модель whisper: tiny/base/small/medium/large-v3/large-v3-turbo")
    group.add_argument("-l", "--language", help="код языка (ru, en, ...); по умолчанию автоопределение")
    group.add_argument("--prompt", help="подсказка модели: тема, термины, имена собственные")
    group.add_argument("--beam-size", type=int, default=5, help="ширина луча (по умолчанию 5)")
    group.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    group.add_argument("--compute-type", default="int8",
                       help="int8 (CPU), float16 (GPU), float32")
    group.add_argument("--threads", type=int, default=0,
                       help="потоков CPU (0 = автоматически)")
    group.add_argument("--denoise", action="store_true",
                       help="шумоподавление и выравнивание громкости перед распознаванием")
    group.add_argument("--keep-wav", action="store_true", help="сохранить извлечённый WAV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"Файл не найден: {args.input}", file=sys.stderr)
        return 1

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        print(f"Неизвестный формат: {', '.join(unknown)}. Доступны: {', '.join(FORMATS)}",
              file=sys.stderr)
        return 1

    if args.output is None:
        stem = args.input.with_suffix("")
    elif args.output.suffix:
        stem = args.output.with_suffix("")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        stem = args.output / args.input.stem

    import os
    threads = args.threads or max(1, (os.cpu_count() or 4) - 2)
    started = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio16k.wav" if not args.keep_wav else stem.with_suffix(".wav")
        print(f"[1/4] Извлекаю аудио из {args.input.name} ...", flush=True)
        audio.extract_wav(args.input, wav, denoise=args.denoise)
        duration = audio.probe_duration(wav)
        print(f"      {duration/60:.1f} мин звука", flush=True)

        tiles = []
        if not args.no_diarization and not args.no_video and video.has_video(args.input):
            print("[2/4] Ищу подсветку активного говорящего в видео ...", flush=True)
            try:
                tiles = video.find_tiles(args.input)
            except Exception as exc:  # видео может быть нестандартным — не повод падать
                print(f"      не получилось ({exc}); останусь на звуке", flush=True)
        if not tiles:
            print("[2/4] Подсветка не найдена — участники только по голосу", flush=True)

        print("[3/4] Распознаю речь ...", flush=True)
        segments, language = asr.transcribe(
            wav, model_name=args.model, language=args.language, beam_size=args.beam_size,
            compute_type=args.compute_type, device=args.device, num_threads=threads,
            initial_prompt=args.prompt,
        )

        print("[4/4] Определяю участников ...", flush=True)
        turns = []
        if not args.no_diarization:
            samples = audio.load_wav(wav)
            if tiles:
                labels, step = video.active_speaker(args.input, tiles)
                video_turns = video.to_turns(labels, step, video.word_gaps(segments),
                                             lag=args.highlight_lag)
                per_tile = [int(x) for x in args.tile_speakers.split(",")] if args.tile_speakers else None
                turns, counts = diarization.diarize_with_video(
                    samples, video_turns, num_speakers=args.speakers,
                    tile_speakers=per_tile, num_threads=max(2, threads // 2),
                )
            else:
                turns = diarization.diarize(
                    samples, num_speakers=args.speakers, max_speakers=args.max_speakers,
                    num_threads=max(2, threads // 2),
                )
            print(f"      участников: {len({t.speaker for t in turns})}, "
                  f"реплик: {len(turns)}", flush=True)
            del samples

    print("Собираю стенограмму ...", flush=True)
    utterances = align.assign(segments, turns, max_pause=3.0)
    names = parse_names(args.names)
    meta = {
        "Источник": args.input.name,
        "Длительность": render.hhmmss(duration),
        "Язык": language,
        "Модель": args.model,
        "Участников": len({u.speaker for u in utterances}),
        "Реплик": len(utterances),
        "Разметка участников": "видео + голос" if tiles else "голос",
    }
    written = render.write(utterances, stem, formats, names=names, meta=meta)

    print(f"\nГотово за {(time.time()-started)/60:.1f} мин:")
    for index, entry in align.speaker_stats(utterances).items():
        print(f"  {render.speaker_name(index, names):<18} {entry['seconds']/60:5.1f} мин "
              f"{entry['share']:4.0%}  реплик: {entry['turns']:>4}  слов: {entry['words']}")
    print()
    for path in written:
        print(f"  → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
