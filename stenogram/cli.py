"""Командный интерфейс: медиафайл → стенограмма с разбивкой по участникам.

tarfile, subprocess и wave импортируются на уровне модуля ради общего перехватчика
исключений в main(): их классы ошибок не наследуются от OSError и потому не ловятся
вместе с ним. Правилу отложенных импортов это не противоречит — оно про faster_whisper,
sherpa_onnx и sklearn; здесь же лёгкие модули стандартной библиотеки, все три и так уже
загружены (audio импортирует subprocess и wave, models — tarfile), так что на время
`--help` они не влияют.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import wave
from pathlib import Path
from urllib.error import URLError

from . import align, asr, audio, diarization, render, video

FORMATS = list(render.WRITERS)

# Переменная окружения, а не флаг: включается перед любой командой, в том числе перед
# той, что падает раньше разбора аргументов, и не расширяет и без того длинный --help.
_TRACEBACK_ENV = "STENOGRAM_TRACEBACK"


def _traceback_enabled() -> bool:
    """Показывать ли полный трейсбек вместо одной строки «Ошибка: ...»."""
    return os.environ.get(_TRACEBACK_ENV, "").strip() not in ("", "0")


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
            index = int(key.strip())
            if index < 1:
                raise ValueError(f'Неверный номер участника в --names: «{key.strip()}» (нумерация с 1, например "3=Эксперт")')
            names[index - 1] = value.strip()
        else:
            names[position] = item
    return names


def parse_tile_speakers(raw: str | None) -> list[int] | None:
    """'7,1' → [7, 1]: сколько человек сидит за каждой плиткой видео.

    Числа идут по плиткам, найденным в кадре, слева направо — ровно в том порядке, в
    каком их отдаёт video.find_tiles, и по одному на каждую. Плитка, которая за весь
    разговор ни разу не подсветилась, свой номер в списке всё равно занимает: метка
    плитки — это индекс в списке find_tiles, и пропуск сдвинул бы числа на соседей.

    Разбор ни от чего в пайплайне не зависит, поэтому вызывается на границе CLI, рядом
    с проверкой --max-speakers: опечатка в аргументе не должна стоить пользователю
    полного прохода распознавания (на часовой записи — 35 минут).

    Длину списка здесь проверить нельзя — число плиток известно только после их поиска
    в видео. Эта проверка стоит в main() сразу после шага [2/4], всё ещё до распознавания.
    """
    if not raw:
        return None
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f'--tile-speakers: пустое значение в списке «{raw}»; ожидаются целые числа через запятую, например "7,1"')
        try:
            count = int(item)
        except ValueError:
            raise ValueError(f'--tile-speakers: «{item}» — не целое число; ожидаются целые числа через запятую, например "7,1"') from None
        if count < 1:
            raise ValueError(f"--tile-speakers: «{item}» — за плиткой не может быть меньше одного человека, нужно число не меньше 1")
        values.append(count)
    return values


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

если что-то пошло не так:
  ошибки печатаются одной строкой «Ошибка: ...». Для баг-репорта нужен трейсбек:
  STENOGRAM_TRACEBACK=1 stenogram запись.mp4
""",
    )
    parser.add_argument("input", type=Path, help="видео- или аудиофайл")
    parser.add_argument("-o", "--output", type=Path, help="папка или путь без расширения (по умолчанию рядом с исходником)")
    parser.add_argument("-f", "--formats", default="txt,md,srt,json", help=f"через запятую: {', '.join(FORMATS)} (по умолчанию txt,md,srt,json)")

    group = parser.add_argument_group("говорящие")
    group.add_argument("-s", "--speakers", type=int, help="точное число участников; без него оценивается автоматически")
    group.add_argument("--max-speakers", type=int, default=12, help="верхняя граница при автооценке (по умолчанию 12)")
    group.add_argument("--names", help='имена участников: "Судья,Эксперт" или "3=Эксперт"')
    group.add_argument("--no-diarization", action="store_true", help="только текст, без разделения на участников")
    group.add_argument("--no-video", action="store_true", help="не использовать подсветку активного говорящего из видео")
    group.add_argument("--tile-speakers", help='сколько человек за каждой плиткой видео, слева направо, например "7,1"')
    group.add_argument("--highlight-lag", type=float, default=1.2, help="на сколько секунд подсветка отстаёт от речи (по умолчанию 1.2)")

    group = parser.add_argument_group("распознавание")
    group.add_argument("-m", "--model", default="large-v3-turbo", help="модель whisper: tiny/base/small/medium/large-v3/large-v3-turbo")
    group.add_argument("-l", "--language", help="код языка (ru, en, ...); по умолчанию автоопределение")
    group.add_argument("--prompt", help="подсказка модели: тема, термины, имена собственные")
    group.add_argument("--beam-size", type=int, default=5, help="ширина луча (по умолчанию 5)")
    group.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    group.add_argument("--compute-type", default="int8", help="int8 (CPU), float16 (GPU), float32")
    group.add_argument("--threads", type=int, default=0, help="потоков CPU (0 = автоматически)")
    group.add_argument("--denoise", action="store_true", help="шумоподавление и выравнивание громкости перед распознаванием")
    group.add_argument("--keep-wav", action="store_true", help="сохранить извлечённый WAV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if not args.input.exists():
            print(f"Файл не найден: {args.input}", file=sys.stderr)
            return 1

        formats = [f.strip() for f in args.formats.split(",") if f.strip()]
        unknown = [f for f in formats if f not in FORMATS]
        if unknown:
            print(f"Неизвестный формат: {', '.join(unknown)}. Доступны: {', '.join(FORMATS)}", file=sys.stderr)
            return 1

        if args.max_speakers < 1:
            print(f"--max-speakers должен быть не меньше 1 (получено {args.max_speakers})", file=sys.stderr)
            return 1

        # Без этой проверки --speakers 0 и --speakers -3 не падают, а молча дают
        # стенограмму на одного участника: cluster() отвечает нулями на любое
        # num_speakers <= 1, и узнать об этом можно только по готовому файлу — после
        # полного прохода. Отказ на границе CLI дешевле на 35 минут.
        if args.speakers is not None and args.speakers < 1:
            print(f"--speakers должен быть не меньше 1 (получено {args.speakers})", file=sys.stderr)
            return 1

        try:
            per_tile = parse_tile_speakers(args.tile_speakers)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        names = parse_names(args.names)

        if args.output is None:
            stem = args.input.with_suffix("")
        elif args.output.suffix:
            stem = args.output.with_suffix("")
        else:
            stem = args.output / args.input.stem
        stem.parent.mkdir(parents=True, exist_ok=True)

        threads = args.threads or max(1, (os.cpu_count() or 4) - 2)
        started = time.time()

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio16k.wav" if not args.keep_wav else stem.with_suffix(".wav")
            print(f"[1/4] Извлекаю аудио из {args.input.name} ...", flush=True)
            audio.extract_wav(args.input, wav, denoise=args.denoise)
            duration = audio.probe_duration(wav)
            print(f"      {duration / 60:.1f} мин звука", flush=True)

            tiles = []
            if not args.no_diarization and not args.no_video and video.has_video(args.input):
                print("[2/4] Ищу подсветку активного говорящего в видео ...", flush=True)
                try:
                    tiles = video.find_tiles(args.input)
                except Exception as exc:  # видео может быть нестандартным — не повод падать
                    print(f"      не получилось ({exc}); останусь на звуке", flush=True)
                    if _traceback_enabled():
                        # Печатаем, но не роняем: тихий откат на звук — штатное поведение,
                        # и режим отладки его не отменяет, а только объясняет.
                        import traceback  # noqa: PLC0415 -- лениво: иначе модуль грузится ради каждого `--help`

                        traceback.print_exc()
            if not tiles:
                print("[2/4] Подсветка не найдена — участники только по голосу", flush=True)
                if per_tile is not None:
                    # Иначе флаг молча ничего не делает: без плиток делить некого,
                    # и пользователь узнаёт об этом только по числу участников в сводке.
                    print("      --tile-speakers не пригодился: плиток нет, помочь может --speakers", flush=True)

            # Число плиток известно только сейчас — зато всё ещё до распознавания,
            # то есть несовпадение длины не стоит пользователю самого долгого этапа.
            # Сверяется именно с найденными в кадре плитками — по ним и нумеруются числа
            # (diarization._tile_counts). Дальше по пайплайну часть плиток может отсеяться,
            # но список от этого короче не становится: там числа берутся по индексу плитки.
            if tiles and per_tile is not None and len(per_tile) != len(tiles):
                print(
                    f"--tile-speakers: чисел передано — {len(per_tile)}, а плиток в видео найдено — {len(tiles)}; нужно по одному числу "
                    "на каждую найденную плитку, слева направо — включая те, что за весь разговор ни разу не подсвечены",
                    file=sys.stderr,
                )
                return 1

            print("[3/4] Распознаю речь ...", flush=True)
            segments, language = asr.transcribe(
                wav,
                model_name=args.model,
                language=args.language,
                beam_size=args.beam_size,
                compute_type=args.compute_type,
                device=args.device,
                num_threads=threads,
                initial_prompt=args.prompt,
            )

            print("[4/4] Определяю участников ...", flush=True)
            turns = []
            if not args.no_diarization:
                samples = audio.load_wav(wav)
                if tiles:
                    labels, step = video.active_speaker(args.input, tiles)
                    video_turns = video.to_turns(labels, step, video.word_gaps(segments), lag=args.highlight_lag)
                    turns, counts = diarization.diarize_with_video(  # noqa: RUF059 -- counts не используется в CLI: распределение по плиткам печатает сама diarization
                        samples,
                        video_turns,
                        num_speakers=args.speakers,
                        tile_speakers=per_tile,
                        num_threads=max(2, threads // 2),
                    )
                else:
                    turns = diarization.diarize(
                        samples,
                        num_speakers=args.speakers,
                        max_speakers=args.max_speakers,
                        num_threads=max(2, threads // 2),
                    )
                print(f"      участников: {len({t.speaker for t in turns})}, реплик: {len(turns)}", flush=True)
                del samples

        print("Собираю стенограмму ...", flush=True)
        utterances = align.assign(segments, turns, max_pause=3.0)
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

        print(f"\nГотово за {(time.time() - started) / 60:.1f} мин:", flush=True)
        for index, entry in align.speaker_stats(utterances).items():
            row = f"  {render.speaker_name(index, names):<18} {entry['seconds'] / 60:5.1f} мин {entry['share']:4.0%}  реплик: {entry['turns']:>4}  слов: {entry['words']}"
            print(row, flush=True)
        print(flush=True)
        for path in written:
            print(f"  → {path}", flush=True)

        # Последний сброс — обязательно ВНУТРИ try. Иначе всё, что осталось в буфере
        # stdout, уходит в трубу уже при завершении интерпретатора, то есть вне всякого
        # перехватчика: `stenogram … | head` даёт «Exception ignored» в stderr и код
        # возврата 120. Ровно из-за этого одного flush ветка BrokenPipeError ниже
        # закрывала проблему лишь наполовину.
        sys.stdout.flush()
        return 0

    except KeyboardInterrupt:
        print("\nПрервано пользователем", file=sys.stderr)
        return 130

    # BrokenPipeError — подкласс OSError, но не отказ: так `stenogram … | head` сообщает,
    # что получателю хватит. Ветка обязана стоять ВЫШЕ общего перехватчика, иначе штатное
    # завершение конвейера выглядит как ошибка пользователя.
    except BrokenPipeError:
        # Иначе интерпретатор при выходе ещё раз попробует сбросить буфер stdout в ту же
        # закрытую трубу и допишет в stderr своё «Exception ignored».
        try:
            fd = sys.stdout.fileno()
        except (OSError, ValueError):  # stdout без файлового дескриптора (перехват в тестах)
            return 0
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, fd)
        os.close(devnull)
        return 0

    # Три класса, которые не наследуются от OSError и потому раньше всплывали сырым
    # трейсбеком:
    #   tarfile.ReadError          — прокси отдал HTML вместо архива модели (models.py);
    #   subprocess.CalledProcessError — ffprobe/ffmpeg вернул ненулевой код
    #                                (video.active_speaker, audio.probe_duration);
    #   wave.Error                 — WAV нечитаем (audio.load_wav).
    # Перечислены базовые классы семейств, а не только эти три: соседи по семейству
    # (ExtractError у битого архива, TimeoutExpired у внешнего процесса) — тот же режим
    # отказа и требуют того же обращения.
    except (RuntimeError, ValueError, OSError, URLError, tarfile.TarError, subprocess.SubprocessError, wave.Error) as exc:
        # Перехватчик покрывает всё тело main(), поэтому под него попадают и ошибки,
        # у которых текст ничего не объясняет: RecursionError (подкласс RuntimeError),
        # ValueError из numpy. Одна строка — плохой баг-репорт от постороннего человека,
        # поэтому есть способ получить обычный трейсбек, не правя код.
        if _traceback_enabled():
            raise
        print(f"Ошибка: {exc}", file=sys.stderr)
        print(f"Трейсбек для баг-репорта: повторите команду с {_TRACEBACK_ENV}=1", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
