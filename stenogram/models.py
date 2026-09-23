"""Загрузка и кэширование моделей: ONNX-разметка участников и распознавание речи.

Источников два, и они живут в разных кэшах:

- ONNX-модели диаризации (сегментация, отпечатки) — с GitHub Releases в `cache_dir()`
  (`STENOGRAM_HOME`, по умолчанию `~/.cache/stenogram`). Это единственное место, где их
  ищет и `diarization.py` (`ensure_all()` без параметров) — поэтому у CLI здесь нет
  флага-обхода: разойдись он со STENOGRAM_HOME, `--check` мог бы говорить «всё на месте»,
  а воркер без переменной окружения уходил бы в сеть за той же моделью;
- модель распознавания whisper — с HuggingFace Hub через сам `faster-whisper`, в
  стандартный кэш `HF_HOME` (по умолчанию `~/.cache/huggingface`) либо в каталог,
  заданный `download_root`/`--whisper-dir`.

`python -m stenogram.models --fetch` (на машине с сетью) и `--check` (в закрытом
контуре, без единого сетевого запроса) — см. `docs/эксплуатация.md`, раздел «Работа
без интернета»."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

SEGMENTATION_URL = f"{RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEGMENTATION_FILE = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"

# CAM++, обучен на VoxCeleb — многоязычный, устойчив к русской речи.
EMBEDDING_URL = f"{RELEASES}/speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx"
EMBEDDING_FILE = "wespeaker_en_voxceleb_CAM++.onnx"

WHISPER_MODEL_DEFAULT = "large-v3-turbo"
# Файлы, без которых WhisperModel либо не загрузится, либо на "tokenizer.json"
# незаметно уйдёт в сеть за токенизатором openai/whisper-*, что на закрытом контуре
# и есть тот самый отказ, который --check обязан поймать заранее.
WHISPER_REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json")

# Зависание на частично живой сети (прокси принимает соединение и молчит) иначе
# ждало бы бесконечно — urlopen без timeout ждёт вечно.
DEFAULT_TIMEOUT = 30.0


def _stenogram_home() -> Path:
    """Путь к кэшу ONNX-моделей — без побочного mkdir. Используется там, где нужно
    только читать (`check_all`): на несмонтированном или недоступном на запись томе
    `cache_dir()` уронил бы mkdir необработанным исключением вместо аккуратного
    «нет модели ... в ...»."""
    root = os.environ.get("STENOGRAM_HOME")
    return Path(root) if root else Path.home() / ".cache" / "stenogram"


def cache_dir() -> Path:
    path = _stenogram_home()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, dst: Path, timeout: float = DEFAULT_TIMEOUT) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  скачиваю {dst.name} ...", flush=True)
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as f:  # noqa: S310
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.rename(dst)
    return dst


def ensure_segmentation_model(root: Path | None = None, timeout: float = DEFAULT_TIMEOUT) -> Path:
    """Скачивает (при необходимости) и распаковывает архив сегментации.

    Распаковка идёт во временный каталог рядом, и в `root` попадает готовая папка
    единственным атомарным `rename` в конце — иначе оборванная распаковка (процесс
    убит на середине записи `model.onnx`) оставляла бы на месте усечённый файл,
    который `Path.exists()` не отличает от целого, и следующий вызов принимал бы его
    за готовую модель.

    Временный каталог — с уникальным именем (`tempfile.mkdtemp`), не фиксированным:
    на общем томе (например, реплики воркера) несколько процессов могут звать эту
    функцию одновременно, оба увидев `target.exists() is False`. Если сосед финиширует
    первым и `root / SEGMENTATION_FILE` появляется, пока мы ещё распаковываем свою
    копию, — не падаем, а принимаем готовую модель соседа."""
    root = root if root is not None else cache_dir()
    target = root / SEGMENTATION_FILE
    if target.exists():
        return target
    archive = _download(SEGMENTATION_URL, root / "segmentation.tar.bz2", timeout)

    extract_dir = Path(tempfile.mkdtemp(dir=root, prefix=f"{target.parent.name}.part-extract-"))
    try:
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(extract_dir, filter="data")
        archive.unlink(missing_ok=True)

        extracted = extract_dir / target.parent.name
        if not (extracted / target.name).exists():  # pragma: no cover
            raise RuntimeError(f"В архиве нет {SEGMENTATION_FILE}")

        if target.exists():
            return target
        try:
            extracted.rename(target.parent)
        except OSError:
            # Гонка между проверкой выше и самим rename: сосед успел создать
            # target.parent (непустой) ровно в этом окне — os.rename на непустой
            # каталог падает. Чужая готовая модель ничем не хуже нашей.
            if target.exists():
                return target
            raise
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
    return target


def ensure_embedding_model(root: Path | None = None, timeout: float = DEFAULT_TIMEOUT) -> Path:
    root = root if root is not None else cache_dir()
    target = root / EMBEDDING_FILE
    return target if target.exists() else _download(EMBEDDING_URL, target, timeout)


def ensure_all(root: Path | None = None, timeout: float = DEFAULT_TIMEOUT) -> tuple[Path, Path]:
    return ensure_segmentation_model(root, timeout), ensure_embedding_model(root, timeout)


def _check_whisper_directory(path: str) -> None:
    """Тот же случай, что `WhisperModel(model_size_or_path)`, когда `model_size_or_path`
    — путь к уже готовому каталогу модели (`os.path.isdir` в faster_whisper/transcribe.py):
    `download_model` тут не зовётся вовсе. Он и не должен зваться — `download_model`
    трактует любую строку с "/" как HuggingFace `repo_id` и падает на локальном пути
    с «Repo id must be in the form...», хотя `WhisperModel(путь)` этот же путь
    загружает нормально."""
    missing = [name for name in WHISPER_REQUIRED_FILES if not (Path(path) / name).exists()]
    if not any(Path(path).glob("vocabulary.*")):
        missing.append("vocabulary.*")
    if missing:
        raise FileNotFoundError(f"неполный каталог модели: нет {', '.join(missing)}")


def ensure_whisper_model(
    model_name: str = WHISPER_MODEL_DEFAULT,
    download_root: str | None = None,
    local_files_only: bool = False,
) -> str:
    """Скачивает или проверяет модель распознавания faster-whisper/CTranslate2.

    `model_name` — имя модели (`small`, `large-v3-turbo`, ...) либо путь к уже готовому
    каталогу модели; путь никогда не скачивается, только проверяется — ровно как
    `WhisperModel` его использует как есть, без похода в HuggingFace.

    Для имени модели повторяет ровно то, что делает `WhisperModel` при загрузке:
    `download_root` уходит в параметр `cache_dir` библиотеки `faster_whisper`, а не
    `output_dir` — если бы было наоборот, `--check`/`--fetch` смотрели бы не в тот
    каталог, где модель на самом деле ищет `asr.load_model(..., download_root=...)`.

    С `local_files_only=True` сеть не трогается: либо возвращается путь к уже
    закэшированной модели, либо летит исключение (`FileNotFoundError` или подкласс
    ошибок `huggingface_hub`, если сеть отключена, а модели на диске нет)."""
    if os.path.isdir(model_name):
        _check_whisper_directory(model_name)
        return model_name

    from faster_whisper import download_model  # noqa: PLC0415 -- тяжёлый импорт: только внутри функции

    path = download_model(model_name, local_files_only=local_files_only, cache_dir=download_root)
    if local_files_only and not (Path(path) / "model.bin").exists():
        # Старый кэш huggingface_hub без tree-listing лениво отдаёт папку снапшота как
        # есть, даже если докачка была прервана: `snapshot_download` со старым кэшем
        # ничего не сверяет с составом репозитория (huggingface_hub/_snapshot_download.py,
        # `_raise_if_incomplete_snapshot` тихо выходит, если trees/<hash>.json нет).
        # Без этой проверки --check мог бы соврать «всё на месте».
        raise FileNotFoundError(f"модель whisper неполная: нет model.bin в {path}")
    return path


def _whisper_location(whisper_model: str, whisper_dir: str | None) -> str:
    """Где на диске искать модель whisper — для сообщений об ошибке. Если
    `whisper_model` сам путь к каталогу, каталог и есть местонахождение. Иначе
    `whisper_dir` (если задан) — это `download_root`/`cache_dir`, а НЕ `HF_HOME`:
    `HF_HOME=X` кладёт модели в `X/hub/...`, а `--whisper-dir X` — прямо в `X/...`."""
    if os.path.isdir(whisper_model):
        return whisper_model
    return whisper_dir or os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")


def check_all(
    *,
    whisper_model: str = WHISPER_MODEL_DEFAULT,
    whisper_dir: str | None = None,
) -> list[str]:
    """Проверяет, что все модели на месте — без единого сетевого запроса и без единой
    записи на диск (только чтение путей, никаких `mkdir`).

    Возвращает список проблем (одна строка на отсутствующую модель, с именем и
    каталогом, где её ждали); пустой список — всё на месте."""
    problems: list[str] = []
    root = _stenogram_home()

    for filename in (SEGMENTATION_FILE, EMBEDDING_FILE):
        target = root / filename
        try:
            found = target.exists()
        except OSError as exc:
            # Path.exists() глотает ENOENT/ENOTDIR и им подобные, но не EACCES:
            # непрочитываемый каталог тома (например, из-за несовпадения UID между
            # хостом и контейнером) иначе ронял бы check_all необработанным
            # исключением вместо понятного «нет модели ... в ...».
            problems.append(f"нет доступа к {filename} в {root}: {exc}")
            continue
        if not found:
            problems.append(f"нет модели {filename} в {root}")
        elif not os.access(target, os.R_OK):
            # exists() успешен по правам РОДИТЕЛЯ (traversal), не по правам самого
            # файла: файл без прав на чтение (мод 000/родительский volume с чужим
            # владельцем) иначе засчитывался бы «моделью на месте», хотя открыть
            # его на чтение нельзя — тот же класс отказа, что ловит diarization
            # при настоящей попытке загрузить ONNX-модель, только раньше по времени.
            problems.append(f"нет доступа к {filename} в {root}: файл не читается (нет прав)")

    try:
        ensure_whisper_model(whisper_model, download_root=whisper_dir, local_files_only=True)
    except Exception as exc:
        problems.append(f"нет модели whisper {whisper_model} в {_whisper_location(whisper_model, whisper_dir)}: {exc}")

    return problems


def fetch_all(
    *,
    whisper_model: str = WHISPER_MODEL_DEFAULT,
    whisper_dir: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    """Скачивает все модели на машине с сетью: ONNX-разметку в `cache_dir()` (то есть
    `STENOGRAM_HOME`) и whisper в `whisper_dir` (умолчание — стандартный кэш `HF_HOME`).
    Повторный вызов бесплатен — уже скачанное не трогается. Если `whisper_model` —
    путь к готовому каталогу, ничего не скачивает, только проверяет.

    Каждый источник скачивается отдельным вызовом, а не через `ensure_all`: при отказе
    сообщение обязано называть модель и каталог (какая из трёх не докачалась), а не
    просто пересказывать текст сетевой ошибки."""
    root = cache_dir()

    try:
        ensure_segmentation_model(root, timeout)
    except Exception as exc:
        raise RuntimeError(f"не удалось скачать {SEGMENTATION_FILE} в {root}: {exc}") from exc

    try:
        ensure_embedding_model(root, timeout)
    except Exception as exc:
        raise RuntimeError(f"не удалось скачать {EMBEDDING_FILE} в {root}: {exc}") from exc

    print(f"  скачиваю модель whisper {whisper_model} ...", flush=True)
    try:
        ensure_whisper_model(whisper_model, download_root=whisper_dir, local_files_only=False)
    except Exception as exc:
        raise RuntimeError(f"не удалось скачать модель whisper {whisper_model} в {_whisper_location(whisper_model, whisper_dir)}: {exc}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m stenogram.models",
        description="Подготовка моделей для машины без интернета: --fetch там, где сеть есть, --check в закрытом контуре.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fetch", action="store_true", help="скачать все модели")
    mode.add_argument("--check", action="store_true", help="проверить, что все модели на месте, без сети")
    parser.add_argument("--whisper", default=WHISPER_MODEL_DEFAULT, help=f"модель распознавания: имя (умолчание {WHISPER_MODEL_DEFAULT}) либо путь к готовому каталогу модели")
    parser.add_argument(
        "--whisper-dir",
        default=None,
        help="каталог модели распознавания — это download_root/cache_dir библиотеки faster-whisper, НЕ HF_HOME "
        "(умолчание: стандартный кэш HF_HOME); чтобы stenogram нашёл модель здесь же, тот же путь нужно передать "
        "в asr.load_model(download_root=...) — у ./stenogram.sh отдельного флага для этого нет, он смотрит на HF_HOME",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"таймаут скачивания ONNX-моделей с GitHub, секунды (умолчание: {DEFAULT_TIMEOUT:g}); "
        "на скачивание whisper с HuggingFace не влияет — там свой таймаут, HF_HUB_DOWNLOAD_TIMEOUT",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.fetch:
        try:
            fetch_all(whisper_model=args.whisper, whisper_dir=args.whisper_dir, timeout=args.timeout)
        except Exception as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            return 1
        print("готово: все модели на месте")
        return 0

    problems = check_all(whisper_model=args.whisper, whisper_dir=args.whisper_dir)
    if problems:
        for line in problems:
            print(line, file=sys.stderr)
        return 1
    print("готово: все модели на месте")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
