"""Тесты stenogram.models без сети и без настоящих моделей.

`faster_whisper.download_model` подменяется в sys.modules — тот же приём, что в
tests/test_asr.py для WhisperModel. urllib.request.urlopen подменяется точечно, там,
где нужно доказать «сеть не потревожена» или «таймаут дошёл до вызова».

ONNX-каталог всегда переопределяется переменной STENOGRAM_HOME (monkeypatch.setenv),
а не отдельным параметром функций: `diarization.ensure_all()` знает только про
STENOGRAM_HOME, и если бы check_all/fetch_all умели смотреть куда-то ещё, `--check`
мог бы сказать «всё на месте» про каталог, который воркер не найдёт (см. review
фазы 2). whisper-каталог по-прежнему передаётся явно, у него есть `--whisper-dir`."""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import tempfile
import types
from pathlib import Path

import pytest

from stenogram import models


def _boom_urlopen(*args, **kwargs):
    raise AssertionError("сеть не должна вызываться в этом тесте")


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    """Подменяет faster_whisper.download_model. Тест сам решает поведение через
    monkeypatch.setattr(module, "download_model", ...)."""
    module = types.ModuleType("faster_whisper")
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return module


def _model_dir_with_bin(tmp_path: Path, name: str = "snapshot") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "model.bin").write_bytes(b"stub")
    (d / "config.json").write_text("{}")
    return d


def _whisper_dir_with_required_files(tmp_path: Path, name: str = "whisper-model") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "model.bin").write_bytes(b"stub")
    (d / "config.json").write_text("{}")
    (d / "tokenizer.json").write_text("{}")
    (d / "vocabulary.json").write_text("[]")
    return d


def _build_tar_bz2(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _download: таймаут доходит до urlopen
# ---------------------------------------------------------------------------


def test_download_passes_timeout_to_urlopen(monkeypatch, tmp_path):
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n):
            return b""

    def fake_urlopen(url, timeout=None):
        seen["url"] = url
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(models.urllib.request, "urlopen", fake_urlopen)

    models._download("https://example.invalid/model.bin", tmp_path / "model.bin", timeout=5.0)

    assert seen["timeout"] == 5.0


def test_download_default_timeout_is_not_none(monkeypatch, tmp_path):
    """urlopen без timeout ждёт вечно — умолчание обязано быть конечным числом."""
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n):
            return b""

    def fake_urlopen(url, timeout=None):
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(models.urllib.request, "urlopen", fake_urlopen)

    models._download("https://example.invalid/model.bin", tmp_path / "model.bin")

    assert seen["timeout"] == models.DEFAULT_TIMEOUT
    assert seen["timeout"] is not None


# ---------------------------------------------------------------------------
# ensure_segmentation_model: атомарная распаковка (регресс на оборванный tar)
# ---------------------------------------------------------------------------


def test_ensure_segmentation_model_extracts_atomically_and_cleans_up_temp_dir(monkeypatch, tmp_path):
    archive_bytes = _build_tar_bz2(
        {
            models.SEGMENTATION_FILE: b"stub-onnx-bytes",
            "sherpa-onnx-pyannote-segmentation-3-0/LICENSE": b"MIT",
        }
    )

    def fake_download(url, dst, timeout=models.DEFAULT_TIMEOUT):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(archive_bytes)
        return dst

    monkeypatch.setattr(models, "_download", fake_download)

    target = models.ensure_segmentation_model(root=tmp_path)

    assert target == tmp_path / models.SEGMENTATION_FILE
    assert target.read_bytes() == b"stub-onnx-bytes"
    assert (tmp_path / "sherpa-onnx-pyannote-segmentation-3-0" / "LICENSE").exists()
    assert list(tmp_path.glob("*.part-extract-*")) == []
    assert not (tmp_path / "segmentation.tar.bz2").exists()


def test_ensure_segmentation_model_interrupted_extraction_leaves_no_partial_target(monkeypatch, tmp_path):
    """Регресс: extractall раньше писал model.onnx прямо на итоговое место — если
    процесс убит на середине записи, следующий запуск видел «файл существует» и
    считал модель готовой, хотя она усечена. Теперь распаковка идёт во временный
    каталог, и на итоговое место модель попадает только одним rename в конце —
    при отказе посередине target вообще не должен появиться."""

    def fake_download(url, dst, timeout=models.DEFAULT_TIMEOUT):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"content doesn't matter - tarfile.open is faked below")
        return dst

    class _BoomTar:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extractall(self, path, filter=None):  # noqa: A002 -- сигнатура должна совпадать с tarfile.TarFile.extractall
            partial = Path(path) / "sherpa-onnx-pyannote-segmentation-3-0"
            partial.mkdir(parents=True, exist_ok=True)
            (partial / "model.onnx").write_bytes(b"PARTIAL-TRUNCATED")
            raise tarfile.ReadError("boom: truncated archive")

    monkeypatch.setattr(models, "_download", fake_download)
    monkeypatch.setattr(models.tarfile, "open", lambda *a, **k: _BoomTar())

    with pytest.raises(tarfile.ReadError):
        models.ensure_segmentation_model(root=tmp_path)

    assert not (tmp_path / models.SEGMENTATION_FILE).exists()
    assert list(tmp_path.glob("*.part-extract-*")) == []


def test_ensure_segmentation_model_uses_unique_extract_dir_name(monkeypatch, tmp_path):
    """Фиксированное имя временного каталога сталкивало бы два одновременных вызова
    (например, реплики воркера на общем томе) друг с другом. tempfile.mkdtemp даёт
    каждому вызову свой каталог."""
    archive_bytes = _build_tar_bz2({models.SEGMENTATION_FILE: b"stub-onnx-bytes"})
    seen_dirs: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def fake_download(url, dst, timeout=models.DEFAULT_TIMEOUT):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(archive_bytes)
        return dst

    def spying_mkdtemp(*args, **kwargs):
        made = real_mkdtemp(*args, **kwargs)
        seen_dirs.append(Path(made))
        return made

    monkeypatch.setattr(models, "_download", fake_download)
    monkeypatch.setattr(models.tempfile, "mkdtemp", spying_mkdtemp)

    models.ensure_segmentation_model(root=tmp_path / "a")
    models.ensure_segmentation_model(root=tmp_path / "b")

    assert len(seen_dirs) == 2
    assert seen_dirs[0] != seen_dirs[1]


# ---------------------------------------------------------------------------
# ensure_segmentation_model: гонка двух процессов на общем томе
# ---------------------------------------------------------------------------


def test_ensure_segmentation_model_accepts_sibling_dir_created_during_race(monkeypatch, tmp_path):
    """Оба процесса видят target.exists() is False на входе и начинают качать/
    распаковывать; один финиширует первым. Второй не должен падать на rename —
    обязан принять уже готовый каталог соседа, а не свой (более свежий) вариант."""
    archive_bytes = _build_tar_bz2({models.SEGMENTATION_FILE: b"stub-onnx-bytes"})

    def fake_download(url, dst, timeout=models.DEFAULT_TIMEOUT):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(archive_bytes)
        return dst

    monkeypatch.setattr(models, "_download", fake_download)

    target = tmp_path / models.SEGMENTATION_FILE
    original_rename = Path.rename
    calls = {"n": 0}

    def racy_rename(self, dst_path):
        calls["n"] += 1
        if calls["n"] == 1 and self.name == "sherpa-onnx-pyannote-segmentation-3-0":
            # Сосед успел закончить между нашей проверкой target.exists() и вызовом
            # rename — тот же эффект даёт os.rename на непустой каталог назначения.
            sibling = Path(dst_path)
            sibling.mkdir(parents=True, exist_ok=True)
            (sibling / "model.onnx").write_bytes(b"from-sibling")
            raise OSError(39, "Directory not empty")
        return original_rename(self, dst_path)

    monkeypatch.setattr(Path, "rename", racy_rename)

    result = models.ensure_segmentation_model(root=tmp_path)

    assert result == target
    assert target.read_bytes() == b"from-sibling"


# ---------------------------------------------------------------------------
# ensure_whisper_model: download_root -> cache_dir, не output_dir
# ---------------------------------------------------------------------------


def test_ensure_whisper_model_passes_download_root_as_cache_dir_not_output_dir(fake_faster_whisper, tmp_path):
    """WhisperModel сам передаёт download_root в параметр cache_dir (faster_whisper/
    transcribe.py). Если бы утилита прокидывала его в output_dir, --check и --fetch
    смотрели бы не в тот каталог, где модель ищет asr.load_model(download_root=...)."""
    seen = {}
    snapshot = _model_dir_with_bin(tmp_path)

    def fake_download_model(size_or_id, local_files_only=False, cache_dir=None, **kwargs):
        seen["kwargs"] = {"size_or_id": size_or_id, "local_files_only": local_files_only, "cache_dir": cache_dir, **kwargs}
        return str(snapshot)

    fake_faster_whisper.download_model = fake_download_model

    result = models.ensure_whisper_model("small", download_root=str(tmp_path / "hf"), local_files_only=False)

    assert seen["kwargs"]["cache_dir"] == str(tmp_path / "hf")
    assert "output_dir" not in seen["kwargs"]
    assert result == str(snapshot)


def test_ensure_whisper_model_local_files_only_true_raises_when_model_bin_missing(fake_faster_whisper, tmp_path):
    """Старый кэш huggingface_hub без tree-listing может лениво отдать неполный
    снапшот как есть — проверяем model.bin сами, а не доверяем одному факту, что
    snapshot_download не бросил исключение."""
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}")  # model.bin нет

    fake_faster_whisper.download_model = lambda *a, **k: str(incomplete)

    with pytest.raises(FileNotFoundError, match=r"model\.bin"):
        models.ensure_whisper_model("small", local_files_only=True)


def test_ensure_whisper_model_local_files_only_false_does_not_require_model_bin_check(fake_faster_whisper, tmp_path):
    """Проверка на неполноту снапшота — только для local_files_only=True: во время
    настоящей загрузки download_model сам гарантирует полноту."""
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()

    fake_faster_whisper.download_model = lambda *a, **k: str(incomplete)

    assert models.ensure_whisper_model("small", local_files_only=False) == str(incomplete)


# ---------------------------------------------------------------------------
# ensure_whisper_model: путь к готовому каталогу — download_model не зовётся вовсе
# ---------------------------------------------------------------------------


def test_ensure_whisper_model_local_directory_path_does_not_call_download_model(fake_faster_whisper, tmp_path):
    """download_model трактует любую строку с "/" как HuggingFace repo_id и падает
    на локальном пути с «Repo id must be...», хотя WhisperModel(путь) этот же путь
    загружает нормально (os.path.isdir-ветка в faster_whisper/transcribe.py) — то
    же самое обязана делать и наша утилита, а не звать download_model."""
    model_dir = _whisper_dir_with_required_files(tmp_path)

    def boom(*a, **k):
        raise AssertionError("для пути к каталогу download_model звать нельзя")

    fake_faster_whisper.download_model = boom

    assert models.ensure_whisper_model(str(model_dir), local_files_only=True) == str(model_dir)
    assert models.ensure_whisper_model(str(model_dir), local_files_only=False) == str(model_dir)


def test_ensure_whisper_model_local_directory_missing_files_raises_and_names_them(fake_faster_whisper, tmp_path):
    model_dir = tmp_path / "incomplete-dir"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")  # model.bin, tokenizer.json, vocabulary.* отсутствуют

    def boom(*a, **k):
        raise AssertionError("для пути к каталогу download_model звать нельзя")

    fake_faster_whisper.download_model = boom

    with pytest.raises(FileNotFoundError) as exc_info:
        models.ensure_whisper_model(str(model_dir), local_files_only=True)

    message = str(exc_info.value)
    assert "model.bin" in message
    assert "tokenizer.json" in message
    assert "vocabulary" in message


def test_ensure_whisper_model_local_directory_fetch_mode_only_checks_does_not_download(fake_faster_whisper, tmp_path):
    """--fetch с путём к каталогу ничего не качает, только проверяет."""
    model_dir = _whisper_dir_with_required_files(tmp_path)

    def boom(*a, **k):
        raise AssertionError("--fetch с путём к каталогу не должен качать")

    fake_faster_whisper.download_model = boom

    assert models.ensure_whisper_model(str(model_dir), local_files_only=False) == str(model_dir)


# ---------------------------------------------------------------------------
# check_all
# ---------------------------------------------------------------------------


def test_check_all_empty_dirs_reports_all_three_models_by_name_and_dir(fake_faster_whisper, tmp_path, monkeypatch):
    stenogram_home = tmp_path / "stenogram-empty"
    whisper_dir = tmp_path / "hf-empty"
    monkeypatch.setenv("STENOGRAM_HOME", str(stenogram_home))
    monkeypatch.setattr(models.urllib.request, "urlopen", _boom_urlopen)

    def fake_download_model(size_or_id, local_files_only=False, cache_dir=None, **kwargs):
        raise FileNotFoundError(f"local_files_only=True и модели нет в {cache_dir}")

    fake_faster_whisper.download_model = fake_download_model

    problems = models.check_all(whisper_model="large-v3-turbo", whisper_dir=str(whisper_dir))

    assert len(problems) == 3
    joined = "\n".join(problems)
    assert models.SEGMENTATION_FILE in joined
    assert models.EMBEDDING_FILE in joined
    assert "large-v3-turbo" in joined
    assert str(stenogram_home) in joined
    assert str(whisper_dir) in joined


def test_check_all_present_returns_empty_and_makes_no_network_call(fake_faster_whisper, tmp_path, monkeypatch):
    stenogram_home = tmp_path / "stenogram-ok"
    seg = stenogram_home / models.SEGMENTATION_FILE
    seg.parent.mkdir(parents=True)
    seg.write_bytes(b"stub")
    emb = stenogram_home / models.EMBEDDING_FILE
    emb.write_bytes(b"stub")
    monkeypatch.setenv("STENOGRAM_HOME", str(stenogram_home))

    whisper_dir = tmp_path / "hf-ok"
    snapshot = _model_dir_with_bin(tmp_path, "snapshot-ok")
    seen = {}

    def fake_download_model(size_or_id, local_files_only=False, cache_dir=None, **kwargs):
        seen["local_files_only"] = local_files_only
        seen["cache_dir"] = cache_dir
        return str(snapshot)

    fake_faster_whisper.download_model = fake_download_model
    monkeypatch.setattr(models.urllib.request, "urlopen", _boom_urlopen)  # сеть — провал теста

    problems = models.check_all(whisper_model="small", whisper_dir=str(whisper_dir))

    assert problems == []
    assert seen["local_files_only"] is True
    assert seen["cache_dir"] == str(whisper_dir)


def test_check_all_names_directory_itself_as_location_for_path_based_whisper_model(tmp_path, monkeypatch):
    stenogram_home = tmp_path / "stenogram-ok"
    (stenogram_home / models.SEGMENTATION_FILE).parent.mkdir(parents=True)
    (stenogram_home / models.SEGMENTATION_FILE).write_bytes(b"stub")
    (stenogram_home / models.EMBEDDING_FILE).write_bytes(b"stub")
    monkeypatch.setenv("STENOGRAM_HOME", str(stenogram_home))
    monkeypatch.setattr(models.urllib.request, "urlopen", _boom_urlopen)

    incomplete_model_dir = tmp_path / "incomplete-dir"
    incomplete_model_dir.mkdir()

    problems = models.check_all(whisper_model=str(incomplete_model_dir))

    assert len(problems) == 1
    assert str(incomplete_model_dir) in problems[0]


def test_check_all_does_not_create_stenogram_home_directory(fake_faster_whisper, tmp_path, monkeypatch):
    """--check только читает: на несмонтированном/недоступном на запись томе mkdir
    внутри cache_dir() уронил бы необработанное исключение вместо «нет модели ... в
    ...». check_all обязан работать через путь без побочного mkdir."""
    stenogram_home = tmp_path / "does-not-exist-yet" / "stenogram"
    monkeypatch.setenv("STENOGRAM_HOME", str(stenogram_home))
    monkeypatch.setattr(models.urllib.request, "urlopen", _boom_urlopen)
    fake_faster_whisper.download_model = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("нет модели"))

    problems = models.check_all(whisper_model="large-v3-turbo")

    assert len(problems) == 3
    assert not stenogram_home.exists()
    assert not stenogram_home.parent.exists()


# ---------------------------------------------------------------------------
# fetch_all
# ---------------------------------------------------------------------------


def test_fetch_all_calls_both_sources_with_stenogram_home_and_timeout(monkeypatch, tmp_path):
    stenogram_home = tmp_path / "stenogram"
    whisper_dir = tmp_path / "hf"
    monkeypatch.setenv("STENOGRAM_HOME", str(stenogram_home))
    seen = {}

    def fake_ensure_segmentation_model(root=None, timeout=models.DEFAULT_TIMEOUT):
        seen["segmentation"] = {"root": root, "timeout": timeout}

    def fake_ensure_embedding_model(root=None, timeout=models.DEFAULT_TIMEOUT):
        seen["embedding"] = {"root": root, "timeout": timeout}

    def fake_ensure_whisper_model(model_name, download_root=None, local_files_only=False):
        seen["whisper"] = {
            "model_name": model_name,
            "download_root": download_root,
            "local_files_only": local_files_only,
        }

    monkeypatch.setattr(models, "ensure_segmentation_model", fake_ensure_segmentation_model)
    monkeypatch.setattr(models, "ensure_embedding_model", fake_ensure_embedding_model)
    monkeypatch.setattr(models, "ensure_whisper_model", fake_ensure_whisper_model)

    models.fetch_all(whisper_model="tiny", whisper_dir=str(whisper_dir), timeout=7.0)

    assert seen["segmentation"] == {"root": stenogram_home, "timeout": 7.0}
    assert seen["embedding"] == {"root": stenogram_home, "timeout": 7.0}
    assert seen["whisper"] == {
        "model_name": "tiny",
        "download_root": str(whisper_dir),
        "local_files_only": False,
    }
    assert stenogram_home.exists()  # --fetch пишет на диск — это ожидаемо, в отличие от --check


def test_fetch_all_onnx_failure_names_model_and_dir(monkeypatch, tmp_path):
    stenogram_home = tmp_path / "stenogram"
    monkeypatch.setenv("STENOGRAM_HOME", str(stenogram_home))

    def boom(root=None, timeout=models.DEFAULT_TIMEOUT):
        raise OSError("connection refused")

    monkeypatch.setattr(models, "ensure_segmentation_model", boom)

    with pytest.raises(RuntimeError) as exc_info:
        models.fetch_all()

    message = str(exc_info.value)
    assert models.SEGMENTATION_FILE in message
    assert str(stenogram_home) in message


def test_fetch_all_whisper_failure_names_model_and_dir(monkeypatch, tmp_path):
    stenogram_home = tmp_path / "stenogram"
    seg = stenogram_home / models.SEGMENTATION_FILE
    seg.parent.mkdir(parents=True)
    seg.write_bytes(b"stub")
    emb = stenogram_home / models.EMBEDDING_FILE
    emb.write_bytes(b"stub")
    monkeypatch.setenv("STENOGRAM_HOME", str(stenogram_home))
    whisper_dir = tmp_path / "hf"

    def boom(model_name, download_root=None, local_files_only=False):
        raise OSError("connection refused")

    monkeypatch.setattr(models, "ensure_whisper_model", boom)

    with pytest.raises(RuntimeError) as exc_info:
        models.fetch_all(whisper_model="tiny", whisper_dir=str(whisper_dir))

    message = str(exc_info.value)
    assert "tiny" in message
    assert str(whisper_dir) in message


# ---------------------------------------------------------------------------
# CLI: main()
# ---------------------------------------------------------------------------


def test_main_check_missing_returns_1_and_prints_problem_lines(monkeypatch, capsys):
    monkeypatch.setattr(models, "check_all", lambda **kwargs: ["нет модели X в /tmp/y"])

    code = models.main(["--check"])

    assert code == 1
    assert "нет модели X в /tmp/y" in capsys.readouterr().err


def test_main_check_ok_returns_0(monkeypatch, capsys):
    monkeypatch.setattr(models, "check_all", lambda **kwargs: [])

    code = models.main(["--check"])

    assert code == 0
    assert "готово" in capsys.readouterr().out


def test_main_fetch_calls_fetch_all_with_parsed_args(monkeypatch):
    seen = {}
    monkeypatch.setattr(models, "fetch_all", lambda **kwargs: seen.update(kwargs))

    code = models.main(["--fetch", "--whisper", "tiny", "--whisper-dir", "/tmp/hf", "--timeout", "9"])

    assert code == 0
    assert seen == {"whisper_model": "tiny", "whisper_dir": "/tmp/hf", "timeout": 9.0}


def test_main_fetch_failure_returns_1_with_message(monkeypatch, capsys):
    def boom(**kwargs):
        raise RuntimeError("нет модели tiny в /tmp/hf: сеть недоступна")

    monkeypatch.setattr(models, "fetch_all", boom)

    code = models.main(["--fetch"])

    assert code == 1
    assert "нет модели tiny" in capsys.readouterr().err


def test_main_requires_fetch_or_check(capsys):
    with pytest.raises(SystemExit) as exc_info:
        models.main([])

    assert exc_info.value.code == 2  # argparse: аргумент обязателен


def test_main_has_no_models_dir_flag():
    """Флаг убран по итогам ревью: --models-dir расходился с STENOGRAM_HOME, который
    единственно знает diarization.ensure_all() — --check с --models-dir мог сказать
    «всё на месте» про каталог, который воркер без той же переменной не найдёт."""
    with pytest.raises(SystemExit) as exc_info:
        models.main(["--check", "--models-dir", "/tmp/x"])

    assert exc_info.value.code == 2


def test_import_stenogram_models_does_not_import_faster_whisper():
    """Тяжёлый импорт — только внутри функций: `import stenogram.models` не должен
    тянуть faster_whisper (иначе `--help` ждёт загрузки ONNX Runtime, см. CLAUDE.md).
    Отдельный подпроцесс — иначе тест не отличит «модуль сам не импортирует» от
    «faster_whisper уже был в sys.modules из-за другого теста, запущенного раньше»."""
    code = "import sys; import stenogram.cli, stenogram.models; sys.exit(1 if 'faster_whisper' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parent.parent, check=False)

    assert result.returncode == 0
