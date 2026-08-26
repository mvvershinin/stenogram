"""Загрузка и кэширование ONNX-моделей для диаризации."""

from __future__ import annotations

import os
import tarfile
import urllib.request
from pathlib import Path

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

SEGMENTATION_URL = f"{RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEGMENTATION_FILE = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"

# CAM++, обучен на VoxCeleb — многоязычный, устойчив к русской речи.
EMBEDDING_URL = f"{RELEASES}/speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx"
EMBEDDING_FILE = "wespeaker_en_voxceleb_CAM++.onnx"


def cache_dir() -> Path:
    root = os.environ.get("STENOGRAM_HOME")
    path = Path(root) if root else Path.home() / ".cache" / "stenogram"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  скачиваю {dst.name} ...", flush=True)
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:  # noqa: S310
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.rename(dst)
    return dst


def ensure_segmentation_model() -> Path:
    root = cache_dir()
    target = root / SEGMENTATION_FILE
    if target.exists():
        return target
    archive = _download(SEGMENTATION_URL, root / "segmentation.tar.bz2")
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(root, filter="data")
    archive.unlink(missing_ok=True)
    if not target.exists():  # pragma: no cover
        raise RuntimeError(f"В архиве нет {SEGMENTATION_FILE}")
    return target


def ensure_embedding_model() -> Path:
    target = cache_dir() / EMBEDDING_FILE
    return target if target.exists() else _download(EMBEDDING_URL, target)


def ensure_all() -> tuple[Path, Path]:
    return ensure_segmentation_model(), ensure_embedding_model()
