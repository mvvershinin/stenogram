"""Общее для тестов сервиса: фикстура хранилища в памяти и автопропуск без ffmpeg.

Ни сети, ни моделей, ни настоящих записей. Тесты с настоящим ffmpeg лежат в tests/integration,
в CI не входят и пропускаются, если бинарника нет.
"""

from __future__ import annotations

import pytest
from fakes import MemoryStorage

from stenogram.audio import find_ffmpeg


@pytest.fixture
def storage() -> MemoryStorage:
    return MemoryStorage()


def _has_ffmpeg() -> bool:
    try:
        find_ffmpeg()
    except RuntimeError:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    if _has_ffmpeg():
        return
    skip = pytest.mark.skip(reason="ffmpeg не найден")
    for item in items:
        if "ffmpeg" in item.keywords:
            item.add_marker(skip)
