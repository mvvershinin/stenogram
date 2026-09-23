"""Подделки для тестов: хранилище в памяти за тем же интерфейсом, что MinioStorage, и синтетический звук."""

from __future__ import annotations

import math
import struct
import wave
from collections.abc import Iterator
from pathlib import Path

from stenogram_service.storage import Location, ObjectNotFound, Storage, StorageError


class MemoryStorage(Storage):
    """Словарь вместо бакета. `fail` — какую ошибку бросать на каком ключе;
    `writes` — порядок записей, чтобы проверять, что метка целостности пишется последней."""

    def __init__(self, chunk: int = 7):
        self.objects: dict[Location, bytes] = {}
        self.content_types: dict[Location, str] = {}
        self.writes: list[Location] = []
        self.reads: list[Location] = []
        self.fail: dict[Location, StorageError] = {}
        self.chunk = chunk

    def _check(self, loc: Location) -> None:
        if loc in self.fail:
            raise self.fail[loc]

    def iter_object(self, loc: Location) -> Iterator[bytes]:
        self.reads.append(loc)
        self._check(loc)
        if loc not in self.objects:
            raise ObjectNotFound("объект не найден", "NoSuchKey")
        data = self.objects[loc]
        for i in range(0, len(data), self.chunk):
            yield data[i : i + self.chunk]

    def read_bytes(self, loc: Location) -> bytes:
        return b"".join(self.iter_object(loc))

    def put_bytes(self, loc: Location, data: bytes, content_type: str = "application/octet-stream") -> None:
        self._check(loc)
        self.objects[loc] = bytes(data)
        self.content_types[loc] = content_type
        self.writes.append(loc)

    def put_file(self, loc: Location, path: Path, content_type: str = "application/octet-stream") -> None:
        self.put_bytes(loc, Path(path).read_bytes(), content_type)

    def exists(self, loc: Location) -> bool:
        self._check(loc)
        return loc in self.objects


def write_tone(path: Path, seconds: float = 1.0, rate: int = 16000) -> Path:
    """Моно 16 кГц, 16 бит: синус с медленно меняющейся громкостью — FLAC не сожмёт его в ноль."""
    frames = bytearray()
    for n in range(int(seconds * rate)):
        amp = 0.2 + 0.6 * (n / (seconds * rate))
        frames += struct.pack("<h", int(32767 * amp * math.sin(2 * math.pi * 440 * n / rate)))
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(bytes(frames))
    return path
