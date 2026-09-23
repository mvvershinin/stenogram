"""Объектное хранилище: интерфейс и реализация на MinIO/S3.

Ключ исходника обычно несёт фамилию и номер дела. Поэтому ни одно исключение отсюда
не содержит ни ключа, ни текста исходной ошибки: у `S3Error` в строке ресурс, у
`MaxRetryError` — URL, у `InvalidResponseError` — тело ответа. Наружу выходит только
код S3 или HTTP-статус, исходная ошибка отбрасывается (`from None`), чтобы
`logger.exception` не напечатал её как причину.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import urllib3.exceptions
from minio import Minio
from minio.error import InvalidResponseError, S3Error, ServerError

CHUNK = 1 << 20


class StorageError(Exception):
    """Отказ хранилища. `retryable` — есть ли смысл повторить тот же запрос позже."""

    retryable = False

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message if code is None else f"{message} ({code})")
        self.code = code


class InvalidURI(StorageError, ValueError):
    """Адрес не вида s3://bucket/key."""


class ObjectNotFound(StorageError):
    """Нет объекта или бакета. Повтор не поможет."""


class AccessDenied(StorageError):
    """Нет прав или неверные ключи доступа. Повтор не поможет — нужна настройка."""


class StorageUnavailable(StorageError):
    """Сеть, таймаут, 5xx, перегрузка. Повтор позже может пройти."""

    retryable = True


@dataclass(frozen=True)
class Location:
    bucket: str
    key: str

    def __str__(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


# Имя бакета по правилам S3: 3–63 символа, строчные буквы, цифры, точка и дефис.
_BUCKET = re.compile(r"[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]")


def parse_uri(uri: str) -> Location:
    """`s3://bucket/key` → Location. Текст ошибки не повторяет адрес целиком."""
    scheme, sep, rest = uri.partition("://")
    if not sep or scheme.lower() != "s3":
        raise InvalidURI("адрес исходника должен начинаться с s3://")
    bucket, _, key = rest.partition("/")
    if not _BUCKET.fullmatch(bucket):
        raise InvalidURI("неверное имя бакета в адресе s3://")
    if not key or key.endswith("/"):
        raise InvalidURI("в адресе s3:// нет ключа объекта")
    return Location(bucket, key)


class Storage(ABC):
    """Что сервису нужно от хранилища. Реализации: `MinioStorage` и фейк в тестах."""

    @abstractmethod
    def iter_object(self, loc: Location) -> Iterator[bytes]:
        """Содержимое объекта кусками. Ошибки — только наследники StorageError, в том числе посреди чтения."""

    @abstractmethod
    def read_bytes(self, loc: Location) -> bytes: ...

    @abstractmethod
    def put_bytes(self, loc: Location, data: bytes, content_type: str = "application/octet-stream") -> None: ...

    @abstractmethod
    def put_file(self, loc: Location, path: Path, content_type: str = "application/octet-stream") -> None: ...

    @abstractmethod
    def exists(self, loc: Location) -> bool:
        """Нет объекта — False; нет бакета, прав или сети — исключение."""

    def download(self, loc: Location, dst: Path) -> str:
        """Скачивает объект в `dst`, по пути считая sha256; возвращает его hex.

        Пишет во временный файл рядом и переименовывает в конце: оборванная загрузка
        не оставляет усечённый файл под итоговым именем."""
        digest = hashlib.sha256()
        tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.part")
        try:
            with open(tmp, "wb") as f:
                for chunk in self.iter_object(loc):
                    digest.update(chunk)
                    f.write(chunk)
            os.replace(tmp, dst)
        finally:
            tmp.unlink(missing_ok=True)
        return digest.hexdigest()

    def read_json(self, loc: Location) -> Any:
        return json.loads(self.read_bytes(loc))

    def put_json(self, loc: Location, data: Any) -> None:
        self.put_bytes(loc, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json")


# Расширение — единственное, что берётся из настоящего имени: ffmpeg оно не нужно,
# но человеку, разбирающему temp, помогает. Всё необычное отбрасывается.
_EXT = re.compile(r"\.[a-z0-9]{1,8}")


def fetch_source(storage: Storage, uri: str, workdir: Path) -> tuple[Path, str]:
    """Скачивает исходник в `workdir` под именем `<sha256>.<ext>`; возвращает (путь, sha256).

    Настоящее имя файла дальше не идёт: иначе оно утекает в `[1/4] Извлекаю аудио из …`,
    в «Источник» метаданных и в тексты ошибок ffmpeg."""
    loc = parse_uri(uri)
    ext = os.path.splitext(loc.key)[1].lower()
    ext = ext if _EXT.fullmatch(ext) else ""
    tmp = workdir / f"download-{uuid.uuid4().hex}{ext}"
    sha256 = storage.download(loc, tmp)
    final = workdir / f"{sha256}{ext}"
    os.replace(tmp, final)
    return final, sha256


# Коды S3, при которых повтор не поможет. Всё прочее с кодом — отказ без повтора,
# кроме явно временных (_RETRY_CODES).
_NOT_FOUND_CODES = {"NoSuchKey", "NoSuchBucket", "NoSuchObject", "NotFound", "ResourceNotFound"}
_DENIED_CODES = {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch", "AllAccessDisabled", "AccountProblem", "ExpiredToken", "InvalidToken"}
_RETRY_CODES = {"SlowDown", "InternalError", "ServiceUnavailable", "RequestTimeout", "RequestTimeTooSkewed", "OperationAborted", "XMinioServerNotInitialized"}


def translate_error(exc: BaseException) -> StorageError | None:
    """Ошибка клиента minio/urllib3 → своя, без ключа и текста исходной. None — не наша."""
    if isinstance(exc, StorageError):
        return exc
    if isinstance(exc, S3Error):
        code = exc.code or "unknown"
        if code in _NOT_FOUND_CODES:
            return ObjectNotFound("объект не найден", code)
        if code in _DENIED_CODES:
            return AccessDenied("нет доступа к хранилищу", code)
        if code in _RETRY_CODES:
            return StorageUnavailable("хранилище временно недоступно", code)
        return StorageError("хранилище отказало", code)
    if isinstance(exc, ServerError):
        return StorageUnavailable("хранилище временно недоступно", f"HTTP {exc.status_code}")
    if isinstance(exc, InvalidResponseError):
        # Не-XML ответ: обычно прокси или балансировщик между нами и хранилищем.
        return StorageUnavailable("хранилище ответило не по протоколу S3")
    # Только сетевые: FileNotFoundError локального файла в put_file — не повод повторять.
    if isinstance(exc, (urllib3.exceptions.HTTPError, ConnectionError, TimeoutError)):
        return StorageUnavailable("нет связи с хранилищем", type(exc).__name__)
    return None


def _call(fn, *args, **kwargs):
    """Вызов клиента с переводом ошибок. Своё исключение поднимается вне блока `except`:
    так у него нет ни `__cause__`, ни `__context__` с ключом и URL."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        mapped = translate_error(exc)
        if mapped is None:
            raise
    raise mapped


class MinioStorage(Storage):
    """Хранилище на клиенте `minio`. Клиент передаётся готовым — в тестах подменный."""

    def __init__(self, client: Minio):
        self.client = client

    @classmethod
    def connect(cls, endpoint: str, access_key: str, secret_key: str, *, secure: bool = True, region: str | None = None) -> MinioStorage:
        return cls(Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure, region=region))

    def iter_object(self, loc: Location) -> Iterator[bytes]:
        response = _call(self.client.get_object, loc.bucket, loc.key)
        try:
            stream = response.stream(CHUNK)
            # Обрыв сети посреди чтения всплывает из next(), а не из get_object.
            while (chunk := _call(next, stream, None)) is not None:
                yield chunk
        finally:
            response.close()
            response.release_conn()

    def read_bytes(self, loc: Location) -> bytes:
        return b"".join(self.iter_object(loc))

    def put_bytes(self, loc: Location, data: bytes, content_type: str = "application/octet-stream") -> None:
        _call(self.client.put_object, loc.bucket, loc.key, io.BytesIO(data), len(data), content_type=content_type)

    def put_file(self, loc: Location, path: Path, content_type: str = "application/octet-stream") -> None:
        _call(self.client.fput_object, loc.bucket, loc.key, str(path), content_type=content_type)

    def exists(self, loc: Location) -> bool:
        try:
            self.client.stat_object(loc.bucket, loc.key)
            return True
        except S3Error as exc:
            # HEAD без тела: minio называет 404 на объекте NoSuchKey, даже если нет бакета.
            # Отличить нельзя; отсутствие бакета всплывёт на первой записи.
            if exc.code == "NoSuchKey":
                return False
            mapped = translate_error(exc)
        except Exception as exc:
            mapped = translate_error(exc)
            if mapped is None:
                raise
        raise mapped
