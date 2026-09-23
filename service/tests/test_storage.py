"""Хранилище: разбор адреса, скачивание с sha256, перевод ошибок клиента minio.

Главное, что здесь держится: ни одно исключение хранилища не несёт ключ объекта —
в ключе исходника обычно фамилия и номер дела, а исключения попадают в логи.
Настоящий MinIO проверяется на сквозной приёмке; здесь клиент подменён.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import urllib3.exceptions
from fakes import MemoryStorage
from minio.error import InvalidResponseError, S3Error, ServerError

from stenogram_service.storage import (
    AccessDenied,
    InvalidURI,
    Location,
    MinioStorage,
    ObjectNotFound,
    StorageError,
    StorageUnavailable,
    fetch_source,
    parse_uri,
)

# Вымышленные имя и номер дела — ровно то, что не должно утечь.
SECRET_KEY = "заседания/Петров_А40-12345-2026.mp4"
LOC = Location("records", SECRET_KEY)


# --- разбор адреса ------------------------------------------------------------


def test_parse_uri_keeps_nested_key_as_is():
    assert parse_uri(f"s3://records/{SECRET_KEY}") == LOC
    assert str(LOC) == f"s3://records/{SECRET_KEY}"


@pytest.mark.parametrize(
    "uri",
    ["", "records/x.mp4", "http://records/x.mp4", "s3://", "s3:///x.mp4", "s3://records", "s3://records/", "s3://records/dir/", "s3://Bad_Bucket/x.mp4"],
)
def test_parse_uri_rejects_without_echoing_uri(uri):
    with pytest.raises(InvalidURI) as err:
        parse_uri(uri)
    assert isinstance(err.value, ValueError)  # API отвечает на это 400, а не 500
    assert not err.value.retryable
    assert "x.mp4" not in str(err.value)


# --- скачивание ---------------------------------------------------------------


def test_download_streams_and_hashes(tmp_path, storage):
    data = bytes(range(256)) * 50
    storage.objects[LOC] = data

    sha = storage.download(LOC, tmp_path / "out.bin")

    assert sha == hashlib.sha256(data).hexdigest()
    assert (tmp_path / "out.bin").read_bytes() == data
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.bin"]


def test_fetch_source_names_file_by_hash_not_by_key(tmp_path, storage):
    data = b"\x00\x01video" * 1000
    storage.objects[LOC] = data

    path, sha = fetch_source(storage, f"s3://records/{SECRET_KEY}", tmp_path)

    assert sha == hashlib.sha256(data).hexdigest()
    assert path == tmp_path / f"{sha}.mp4"
    assert [p.name for p in tmp_path.iterdir()] == [f"{sha}.mp4"]
    assert "Петров" not in str(path)


@pytest.mark.parametrize(("key", "ext"), [("x.MKV", ".mkv"), ("без расширения", ""), ("x.тест", ""), ("x.mp4 ; rm -rf", ""), ("x.tar.gz", ".gz")])
def test_fetch_source_keeps_only_plain_extension(tmp_path, storage, key, ext):
    storage.objects[Location("records", key)] = b"data"

    path, sha = fetch_source(storage, f"s3://records/{key}", tmp_path)

    assert path.name == f"{sha}{ext}"


class _Broken(MemoryStorage):
    """Отдаёт кусок и рвёт связь — как обрыв сети посреди большого файла."""

    def iter_object(self, loc):
        yield b"partial"
        raise StorageUnavailable("нет связи с хранилищем", "ProtocolError")


def test_broken_download_leaves_nothing_behind(tmp_path):
    with pytest.raises(StorageUnavailable):
        fetch_source(_Broken(), "s3://records/x.mp4", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_read_and_put_json_round_trip(storage):
    loc = Location("cache", "a/b.json")
    storage.put_json(loc, {"текст": "Добрый день", "n": [1, 2]})

    assert storage.read_json(loc) == {"текст": "Добрый день", "n": [1, 2]}
    assert "Добрый".encode() in storage.objects[loc]  # ensure_ascii=False
    assert storage.content_types[loc] == "application/json"


# --- MinioStorage на подменном клиенте -----------------------------------------


class FakeResponse:
    def __init__(self, chunks, fail_after=None):
        self.chunks, self.fail_after = chunks, fail_after
        self.closed = self.released = False

    def stream(self, amt):
        for i, chunk in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise urllib3.exceptions.ProtocolError(f"Connection broken while reading /records/{SECRET_KEY}")
            yield chunk

    def close(self):
        self.closed = True

    def release_conn(self):
        self.released = True


class FakeClient:
    def __init__(self, error=None, response=None):
        self.error, self.response, self.calls = error, response, []

    def _maybe_fail(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if self.error is not None:
            raise self.error

    def get_object(self, bucket, key):
        self._maybe_fail("get_object", bucket, key)
        return self.response

    def put_object(self, bucket, key, data, length, content_type):
        self._maybe_fail("put_object", bucket, key, data.read(), length, content_type=content_type)

    def fput_object(self, bucket, key, path, content_type):
        self._maybe_fail("fput_object", bucket, key, path, content_type=content_type)

    def stat_object(self, bucket, key):
        self._maybe_fail("stat_object", bucket, key)


def _s3_error(code):
    return S3Error(None, code, f"message about {SECRET_KEY}", f"/records/{SECRET_KEY}", "req-1", "host-1", "records", SECRET_KEY)


CLIENT_ERRORS = [
    (_s3_error("NoSuchKey"), ObjectNotFound),
    (_s3_error("NoSuchBucket"), ObjectNotFound),
    (_s3_error("AccessDenied"), AccessDenied),
    (_s3_error("SignatureDoesNotMatch"), AccessDenied),
    (_s3_error("SlowDown"), StorageUnavailable),
    (_s3_error("InternalError"), StorageUnavailable),
    (_s3_error("InvalidObjectName"), StorageError),
    (ServerError(f"server failed for {SECRET_KEY}", 503), StorageUnavailable),
    (InvalidResponseError(502, "text/html", f"<html>Bad gateway: /records/{SECRET_KEY}</html>"), StorageUnavailable),
    (urllib3.exceptions.MaxRetryError(None, f"/records/{SECRET_KEY}", "refused"), StorageUnavailable),
    (ConnectionResetError(f"reset while sending {SECRET_KEY}"), StorageUnavailable),
]


def _assert_clean(err: BaseException) -> None:
    """Ни ключа в тексте, ни исходной ошибки в цепочке — logger.exception её не напечатает."""
    assert "Петров" not in str(err)
    assert "А40" not in str(err)
    assert err.__cause__ is None
    assert err.__context__ is None


@pytest.mark.parametrize(("error", "expected"), CLIENT_ERRORS, ids=lambda v: getattr(v, "__name__", None))
@pytest.mark.parametrize("operation", ["read", "put_bytes", "put_file", "download"])
def test_client_errors_are_translated_without_key(tmp_path, error, expected, operation):
    store = MinioStorage(FakeClient(error=error))
    call = {
        "read": lambda: store.read_bytes(LOC),
        "put_bytes": lambda: store.put_bytes(LOC, b"x"),
        "put_file": lambda: store.put_file(LOC, tmp_path / "f"),
        "download": lambda: store.download(LOC, tmp_path / "f"),
    }[operation]

    with pytest.raises(expected) as err:
        call()

    assert type(err.value) is expected
    assert err.value.retryable is (expected is StorageUnavailable)
    _assert_clean(err.value)


def test_network_drop_mid_stream_is_retryable_and_releases_connection(tmp_path):
    response = FakeResponse([b"abc", b"def", b"ghi"], fail_after=2)
    store = MinioStorage(FakeClient(response=response))

    with pytest.raises(StorageUnavailable) as err:
        store.download(LOC, tmp_path / "f")

    _assert_clean(err.value)
    assert response.closed and response.released
    assert list(tmp_path.iterdir()) == []


def test_download_through_client_hashes_every_chunk(tmp_path):
    response = FakeResponse([b"abc", b"def", b"ghi"])
    store = MinioStorage(FakeClient(response=response))

    sha = store.download(LOC, tmp_path / "f")

    assert sha == hashlib.sha256(b"abcdefghi").hexdigest()
    assert response.closed and response.released


def test_put_bytes_passes_length_and_type():
    client = FakeClient()
    MinioStorage(client).put_bytes(LOC, "Добрый".encode(), "text/plain")

    assert client.calls == [("put_object", ("records", SECRET_KEY, "Добрый".encode(), len("Добрый".encode())), {"content_type": "text/plain"})]


def test_exists_is_false_only_for_missing_object():
    assert MinioStorage(FakeClient()).exists(LOC) is True
    assert MinioStorage(FakeClient(error=_s3_error("NoSuchKey"))).exists(LOC) is False
    with pytest.raises(AccessDenied) as err:
        MinioStorage(FakeClient(error=_s3_error("AccessDenied"))).exists(LOC)
    _assert_clean(err.value)
    with pytest.raises(StorageUnavailable):
        MinioStorage(FakeClient(error=urllib3.exceptions.MaxRetryError(None, "/x", "refused"))).exists(LOC)


def test_foreign_errors_are_not_disguised_as_storage_errors(tmp_path):
    """Баг в нашем коде или отсутствующий локальный файл — не «повторить позже»."""
    with pytest.raises(TypeError):
        MinioStorage(FakeClient(error=TypeError("bug"))).read_bytes(LOC)
    with pytest.raises(FileNotFoundError):
        MinioStorage(FakeClient(error=FileNotFoundError(str(Path("/nope"))))).put_file(LOC, Path("/nope"))
