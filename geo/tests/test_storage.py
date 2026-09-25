"""Тесты объектного хранилища (Шаг 1.3, п. 2). MinioObjectStorage не тестируется
здесь реальным сервером — в этой среде разработки нет демона Docker для MinIO;
протокол общий, поэтому FileSystemObjectStorage/InMemoryObjectStorage покрывают
логику, которой пользуется остальной код."""

from __future__ import annotations

import pytest

from topology_geo.storage import FileSystemObjectStorage, InMemoryObjectStorage


@pytest.fixture(params=["memory", "filesystem"])
def storage(request, tmp_path):
    if request.param == "memory":
        return InMemoryObjectStorage()
    return FileSystemObjectStorage(tmp_path / "objects")


def test_upload_then_download_roundtrip(storage):
    storage.upload("a/b/c.txt", b"hello world")
    assert storage.download("a/b/c.txt") == b"hello world"


def test_exists_false_before_upload(storage):
    assert storage.exists("nope.txt") is False


def test_exists_true_after_upload(storage):
    storage.upload("k.bin", b"\x00\x01")
    assert storage.exists("k.bin") is True


def test_download_missing_key_raises(storage):
    with pytest.raises((KeyError, FileNotFoundError)):
        storage.download("missing.txt")


def test_filesystem_storage_rejects_path_escape(tmp_path):
    storage = FileSystemObjectStorage(tmp_path / "objects")
    with pytest.raises(ValueError):
        storage.upload("../escape.txt", b"data")


def test_filesystem_storage_creates_nested_dirs(tmp_path):
    storage = FileSystemObjectStorage(tmp_path / "objects")
    storage.upload("jobs/123/relief.tif", b"tiffdata")
    assert (tmp_path / "objects" / "jobs" / "123" / "relief.tif").read_bytes() == b"tiffdata"
