"""Объектное хранилище результатов конвейера (Шаг 1.3, п. 2: «каждая
[подзадача] пишет результат в MinIO»).

Три реализации одного протокола:
- `MinioObjectStorage` — прод (MinIO/S3), поверх `topology_geo.devcheck.MinioConfig`;
- `FileSystemObjectStorage` — локальный каталог; годится и для разработки без
  MinIO, и для тестов с реальным Celery-воркером в отдельном процессе (общий
  диск делает её видимой из любого процесса, в отличие от `InMemoryObjectStorage`);
- `InMemoryObjectStorage` — для тестов в одном процессе.

Настоящего MinIO в этой среде разработки нет (нет демона Docker) —
`MinioObjectStorage` не имеет прямого теста с реальным сервером, но
реализует тот же протокол, что и остальные, поэтому переключение в проде не
требует изменений в вызывающем коде.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from topology_geo.devcheck import MinioConfig


class ObjectStorage(Protocol):
    def upload(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> None: ...
    def download(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...


class InMemoryObjectStorage:
    """Хранилище в памяти одного процесса — для модульных тестов."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def upload(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> None:
        self._data[key] = data

    def download(self, key: str) -> bytes:
        return self._data[key]

    def exists(self, key: str) -> bool:
        return key in self._data


class FileSystemObjectStorage:
    """Хранилище на локальном диске — dev-режим и тесты с воркером в отдельном процессе."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if self._root.resolve() not in path.parents and path != self._root.resolve():
            raise ValueError(f"ключ {key!r} выходит за пределы хранилища")
        return path

    def upload(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def download(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


class MinioObjectStorage:
    """Хранилище поверх MinIO/S3 (прод), см. `topology_geo.devcheck.MinioConfig`."""

    def __init__(self, config: "MinioConfig", bucket: str) -> None:
        import minio

        self._client = minio.Minio(
            config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure,
        )
        self._bucket = bucket
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)

    def upload(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> None:
        self._client.put_object(self._bucket, key, io.BytesIO(data), length=len(data), content_type=content_type)

    def download(self, key: str) -> bytes:
        response = self._client.get_object(self._bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def exists(self, key: str) -> bool:
        import minio.error

        try:
            self._client.stat_object(self._bucket, key)
            return True
        except minio.error.S3Error:
            return False
