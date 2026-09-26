"""Параллельная генерация тайлов через Celery (Шаг 2.1, п. 3).

`generate_terrain_tile_task` — одна задача очереди на один тайл: сперва
проверяет кеш (`tiling.cache.find_cached_tile`) и, если тайл уже готов,
возвращает его немедленно, не тратя воркер на пересчёт («готовые брать из
кеша»). `generate_terrain_tiles_parallel` разом ставит все тайлы в очередь
через `celery.group` — Celery распределяет их по всем подключённым
воркерам (реальный параллелизм, не имитация последовательным циклом).
"""

from __future__ import annotations

import io

import numpy as np
import psycopg
from celery import group

from topology_geo.devcheck import load_environment_config
from topology_geo.relief.service import read_relief_from_storage
from topology_geo.relief.tin import SiteTin
from topology_geo.tasks.celery_app import app
from topology_geo.tasks.pipeline_tasks import get_storage
from topology_geo.tiling.cache import ensure_schema, find_cached_tile, register_tile
from topology_geo.tiling.grid import TileIndex
from topology_geo.tiling.terrain import build_tile_terrain

LAYER_TERRAIN = "terrain"


def _connect() -> psycopg.Connection:
    config = load_environment_config()
    return psycopg.connect(config.postgres.dsn, autocommit=True)


def serialize_tin(tin: SiteTin) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, vertices=tin.vertices, triangles=tin.triangles)
    return buf.getvalue()


def deserialize_tin(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает `(vertices, triangles)` — не полный `SiteTin`, т.к. для этого
    нужен ещё `scipy.spatial.Delaunay`, а сериализованный тайл — уже готовая
    треугольная сетка, повторная триангуляция не нужна."""
    with np.load(io.BytesIO(data)) as npz:
        return npz["vertices"], npz["triangles"]


@app.task(name="topology.generate_terrain_tile")
def generate_terrain_tile_task(
    zone: int, tx: int, ty: int, data_version: str, generator_version: str, relief_storage_key: str
) -> str:
    """Собрать (или найти в кеше) тайл рельефа `(zone, tx, ty)`. Возвращает
    `storage_key` готового тайла."""
    conn = _connect()
    try:
        ensure_schema(conn)
        tile = TileIndex(zone=zone, tx=tx, ty=ty)
        tile_key = tile.key(LAYER_TERRAIN, data_version, generator_version)

        cached = find_cached_tile(conn, tile_key)
        if cached is not None:
            return cached

        storage = get_storage()
        relief_values, relief_grid = read_relief_from_storage(storage, relief_storage_key)
        tin = build_tile_terrain(relief_values, relief_grid, tile)

        storage_key = f"tiles/{tile_key}.npz"
        storage.upload(storage_key, serialize_tin(tin), content_type="application/octet-stream")
        register_tile(conn, tile, LAYER_TERRAIN, data_version, generator_version, storage_key)
        return storage_key
    finally:
        conn.close()


def generate_terrain_tiles_parallel(
    tiles: list[TileIndex], data_version: str, generator_version: str, relief_storage_key: str, *, timeout_s: float = 60.0
) -> dict[TileIndex, str]:
    """Шаг 2.1, п. 3: поставить все тайлы в очередь разом (`celery.group`) и
    дождаться результатов. Порядок результатов соответствует порядку `tiles`
    (Celery `group` это гарантирует)."""
    if not tiles:
        return {}
    job = group(
        generate_terrain_tile_task.s(t.zone, t.tx, t.ty, data_version, generator_version, relief_storage_key)
        for t in tiles
    )
    async_result = job.apply_async()
    storage_keys = async_result.get(timeout=timeout_s)
    return dict(zip(tiles, storage_keys, strict=True))
