"""Кеш готовых тайлов в PostGIS (Шаг 2.1, п. 2-3): «готовые брать из кеша» —
до генерации тайла проверяется `find_cached_tile`, после — `register_tile`.
Ключ тайла (`grid.TileIndex.key`) уже содержит координаты + слой + версию
данных + версию генератора, поэтому таблица хранит их отдельными колонками
только для выборки «все тайлы такой-то зоны/слоя» (инвалидация/статистика) —
сам кеш-хит определяется исключительно по `tile_key`.

`TILE_CACHE_SCHEMA_SQL` — встроенная копия `geo/sql/005_tile_cache.sql`
(тот же приём, что и в `topology_geo.ifc.registry`, синхронность проверена
тестом `geo/tests/test_tiling_sql_migrations.py`).
"""

from __future__ import annotations

from typing import Any, Protocol

from topology_geo.tiling.grid import TileIndex


class _Connection(Protocol):
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...


TILE_CACHE_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS tile_cache (
    id BIGSERIAL PRIMARY KEY,
    tile_key TEXT NOT NULL UNIQUE,
    zone INTEGER NOT NULL,
    tx INTEGER NOT NULL,
    ty INTEGER NOT NULL,
    layer TEXT NOT NULL,
    data_version TEXT NOT NULL,
    generator_version TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS tile_cache_zone_tx_ty_idx ON tile_cache (zone, tx, ty, layer);
"""


def ensure_schema(conn: _Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(TILE_CACHE_SCHEMA_SQL)
    conn.commit()


def find_cached_tile(conn: _Connection, tile_key: str) -> str | None:
    """Вернуть `storage_key` уже готового тайла, либо None, если тайла с таким
    ключом ещё нет (нужно генерировать). `tile_key` — `grid.TileIndex.key(layer, ...)`."""
    with conn.cursor() as cur:
        cur.execute("SELECT storage_key FROM tile_cache WHERE tile_key = %s", (tile_key,))
        row = cur.fetchone()
    return row[0] if row else None


def register_tile(
    conn: _Connection, tile: TileIndex, layer: str, data_version: str, generator_version: str, storage_key: str
) -> None:
    """Зарегистрировать готовый тайл. Идемпотентно: повторная генерация того
    же ключа (например, гонка двух воркеров) обновляет `storage_key`, а не
    падает на уникальности."""
    tile_key = tile.key(layer, data_version, generator_version)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO tile_cache (tile_key, zone, tx, ty, layer, data_version, generator_version, storage_key) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (tile_key) DO UPDATE SET storage_key = EXCLUDED.storage_key",
            (tile_key, tile.zone, tile.tx, tile.ty, layer, data_version, generator_version, storage_key),
        )
    conn.commit()
