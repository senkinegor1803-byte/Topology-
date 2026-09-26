"""Тесты кеша тайлов (Шаг 2.1, п. 2-3) на реальном Postgres (`pg_test_db`)."""

from __future__ import annotations

from topology_geo.tiling.cache import ensure_schema, find_cached_tile, register_tile
from topology_geo.tiling.grid import TileIndex


def test_find_cached_tile_returns_none_before_registration(pg_test_db):
    ensure_schema(pg_test_db)
    tile = TileIndex(zone=2, tx=1, ty=1)
    assert find_cached_tile(pg_test_db, tile.key("terrain", "osm-2026-01", "gen-v1")) is None


def test_register_and_find_cached_tile(pg_test_db):
    ensure_schema(pg_test_db)
    tile = TileIndex(zone=2, tx=1, ty=1)
    register_tile(pg_test_db, tile, "terrain", "osm-2026-01", "gen-v1", "tiles/msk59-2/1_1/terrain.npz")

    key = tile.key("terrain", "osm-2026-01", "gen-v1")
    assert find_cached_tile(pg_test_db, key) == "tiles/msk59-2/1_1/terrain.npz"


def test_different_layers_of_the_same_tile_are_independent(pg_test_db):
    ensure_schema(pg_test_db)
    tile = TileIndex(zone=2, tx=1, ty=1)
    register_tile(pg_test_db, tile, "terrain", "osm-2026-01", "gen-v1", "tiles/terrain.npz")
    register_tile(pg_test_db, tile, "roads", "osm-2026-01", "gen-v1", "tiles/roads.npz")

    # рельеф (Шаг 2.1) и дороги (Шаг 2.3) одного квадрата - независимые
    # артефакты со своим циклом обновления: слой входит в ключ (grid.py),
    # поэтому регистрация одного не задевает кеш другого.
    assert find_cached_tile(pg_test_db, tile.key("terrain", "osm-2026-01", "gen-v1")) == "tiles/terrain.npz"
    assert find_cached_tile(pg_test_db, tile.key("roads", "osm-2026-01", "gen-v1")) == "tiles/roads.npz"


def test_register_tile_upserts_on_regeneration_race(pg_test_db):
    ensure_schema(pg_test_db)
    tile = TileIndex(zone=2, tx=5, ty=-3)
    register_tile(pg_test_db, tile, "terrain", "osm-2026-01", "gen-v1", "tiles/first.npz")
    register_tile(pg_test_db, tile, "terrain", "osm-2026-01", "gen-v1", "tiles/second.npz")

    key = tile.key("terrain", "osm-2026-01", "gen-v1")
    assert find_cached_tile(pg_test_db, key) == "tiles/second.npz"

    with pg_test_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM tile_cache WHERE tile_key = %s", (key,))
        assert cur.fetchone()[0] == 1


def test_data_version_bump_invalidates_cache(pg_test_db):
    ensure_schema(pg_test_db)
    tile = TileIndex(zone=2, tx=1, ty=1)
    register_tile(pg_test_db, tile, "terrain", "osm-2026-01", "gen-v1", "tiles/old.npz")

    # новая версия данных (например, обновился OSM) - другой tile_key, старый
    # тайл остаётся в кеше нетронутым, но не будет найден по новому ключу
    assert find_cached_tile(pg_test_db, tile.key("terrain", "osm-2026-02", "gen-v1")) is None
    assert find_cached_tile(pg_test_db, tile.key("terrain", "osm-2026-01", "gen-v1")) == "tiles/old.npz"
