"""Тесты колец LOD и тайловой сетки (Шаг 2.1, п. 1-2)."""

from __future__ import annotations

import math

import pytest

from topology_geo.tiling.grid import (
    LOD0,
    LOD1,
    LOD2,
    TILE_SIZE_M,
    TileIndex,
    classify_lod_ring,
    tiles_covering_circle,
    world_to_tile_index,
)


@pytest.mark.parametrize(
    "distance_m,expected",
    [(0.0, LOD2), (499.99, LOD2), (500.0, LOD2), (500.01, LOD1), (1500.0, LOD1), (1500.01, LOD0), (3000.0, LOD0)],
)
def test_classify_lod_ring_boundaries(distance_m, expected):
    assert classify_lod_ring(distance_m) == expected


def test_classify_lod_ring_rejects_negative_distance():
    with pytest.raises(ValueError):
        classify_lod_ring(-1.0)


def test_classify_lod_ring_rejects_beyond_stage2_radius():
    with pytest.raises(ValueError):
        classify_lod_ring(3000.01)


def test_world_to_tile_index_is_floor_division():
    assert world_to_tile_index(2, 0.0, 0.0) == TileIndex(zone=2, tx=0, ty=0)
    assert world_to_tile_index(2, 249.9, 0.0) == TileIndex(zone=2, tx=0, ty=0)
    assert world_to_tile_index(2, 250.0, 0.0) == TileIndex(zone=2, tx=1, ty=0)
    assert world_to_tile_index(2, -0.1, -0.1) == TileIndex(zone=2, tx=-1, ty=-1)


def test_tile_bounds_are_contiguous_with_no_overlap():
    tile = TileIndex(zone=2, tx=3, ty=-2)
    minx, miny, maxx, maxy = tile.bounds()
    assert maxx - minx == pytest.approx(TILE_SIZE_M)
    assert maxy - miny == pytest.approx(TILE_SIZE_M)
    neighbor = TileIndex(zone=2, tx=4, ty=-2)
    n_minx, _, _, _ = neighbor.bounds()
    assert n_minx == pytest.approx(maxx)  # без щели и без нахлёста


def test_tile_key_changes_with_any_component():
    tile = TileIndex(zone=2, tx=1, ty=1)
    base = tile.key("terrain", "osm-2026-01", "gen-v1")
    assert base != tile.key("roads", "osm-2026-01", "gen-v1")  # другой слой
    assert base != tile.key("terrain", "osm-2026-02", "gen-v1")  # изменились исходные данные
    assert base != tile.key("terrain", "osm-2026-01", "gen-v2")  # изменился генератор
    assert base != TileIndex(zone=2, tx=2, ty=1).key("terrain", "osm-2026-01", "gen-v1")  # другой тайл
    assert base != TileIndex(zone=3, tx=1, ty=1).key("terrain", "osm-2026-01", "gen-v1")  # другая зона
    assert base == tile.key("terrain", "osm-2026-01", "gen-v1")  # детерминированность


def test_tiles_covering_circle_all_within_tile_size_of_radius():
    tiles = tiles_covering_circle(zone=2, center_x=1000.0, center_y=1000.0, radius_m=300.0)
    assert len(tiles) > 0
    for tile in tiles:
        minx, miny, maxx, maxy = tile.bounds()
        nearest_x = min(max(1000.0, minx), maxx)
        nearest_y = min(max(1000.0, miny), maxy)
        dist = math.hypot(1000.0 - nearest_x, 1000.0 - nearest_y)
        assert dist <= 300.0 + 1e-9

    # ни один тайл, гарантированно вне круга (весь квадрат дальше radius+diag), не пропущен
    all_tx = [t.tx for t in tiles]
    all_ty = [t.ty for t in tiles]
    for tx in range(min(all_tx) - 1, max(all_tx) + 2):
        for ty in range(min(all_ty) - 1, max(all_ty) + 2):
            candidate = TileIndex(zone=2, tx=tx, ty=ty)
            minx, miny, maxx, maxy = candidate.bounds()
            nearest_x = min(max(1000.0, minx), maxx)
            nearest_y = min(max(1000.0, miny), maxy)
            dist = math.hypot(1000.0 - nearest_x, 1000.0 - nearest_y)
            if dist <= 300.0:
                assert candidate in tiles


def test_tiles_covering_circle_is_symmetric_for_centered_circle():
    # центр круга - точно в углу четырёх тайлов -> сетка симметрична по x и по y
    tiles = tiles_covering_circle(zone=2, center_x=0.0, center_y=0.0, radius_m=300.0)
    tile_set = {(t.tx, t.ty) for t in tiles}
    for tx, ty in tile_set:
        assert (-tx - 1, ty) in tile_set  # отражение по x (тайлы 0-индексированы от границы)
        assert (tx, -ty - 1) in tile_set  # отражение по y


def test_tiles_covering_circle_rejects_non_positive_radius():
    with pytest.raises(ValueError):
        tiles_covering_circle(zone=2, center_x=0.0, center_y=0.0, radius_m=0.0)
