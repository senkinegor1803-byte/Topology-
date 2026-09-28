"""Тесты Шага 2.12, п. 2: тайлы рельефа terrain-RGB."""

from __future__ import annotations

import io

import numpy as np
import rasterio
from affine import Affine

from topology_geo.citymap.terrain_rgb import (
    MAX_ENCODABLE_M,
    MIN_ENCODABLE_M,
    decode_terrain_rgb,
    encode_terrain_rgb,
    lonlat_to_tile,
    render_terrain_rgb_tile,
    tile_bounds_3857,
    tiles_covering_bbox,
)
from topology_geo.relief.service import Grid


def test_tile_bounds_3857_zoom0_covers_whole_globe():
    minx, miny, maxx, maxy = tile_bounds_3857(0, 0, 0)
    origin_shift = 20_037_508.342_789_244
    assert abs(minx + origin_shift) < 1e-2
    assert abs(maxx - origin_shift) < 1e-2
    assert abs(miny + origin_shift) < 1e-2
    assert abs(maxy - origin_shift) < 1e-2


def test_tile_bounds_3857_children_partition_parent():
    parent = tile_bounds_3857(5, 10, 12)
    tl = tile_bounds_3857(6, 20, 24)
    br = tile_bounds_3857(6, 21, 25)
    assert abs(tl[0] - parent[0]) < 1e-6
    assert abs(tl[3] - parent[3]) < 1e-6
    assert abs(br[2] - parent[2]) < 1e-6
    assert abs(br[1] - parent[1]) < 1e-6


def test_encode_decode_terrain_rgb_roundtrip_within_step():
    elevations = np.array([[0.0, 100.0, -50.3, 8848.0]])
    rgb = encode_terrain_rgb(elevations)
    assert rgb.dtype == np.uint8
    assert rgb.shape == (1, 4, 3)
    decoded = decode_terrain_rgb(rgb)
    assert np.allclose(decoded, elevations, atol=0.05)  # округление до шага 0.1 м


def test_encode_terrain_rgb_zero_elevation_reference_value():
    # 0 м -> value=(0-MIN_ENCODABLE_M)/0.1=100000 -> R=1,G=134,B=160 (100000 = 1*65536+134*256+160)
    rgb = encode_terrain_rgb(np.array([[0.0]]))
    assert tuple(int(v) for v in rgb[0, 0]) == (1, 134, 160)


def test_encode_terrain_rgb_clips_out_of_range_values():
    rgb = encode_terrain_rgb(np.array([[MIN_ENCODABLE_M - 1000.0, MAX_ENCODABLE_M + 1000.0]]))
    decoded = decode_terrain_rgb(rgb)
    assert abs(decoded[0, 0] - MIN_ENCODABLE_M) < 0.05
    assert abs(decoded[0, 1] - MAX_ENCODABLE_M) < 0.05


PERM_LON, PERM_LAT = 56.0, 58.0


def test_tiles_covering_bbox_includes_center_tile():
    z = 10
    center_tile = lonlat_to_tile(PERM_LON, PERM_LAT, z)
    tiles = tiles_covering_bbox(PERM_LON - 0.1, PERM_LAT - 0.1, PERM_LON + 0.1, PERM_LAT + 0.1, z)
    assert center_tile in tiles
    assert len(tiles) > 1  # bbox шире одного тайла на z=10


def test_tiles_covering_bbox_single_point_returns_one_tile():
    tiles = tiles_covering_bbox(PERM_LON, PERM_LAT, PERM_LON + 0.0001, PERM_LAT + 0.0001, z=8)
    assert len(tiles) == 1


def _flat_source_grid(elevation: float, half_extent_deg: float = 1.0) -> tuple[np.ndarray, Grid]:
    size = 50
    step = 2 * half_extent_deg / size
    transform = Affine(step, 0.0, PERM_LON - half_extent_deg, 0.0, -step, PERM_LAT + half_extent_deg)
    grid = Grid(transform=transform, width=size, height=size, crs="EPSG:4326")
    values = np.full((size, size), elevation, dtype="float64")
    return values, grid


def test_render_terrain_rgb_tile_is_valid_png_with_correct_size():
    values, grid = _flat_source_grid(150.0)
    tile_x, tile_y = lonlat_to_tile(PERM_LON, PERM_LAT, z=10)  # тайл, реально покрывающий окрестность Перми
    tile_png = render_terrain_rgb_tile(values, grid, z=10, x=tile_x, y=tile_y, tile_size=256)
    assert tile_png[:8] == b"\x89PNG\r\n\x1a\n"

    with rasterio.open(io.BytesIO(tile_png)) as src:
        assert src.width == 256
        assert src.height == 256
        assert src.count == 3
        data = src.read()
    rgb = np.moveaxis(data, 0, -1)
    decoded = decode_terrain_rgb(rgb)
    assert np.allclose(decoded, 150.0, atol=1.0)  # билинейная репроекция плоского поля - почти без искажений


def test_render_terrain_rgb_tile_fills_nodata_with_default():
    values, grid = _flat_source_grid(100.0)
    # Тайл далеко от источника (нет покрытия) - должен быть заполнен nodata_elevation, не падать
    tile_png = render_terrain_rgb_tile(values, grid, z=10, x=0, y=0, tile_size=64, nodata_elevation=0.0)
    with rasterio.open(io.BytesIO(tile_png)) as src:
        data = src.read()
    rgb = np.moveaxis(data, 0, -1)
    decoded = decode_terrain_rgb(rgb)
    assert np.allclose(decoded, 0.0, atol=0.05)
