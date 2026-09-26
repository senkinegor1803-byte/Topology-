"""Тесты стыковки рельефа тайлов (Шаг 2.1, п. 4) — критерий плана «щелей на
стыках нет» проверен не на глаз, а точным совпадением вершин шва."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from topology_geo.relief.service import Grid
from topology_geo.tiling.grid import TileIndex
from topology_geo.tiling.terrain import build_tile_terrain, tile_boundary_points


def _plane_grid(minx=-20.0, miny=-20.0, maxx=520.0, maxy=520.0, pixel=5.0, slope_x=0.01, slope_y=0.02, base=100.0):
    width = int((maxx - minx) / pixel)
    height = int((maxy - miny) / pixel)
    transform = Affine(pixel, 0.0, minx, 0.0, -pixel, maxy)
    grid = Grid(transform=transform, width=width, height=height, crs="")

    rows_idx, cols_idx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    xs = transform.c + (cols_idx + 0.5) * transform.a
    ys = transform.f + (rows_idx + 0.5) * transform.e
    values = base + slope_x * xs + slope_y * ys
    return values, grid


def _exact_plane(x, y, slope_x=0.01, slope_y=0.02, base=100.0):
    return base + slope_x * x + slope_y * y


def _edge_map(vertices, x_value, tol=1e-9):
    return {(round(x, 6), round(y, 6)): z for x, y, z in vertices if abs(x - x_value) < tol}


def test_tile_boundary_points_match_between_horizontal_neighbors():
    tile_a = TileIndex(zone=2, tx=0, ty=0)
    tile_b = TileIndex(zone=2, tx=1, ty=0)
    points_a = {(round(x, 6), round(y, 6)) for x, y in tile_boundary_points(tile_a) if abs(x - 250.0) < 1e-9}
    points_b = {(round(x, 6), round(y, 6)) for x, y in tile_boundary_points(tile_b) if abs(x - 250.0) < 1e-9}
    assert points_a == points_b
    assert len(points_a) > 0


def test_tile_boundary_points_match_between_vertical_neighbors():
    tile_a = TileIndex(zone=2, tx=0, ty=0)
    tile_c = TileIndex(zone=2, tx=0, ty=1)
    points_a = {(round(x, 6), round(y, 6)) for x, y in tile_boundary_points(tile_a) if abs(y - 250.0) < 1e-9}
    points_c = {(round(x, 6), round(y, 6)) for x, y in tile_boundary_points(tile_c) if abs(y - 250.0) < 1e-9}
    assert points_a == points_c
    assert len(points_a) > 0


def test_tile_boundary_points_rejects_non_dividing_step():
    tile = TileIndex(zone=2, tx=0, ty=0)
    with pytest.raises(ValueError):
        tile_boundary_points(tile, tile_size_m=250.0, boundary_step_m=60.0)


def test_adjacent_tiles_share_identical_terrain_along_horizontal_seam():
    values, grid = _plane_grid()
    tile_a = TileIndex(zone=2, tx=0, ty=0)
    tile_b = TileIndex(zone=2, tx=1, ty=0)

    tin_a = build_tile_terrain(values, grid, tile_a)
    tin_b = build_tile_terrain(values, grid, tile_b)

    edge_a = _edge_map(tin_a.vertices, 250.0)
    edge_b = _edge_map(tin_b.vertices, 250.0)
    assert edge_a  # не пусто
    assert set(edge_a) == set(edge_b)
    for key, z in edge_a.items():
        assert z == edge_b[key]  # побитовое совпадение, не "почти равно"


def test_adjacent_tiles_share_identical_terrain_along_vertical_seam():
    values, grid = _plane_grid()
    tile_a = TileIndex(zone=2, tx=0, ty=0)
    tile_c = TileIndex(zone=2, tx=0, ty=1)

    tin_a = build_tile_terrain(values, grid, tile_a)
    tin_c = build_tile_terrain(values, grid, tile_c)

    def edge_map_y(vertices, y_value):
        return {(round(x, 6), round(y, 6)): z for x, y, z in vertices if abs(y - y_value) < 1e-9}

    edge_a = edge_map_y(tin_a.vertices, 250.0)
    edge_c = edge_map_y(tin_c.vertices, 250.0)
    assert edge_a
    assert set(edge_a) == set(edge_c)
    for key, z in edge_a.items():
        assert z == edge_c[key]


def test_tile_terrain_reproduces_exact_plane():
    values, grid = _plane_grid()
    tile = TileIndex(zone=2, tx=0, ty=0)
    tin = build_tile_terrain(values, grid, tile)
    for x, y, z in tin.vertices:
        assert z == pytest.approx(_exact_plane(x, y), abs=1e-9)


def test_tile_terrain_interpolation_has_no_gap_across_seam():
    """Не только вершины совпадают - интерполяция по обе стороны шва на самой
    границе тоже даёт одно и то же значение (что и требует критерий «щелей
    на стыках нет», а не только «вершины по счастливой случайности совпали»)."""
    values, grid = _plane_grid()
    tile_a = TileIndex(zone=2, tx=0, ty=0)
    tile_b = TileIndex(zone=2, tx=1, ty=0)
    tin_a = build_tile_terrain(values, grid, tile_a)
    tin_b = build_tile_terrain(values, grid, tile_b)

    for y in (10.0, 77.5, 130.0, 249.9):
        z_a = tin_a.interpolate_z(250.0 - 1e-6, y)
        z_b = tin_b.interpolate_z(250.0 + 1e-6, y)
        assert z_a is not None and z_b is not None
        assert z_a == pytest.approx(z_b, abs=1e-6)
        assert z_a == pytest.approx(_exact_plane(250.0, y), abs=1e-6)
