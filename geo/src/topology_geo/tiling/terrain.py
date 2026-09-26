"""Рельеф тайла со швом без щели с соседями (Шаг 2.1, п. 4).

Приём стыковки — тот же, что и «врезка» в `relief.tin` (Шаг 1.5): граничные
точки тайла сэмплируются на ФИКСИРОВАННОЙ, привязанной к мировым координатам
сетке (`boundary_step_m`), а не к произвольному шагу фоновой сетки. Два
соседних тайла делят общее ребро квадрата (одинаковые мировые координаты по
построению — `grid.TileIndex.bounds`), поэтому при сэмплировании одного и
того же растра `relief_values`/`relief_grid` по одним и тем же точкам оба
тайла получают ПОБИТОВО одинаковые (x, y, z) вдоль шва — треугольники по
разные стороны границы опираются на одни и те же вершины, щели нет.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from topology_geo.relief.service import Grid
from topology_geo.relief.tin import sample_bilinear
from topology_geo.tiling.grid import TILE_SIZE_M, TileIndex

BACKGROUND_STEP_M = 50.0
BOUNDARY_STEP_M = 25.0


def _round_key(x: float, y: float, precision: int = 6) -> tuple[float, float]:
    return round(x, precision), round(y, precision)


def tile_boundary_points(
    tile: TileIndex, tile_size_m: float = TILE_SIZE_M, boundary_step_m: float = BOUNDARY_STEP_M
) -> list[tuple[float, float]]:
    """Точки на контуре тайла (без высоты) — мировые координаты, кратные
    `boundary_step_m` от границ тайла, поэтому у горизонтального/вертикального
    соседа они совпадают день в день (то же общее ребро, тот же шаг)."""
    if tile_size_m % boundary_step_m != 0:
        raise ValueError(f"boundary_step_m={boundary_step_m} должен делить tile_size_m={tile_size_m} нацело")
    minx, miny, maxx, maxy = tile.bounds(tile_size_m)
    n = round(tile_size_m / boundary_step_m)
    edge = np.linspace(0.0, 1.0, n + 1)

    points: dict[tuple[float, float], tuple[float, float]] = {}
    for t in edge:
        x = minx + t * (maxx - minx)
        for y in (miny, maxy):
            points[_round_key(x, y)] = (x, y)
    for t in edge:
        y = miny + t * (maxy - miny)
        for x in (minx, maxx):
            points[_round_key(x, y)] = (x, y)
    return list(points.values())


def _background_points(
    tile: TileIndex, tile_size_m: float, background_step_m: float
) -> list[tuple[float, float]]:
    minx, miny, maxx, maxy = tile.bounds(tile_size_m)
    xs = np.arange(minx + background_step_m, maxx, background_step_m)
    ys = np.arange(miny + background_step_m, maxy, background_step_m)
    return [(float(x), float(y)) for x in xs for y in ys]


def build_tile_terrain(
    relief_values: np.ndarray,
    relief_grid: Grid,
    tile: TileIndex,
    *,
    tile_size_m: float = TILE_SIZE_M,
    boundary_step_m: float = BOUNDARY_STEP_M,
    background_step_m: float = BACKGROUND_STEP_M,
):
    """TIN рельефа одного тайла (Шаг 2.1, п. 4). `relief_values`/`relief_grid` —
    растр в тех же мировых координатах МСК-59, что и `tile` (тот же приём,
    что `relief.tin.build_site_tin`, п. 1 Шага 1.5)."""
    from topology_geo.relief.tin import SiteTin

    boundary = tile_boundary_points(tile, tile_size_m, boundary_step_m)
    background = _background_points(tile, tile_size_m, background_step_m)

    seen: dict[tuple[float, float], tuple[float, float]] = {}
    for x, y in boundary + background:
        seen.setdefault(_round_key(x, y), (x, y))

    rows = []
    for x, y in seen.values():
        z = sample_bilinear(relief_values, relief_grid, x, y)
        if z is not None:
            rows.append((x, y, z))

    if len(rows) < 3:
        raise ValueError(f"недостаточно точек с покрытием DEM для тайла {tile} ({len(rows)})")

    vertices = np.array(rows, dtype=np.float64)
    delaunay = Delaunay(vertices[:, :2])
    return SiteTin(vertices=vertices, triangles=delaunay.simplices, _delaunay=delaunay)
