"""Рельеф тайла со швом без щели с соседями (Шаг 2.1, п. 4).

Приём стыковки — тот же, что и «врезка» в `relief.tin` (Шаг 1.5): граничные
точки тайла сэмплируются на ФИКСИРОВАННОЙ, привязанной к мировым координатам
сетке (`boundary_step_m`), а не к произвольному шагу фоновой сетки. Два
соседних тайла делят общее ребро квадрата (одинаковые мировые координаты по
построению — `grid.TileIndex.bounds`), поэтому при сэмплировании одного и
того же растра `relief_values`/`relief_grid` по одним и тем же точкам оба
тайла получают ПОБИТОВО одинаковые (x, y, z) вдоль шва — треугольники по
разные стороны границы опираются на одни и те же вершины, щели нет.

Фон (не шов) — та же «сетка призм» с шагом 1 м и то же 2D-сглаживание
(`relief.tin.smooth_grid_elevations`), что и в Шаге 1.5 (см. докстринг
`relief.tin` — почему шаг именно 1 м и почему сглаживание не трогает шов):
шов остаётся НЕсглаженным (сэмплируется отдельно, `tile_boundary_points`) —
иначе сглаживание с краевым эффектом «ближайшего» пикселя за пределами
тайла дало бы разные значения у двух соседей в одной мировой точке, и щель
вернулась бы, только уже от сглаживания, а не от шага сетки.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from topology_geo.relief.service import Grid
from topology_geo.relief.tin import sample_bilinear, sample_bilinear_grid, smooth_grid_elevations
from topology_geo.tiling.grid import TILE_SIZE_M, TileIndex

BACKGROUND_STEP_M = 1.0
BOUNDARY_STEP_M = 1.0
BACKGROUND_SMOOTHING_WINDOW_M = 5.0


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


def _smoothed_background_points(
    tile: TileIndex,
    tile_size_m: float,
    background_step_m: float,
    relief_values: np.ndarray,
    relief_grid: Grid,
    *,
    smoothing_window_m: float,
) -> list[tuple[float, float, float]]:
    """Фон тайла (без шва — тот сэмплируется отдельно, `tile_boundary_points`)
    — та же квадратная сетка призм и то же 2D-сглаживание, что и в
    `relief.tin._smoothed_background_arrays` (Шаг 1.5); векторизовано по той
    же причине — на шаге 1 м у тайла 250×250 м это уже ~62 тыс. точек, а на
    полный радиус 3 км (Шаг 2.1) — сотни тайлов, поточечный `sample_bilinear`
    в Python-цикле был бы заметно медленнее numpy-версии.

    Сетка для сглаживания расширена на `pad` шагов за границы тайла и
    обрезана обратно после — иначе точки у края тайла сглаживались бы с
    `mode="nearest"` (повтор крайнего семпла вместо реального соседнего
    пикселя растра за пределами тайла), что даёт систематическое смещение на
    наклонной плоскости (тот же приём и то же обоснование, что в
    `relief.tin._smoothed_background_arrays`). Расширение остаётся строго
    внутри запаса растра (`RELIEF_MARGIN_M`, `jobs/steps.py`) при разумном
    окне сглаживания, соседний тайл он не задевает — на сам шов (побитовое
    совпадение вершин) это не влияет: шов сэмплируется отдельно и
    сглаживанию не подвергается."""
    minx, miny, maxx, maxy = tile.bounds(tile_size_m)
    window_cells = max(round(smoothing_window_m / background_step_m), 1)
    if window_cells % 2 == 0:
        window_cells += 1
    pad = window_cells // 2
    margin = pad * background_step_m

    xs_padded = np.arange(minx + background_step_m - margin, maxx + margin, background_step_m)
    ys_padded = np.arange(miny + background_step_m - margin, maxy + margin, background_step_m)
    if len(xs_padded) <= 2 * pad or len(ys_padded) <= 2 * pad:
        return []
    gx_padded, gy_padded = np.meshgrid(xs_padded, ys_padded, indexing="ij")

    raw = sample_bilinear_grid(relief_values, relief_grid, gx_padded, gy_padded)
    smoothed_padded = smooth_grid_elevations(raw, window_cells)

    hi_x, hi_y = gx_padded.shape[0] - pad, gx_padded.shape[1] - pad
    gx = gx_padded[pad:hi_x, pad:hi_y]
    gy = gy_padded[pad:hi_x, pad:hi_y]
    smoothed = smoothed_padded[pad:hi_x, pad:hi_y]

    valid = ~np.isnan(smoothed)
    return [(float(x), float(y), float(z)) for x, y, z in zip(gx[valid], gy[valid], smoothed[valid])]


def build_tile_terrain(
    relief_values: np.ndarray,
    relief_grid: Grid,
    tile: TileIndex,
    *,
    tile_size_m: float = TILE_SIZE_M,
    boundary_step_m: float = BOUNDARY_STEP_M,
    background_step_m: float = BACKGROUND_STEP_M,
    background_smoothing_window_m: float = BACKGROUND_SMOOTHING_WINDOW_M,
):
    """TIN рельефа одного тайла (Шаг 2.1, п. 4). `relief_values`/`relief_grid` —
    растр в тех же мировых координатах МСК-59, что и `tile` (тот же приём,
    что `relief.tin.build_site_tin`, п. 1 Шага 1.5)."""
    from topology_geo.relief.tin import SiteTin

    boundary_rows = []
    for x, y in tile_boundary_points(tile, tile_size_m, boundary_step_m):
        z = sample_bilinear(relief_values, relief_grid, x, y)
        if z is not None:
            boundary_rows.append((x, y, z))

    background_rows = _smoothed_background_points(
        tile, tile_size_m, background_step_m, relief_values, relief_grid,
        smoothing_window_m=background_smoothing_window_m,
    )

    seen: dict[tuple[float, float], tuple[float, float, float]] = {}
    for x, y, z in boundary_rows + background_rows:  # шов раньше фона — при совпадении координат точный шов побеждает
        seen.setdefault(_round_key(x, y), (x, y, z))

    rows = list(seen.values())
    if len(rows) < 3:
        raise ValueError(f"недостаточно точек с покрытием DEM для тайла {tile} ({len(rows)})")

    vertices = np.array(rows, dtype=np.float64)
    delaunay = Delaunay(vertices[:, :2])
    return SiteTin(vertices=vertices, triangles=delaunay.simplices, _delaunay=delaunay)
