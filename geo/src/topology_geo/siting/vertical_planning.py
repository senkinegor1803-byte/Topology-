"""Отметка 0.000 и вертикальная планировка участка (Шаг 3.7).

Переиспользует инфраструктуру рельефа Шага 1.2 (`relief.tin.
sample_bilinear`, `relief.merge.merge_with_transition`), а не строит
параллельную — вертикальная планировка это ТА ЖЕ задача «слить новую
поверхность со старой с плавным переходом», что и слияние источников
рельефа, просто источник теперь не другой DEM, а проектная плоская
площадка.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio.features import rasterize
from shapely.geometry.base import BaseGeometry

from topology_geo.relief.merge import merge_with_transition
from topology_geo.relief.service import Grid
from topology_geo.relief.tin import sample_bilinear

# Уклон откоса выемки/насыпи для нескальных грунтов (СП 45.13330.2017
# "Земляные сооружения, основания и фундаменты", типовое представительное
# значение) - заложение 1,5 м на 1 м высоты.
DEFAULT_SLOPE_RATIO = 1.5
# Порог доступного места для устройства откоса без подпорной стенки -
# сверх него для устойчивости борта нужна подпорная стенка (иначе откос
# "не влезает" в отступ от границы участка/соседних объектов).
MAX_SLOPE_RUN_WITHOUT_WALL_M = 5.0


def design_elevation_from_entrances(
    entrance_points_xy: list[tuple[float, float]], relief_values: np.ndarray, grid: Grid,
) -> float:
    """Действие п. 1, вариант «расчётом по рельефу у входов»: 0.000 —
    среднее фактического рельефа в точках входов (не минимум/максимум —
    типовая практика при нескольких входах на разных высотах, среднее
    минимизирует суммарный перепад со всеми входами сразу)."""
    if not entrance_points_xy:
        raise ValueError("нужна хотя бы одна точка входа")
    elevations = []
    for x, y in entrance_points_xy:
        z = sample_bilinear(relief_values, grid, x, y)
        if z is None:
            raise ValueError(f"точка входа ({x}, {y}) вне покрытия рельефа")
        elevations.append(z)
    return sum(elevations) / len(elevations)


def _rasterize_footprint(footprint: BaseGeometry, grid: Grid) -> np.ndarray:
    return rasterize(
        [(footprint, 1)], out_shape=(grid.height, grid.width), transform=grid.transform,
        fill=0, dtype="uint8",
    ).astype(bool)


@dataclass(frozen=True)
class CutFillVolumes:
    cut_m3: float  # выемка - где фактический рельеф выше проектной отметки
    fill_m3: float  # насыпь - где фактический рельеф ниже проектной отметки


def compute_cut_fill_volumes(
    footprint: BaseGeometry, design_elevation_m: float, relief_values: np.ndarray, grid: Grid,
) -> CutFillVolumes:
    """Действие п. 3: объёмы выемки/насыпи под пятном — численное
    интегрирование по пикселям сетки рельефа (та же сетка, что уже несёт
    остальной проект, Шаг 1.2), не отдельная более грубая дискретизация."""
    mask = _rasterize_footprint(footprint, grid)
    pixel_area_m2 = grid.pixel_size_m ** 2

    diff = design_elevation_m - relief_values  # >0 - нужна насыпь, <0 - нужна выемка
    masked_diff = np.where(mask, diff, 0.0)

    fill_m3 = float(np.sum(np.clip(masked_diff, 0.0, None)) * pixel_area_m2)
    cut_m3 = float(np.sum(np.clip(-masked_diff, 0.0, None)) * pixel_area_m2)
    return CutFillVolumes(cut_m3=cut_m3, fill_m3=fill_m3)


def check_retaining_wall_needed(
    footprint: BaseGeometry, design_elevation_m: float, relief_values: np.ndarray, grid: Grid,
    *, apron_width_m: float = 1.0, slope_ratio: float = DEFAULT_SLOPE_RATIO, sample_points: int = 16,
) -> tuple[bool, float]:
    """Действие п. 2 («откосы с заданным уклоном, подпорные стенки при
    превышении перепада»): по кольцу отмостки вокруг пятна сэмплируется
    перепад высоты факт/проект; если для наибольшего перепада требуемое
    заложение откоса (`|перепад| * slope_ratio`) превышает разумный запас
    места (`MAX_SLOPE_RUN_WITHOUT_WALL_M`) — нужна подпорная стенка.
    Возвращает `(нужна_стенка, макс_перепад_м)`."""
    boundary = footprint.buffer(apron_width_m).exterior
    max_diff = 0.0
    needs_wall = False
    for i in range(sample_points):
        pt = boundary.interpolate(i / sample_points, normalized=True)
        ground_z = sample_bilinear(relief_values, grid, pt.x, pt.y)
        if ground_z is None:
            continue
        diff = abs(design_elevation_m - ground_z)
        max_diff = max(max_diff, diff)
        if diff * slope_ratio > MAX_SLOPE_RUN_WITHOUT_WALL_M:
            needs_wall = True
    return needs_wall, max_diff


def update_relief_in_site_zone(
    relief_values: np.ndarray, grid: Grid, footprint: BaseGeometry, design_elevation_m: float,
    *, apron_width_m: float = 1.0, transition_width_m: float | None = None, slope_ratio: float = DEFAULT_SLOPE_RATIO,
) -> np.ndarray:
    """Действие п. 4: «обновить рельеф модели в зоне участка» —
    переиспользует `relief.merge.merge_with_transition` (Шаг 1.2, п. 3):
    площадка (пятно + отмостка) на проектной отметке — это `overlay`,
    существующий рельеф — `base`, ширина перехода по умолчанию — заложение
    откоса при максимальном перепаде вокруг площадки (реальный откос, не
    произвольное число)."""
    pad = footprint.buffer(apron_width_m)
    mask = _rasterize_footprint(pad, grid)

    if transition_width_m is None:
        _needs_wall, max_diff = check_retaining_wall_needed(
            footprint, design_elevation_m, relief_values, grid, apron_width_m=apron_width_m, slope_ratio=slope_ratio,
        )
        transition_width_m = max(max_diff * slope_ratio, 1.0)

    overlay = np.full_like(relief_values, design_elevation_m)
    return merge_with_transition(
        relief_values, overlay, mask, pixel_size_m=grid.pixel_size_m, transition_width_m=transition_width_m,
    )
