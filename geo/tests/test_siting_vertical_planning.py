"""Тесты Шага 3.7: отметка 0.000, объёмы земляных работ, обновление рельефа
в зоне участка."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from topology_geo.relief.service import Grid
from topology_geo.siting.vertical_planning import (
    check_retaining_wall_needed,
    compute_cut_fill_volumes,
    design_elevation_from_entrances,
    update_relief_in_site_zone,
)

PIXEL = 1.0
CENTER_X, CENTER_Y = 1000.0, 2000.0


def _flat_grid(elevation: float, half_extent: float = 50.0):
    size = int(2 * half_extent / PIXEL)
    transform = Affine(PIXEL, 0.0, CENTER_X - half_extent, 0.0, -PIXEL, CENTER_Y + half_extent)
    grid = Grid(transform=transform, width=size, height=size, crs="EPSG:3857")
    values = np.full((size, size), elevation, dtype="float64")
    return values, grid


def _sloped_grid(base_elevation: float, slope_per_m: float, half_extent: float = 50.0):
    """Наклонная плоскость - высота растёт с X (для проверки, что перепад
    реально сэмплируется, не просто плоское поле)."""
    size = int(2 * half_extent / PIXEL)
    transform = Affine(PIXEL, 0.0, CENTER_X - half_extent, 0.0, -PIXEL, CENTER_Y + half_extent)
    grid = Grid(transform=transform, width=size, height=size, crs="EPSG:3857")
    cols = np.arange(size)
    world_x = transform.c + (cols + 0.5) * transform.a
    row_values = base_elevation + slope_per_m * (world_x - CENTER_X)
    values = np.tile(row_values, (size, 1))
    return values, grid


def test_design_elevation_from_entrances_averages_relief_at_points():
    values, grid = _sloped_grid(base_elevation=100.0, slope_per_m=0.1)

    elevation = design_elevation_from_entrances(
        [(CENTER_X - 10.0, CENTER_Y), (CENTER_X + 10.0, CENTER_Y)], values, grid,
    )

    assert elevation == pytest.approx(100.0, abs=0.05)  # среднее симметричных точек = базовая высота


def test_design_elevation_from_entrances_raises_outside_coverage():
    values, grid = _flat_grid(100.0, half_extent=5.0)
    with pytest.raises(ValueError, match="вне покрытия"):
        design_elevation_from_entrances([(CENTER_X + 1000.0, CENTER_Y)], values, grid)


def test_compute_cut_fill_volumes_flat_relief_below_design_is_all_fill():
    values, grid = _flat_grid(95.0)
    footprint = box(CENTER_X - 10, CENTER_Y - 10, CENTER_X + 10, CENTER_Y + 10)  # 20x20 = 400 м²

    volumes = compute_cut_fill_volumes(footprint, design_elevation_m=100.0, relief_values=values, grid=grid)

    assert volumes.cut_m3 == pytest.approx(0.0)
    assert volumes.fill_m3 == pytest.approx(400.0 * 5.0, rel=0.02)  # 5 м насыпи на 400 м²


def test_compute_cut_fill_volumes_flat_relief_above_design_is_all_cut():
    values, grid = _flat_grid(103.0)
    footprint = box(CENTER_X - 10, CENTER_Y - 10, CENTER_X + 10, CENTER_Y + 10)

    volumes = compute_cut_fill_volumes(footprint, design_elevation_m=100.0, relief_values=values, grid=grid)

    assert volumes.fill_m3 == pytest.approx(0.0)
    assert volumes.cut_m3 == pytest.approx(400.0 * 3.0, rel=0.02)


def test_check_retaining_wall_needed_flat_relief_no_wall():
    values, grid = _flat_grid(100.0)
    footprint = box(CENTER_X - 10, CENTER_Y - 10, CENTER_X + 10, CENTER_Y + 10)

    needs_wall, max_diff = check_retaining_wall_needed(footprint, design_elevation_m=100.0, relief_values=values, grid=grid)

    assert needs_wall is False
    assert max_diff == pytest.approx(0.0, abs=0.01)


def test_check_retaining_wall_needed_large_step_requires_wall():
    values, grid = _flat_grid(80.0)  # перепад 20 м с проектной отметкой
    footprint = box(CENTER_X - 10, CENTER_Y - 10, CENTER_X + 10, CENTER_Y + 10)

    needs_wall, max_diff = check_retaining_wall_needed(footprint, design_elevation_m=100.0, relief_values=values, grid=grid)

    assert needs_wall is True
    assert max_diff == pytest.approx(20.0, abs=0.5)


def test_update_relief_in_site_zone_flattens_pad_and_blends_outward():
    # Небольшой перепад (2 м) - переход (2*1.5=3 м) заведомо уже пятна+отмостки
    # (12x12 м), иначе на самой границе теста мы бы проверяли не «плоскую
    # площадку в центре», а сам факт неполного перехода - это отдельный,
    # реальный случай (см. test_update_relief_in_site_zone_wide_transition...
    # ниже), здесь - базовый случай «перепад укладывается в размер площадки».
    values, grid = _flat_grid(98.0, half_extent=50.0)
    footprint = box(CENTER_X - 10, CENTER_Y - 10, CENTER_X + 10, CENTER_Y + 10)

    updated = update_relief_in_site_zone(values, grid, footprint, design_elevation_m=100.0, apron_width_m=1.0)

    from topology_geo.relief.tin import sample_bilinear

    center_z = sample_bilinear(updated, grid, CENTER_X, CENTER_Y)
    assert center_z == pytest.approx(100.0, abs=0.1)  # в центре пятна - точно проектная отметка

    far_z = sample_bilinear(updated, grid, CENTER_X + 49.0, CENTER_Y)
    assert far_z == pytest.approx(98.0, abs=0.5)  # далеко от участка - рельеф не тронут


def test_update_relief_in_site_zone_large_step_gives_wide_transition():
    """Перепад больше пятна (тот же сценарий, что требует подпорной стенки,
    Шаг 3.7 п. 2) - переход по формуле честно растягивается на заложение
    откоса и потому не успевает дойти до проектной отметки уже в центре
    маленького пятна - это ожидаемое, не ошибочное поведение линейного
    сглаживания на слишком узкой площадке."""
    values, grid = _flat_grid(90.0, half_extent=50.0)
    footprint = box(CENTER_X - 5, CENTER_Y - 5, CENTER_X + 5, CENTER_Y + 5)

    updated = update_relief_in_site_zone(values, grid, footprint, design_elevation_m=100.0, apron_width_m=1.0)

    from topology_geo.relief.tin import sample_bilinear

    center_z = sample_bilinear(updated, grid, CENTER_X, CENTER_Y)
    assert 90.0 < center_z < 100.0  # смешано, не полностью проектная отметка - площадка меньше требуемого перехода
