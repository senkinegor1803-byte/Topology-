"""Тесты TIN участка (Шаг 1.5). Синтетический плоский наклонный рельеф
(z = a*x + b*y + c) даёт точную проверку: линейная интерполяция Делоне
воспроизводит плоскость без погрешности, поэтому любое отклонение сразу
видно как реальная ошибка, а не шум округления.

`test_road_stays_within_tolerance_after_edge_padding` — регрессия на
найденный при разработке краевой эффект скользящего среднего (несимметричное
окно на концах линии смещало сглаженную отметку дороги пропорционально
уклону); `test_sample_bilinear_matches_pixel_center_grid` — регрессия на
найденный тогда же баг смещения на полпикселя в `sample_bilinear`.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString, Polygon

from topology_geo.relief.service import Grid
from topology_geo.relief.tin import (
    build_site_tin,
    find_degenerate_triangles,
    max_deviation_along_line,
    sample_bilinear,
)
from topology_geo.selection.service import SiteFeature

CENTER_X, CENTER_Y = 1000.0, 2000.0
RADIUS = 200.0
PIXEL = 5.0
PLANE_A, PLANE_B, PLANE_C = 0.01, 0.02, 100.0


def _true_z(local_x: float, local_y: float) -> float:
    return PLANE_A * (CENTER_X + local_x) + PLANE_B * (CENTER_Y + local_y) + PLANE_C


def _make_planar_grid() -> tuple[np.ndarray, Grid]:
    half = RADIUS + 50
    size = int(2 * half / PIXEL)
    transform = Affine(PIXEL, 0.0, CENTER_X - half, 0.0, -PIXEL, CENTER_Y + half)
    grid = Grid(transform=transform, width=size, height=size, crs="EPSG:3857")

    rows, cols = np.indices((size, size))
    xs = transform.a * (cols + 0.5) + transform.c
    ys = transform.e * (rows + 0.5) + transform.f
    values = PLANE_A * xs + PLANE_B * ys + PLANE_C
    return values, grid


def test_sample_bilinear_matches_pixel_center_grid():
    values, grid = _make_planar_grid()
    for wx, wy in [(CENTER_X, CENTER_Y), (CENTER_X + 37.3, CENTER_Y - 88.1), (CENTER_X - 150, CENTER_Y + 150)]:
        sampled = sample_bilinear(values, grid, wx, wy)
        expected = PLANE_A * wx + PLANE_B * wy + PLANE_C
        assert sampled == pytest.approx(expected, abs=1e-9)


def test_sample_bilinear_returns_none_outside_raster():
    values, grid = _make_planar_grid()
    assert sample_bilinear(values, grid, CENTER_X - 10_000, CENTER_Y) is None


@pytest.fixture()
def planar_site_features():
    road = SiteFeature(
        layer="osm_roads", osm_id=1, osm_type="W",
        geometry=LineString([(-150, 0), (150, 0)]),
        attributes={"lanes": 2}, confidence={},
    )
    building = SiteFeature(
        layer="osm_buildings", osm_id=2, osm_type="W",
        geometry=Polygon([(50, 50), (70, 50), (70, 70), (50, 70)]),
        attributes={}, confidence={},
    )
    water = SiteFeature(
        layer="osm_water_areas", osm_id=3, osm_type="W",
        geometry=Polygon([(-100, -100), (-80, -100), (-80, -80), (-100, -80)]),
        attributes={}, confidence={},
    )
    return [road, building, water]


def test_tin_has_no_degenerate_triangles(planar_site_features):
    values, grid = _make_planar_grid()
    tin = build_site_tin(values, grid, CENTER_X, CENTER_Y, RADIUS, planar_site_features)
    assert find_degenerate_triangles(tin) == []


def test_tin_reproduces_plane_on_background_terrain(planar_site_features):
    values, grid = _make_planar_grid()
    tin = build_site_tin(values, grid, CENTER_X, CENTER_Y, RADIUS, planar_site_features)

    for point in [(120.0, -120.0), (0.0, 150.0), (30.0, 30.0), (-150.0, 100.0)]:
        interp = tin.interpolate_z(*point)
        assert interp is not None
        assert interp == pytest.approx(_true_z(*point), abs=1e-6)


def test_road_stays_within_tolerance_after_edge_padding(planar_site_features):
    values, grid = _make_planar_grid()
    tin = build_site_tin(values, grid, CENTER_X, CENTER_Y, RADIUS, planar_site_features)
    road = planar_site_features[0]

    deviation = max_deviation_along_line(tin, road.geometry, lambda x, y: _true_z(x, y))
    assert deviation <= 0.1  # критерий приёмки Шага 1.5
    assert deviation < 1e-6  # на идеальной плоскости отклонения не должно быть вовсе


def test_building_pad_is_flat_at_minimum_corner_elevation(planar_site_features):
    values, grid = _make_planar_grid()
    tin = build_site_tin(values, grid, CENTER_X, CENTER_Y, RADIUS, planar_site_features)
    building = planar_site_features[1]

    corner_levels = [_true_z(x, y) for x, y in building.geometry.exterior.coords]
    expected_pad = min(corner_levels)

    center = building.geometry.centroid
    assert tin.interpolate_z(center.x, center.y) == pytest.approx(expected_pad, abs=1e-6)


def test_water_is_flat_at_minimum_boundary_elevation(planar_site_features):
    values, grid = _make_planar_grid()
    tin = build_site_tin(values, grid, CENTER_X, CENTER_Y, RADIUS, planar_site_features)
    water = planar_site_features[2]

    boundary_levels = [_true_z(x, y) for x, y in water.geometry.exterior.coords]
    expected_level = min(boundary_levels)

    center = water.geometry.centroid
    assert tin.interpolate_z(center.x, center.y) == pytest.approx(expected_level, abs=1e-6)


def test_build_site_tin_raises_with_too_few_points():
    values, grid = _make_planar_grid()
    with pytest.raises(ValueError):
        build_site_tin(values, grid, CENTER_X, CENTER_Y, radius_m=1.0, features=[], background_step_m=1000.0)
