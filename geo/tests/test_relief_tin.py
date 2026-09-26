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
    sample_bilinear_grid,
    smooth_grid_elevations,
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


def test_build_site_tin_rejects_radius_beyond_untiled_memory_limit():
    """Предохранитель от OOM (докстринг модуля): нетайловый `Delaunay` на шаге
    1 м не тянет большой радиус (реальный прогон на ~7 млн точек уходил за
    12 ГБ) — функция обязана отказать СРАЗУ понятной ошибкой, а не зависнуть
    или упасть по памяти где-то в середине. Проверяем именно это — что отказ
    происходит ДО построения гигантской сетки (`background_step_m` не тронут,
    значит, при радиусе 3 км с шагом 1 м она обязана отказать быстро)."""
    values, grid = _make_planar_grid()
    with pytest.raises(ValueError, match="выше безопасного предела"):
        build_site_tin(values, grid, CENTER_X, CENTER_Y, radius_m=3000.0, features=[])


def test_background_grid_default_step_is_one_metre():
    """«Метод квадратных призм»: шаг фоновой сетки по умолчанию — 1 м, не
    больше (иначе крупные треугольники фона заметны как артефакты при плоском
    затенении в вебвьюере, Шаг 1.9)."""
    values, grid = _make_planar_grid()
    tin = build_site_tin(values, grid, CENTER_X, CENTER_Y, radius_m=20.0, features=[])

    xs = np.unique(np.round(tin.vertices[:, 0], 6))
    xs.sort()
    steps = np.diff(xs)
    assert steps.max() == pytest.approx(1.0, abs=1e-6)
    assert steps.min() == pytest.approx(1.0, abs=1e-6)


def test_sample_bilinear_grid_matches_scalar_sample_bilinear():
    """Векторизованная выборка (используется для фоновой сетки — миллионы
    точек, поточечный `sample_bilinear` был бы на порядки медленнее) должна
    давать те же значения, что и поточечная эталонная функция."""
    values, grid = _make_planar_grid()
    world_x = np.array([CENTER_X, CENTER_X + 37.3, CENTER_X - 150.0, CENTER_X - 10_000.0])
    world_y = np.array([CENTER_Y, CENTER_Y - 88.1, CENTER_Y + 150.0, CENTER_Y])

    vectorized = sample_bilinear_grid(values, grid, world_x, world_y)

    for i in range(len(world_x)):
        scalar = sample_bilinear(values, grid, float(world_x[i]), float(world_y[i]))
        if scalar is None:
            assert np.isnan(vectorized[i])
        else:
            assert vectorized[i] == pytest.approx(scalar, abs=1e-9)


def test_smooth_grid_elevations_preserves_plane_away_from_edges():
    """Скользящее среднее линейной функции по симметричному окну равно её
    значению в центре — сглаживание фона не должно искажать наклон рельефа,
    только убирать локальный шум (докстринг `smooth_grid_elevations`)."""
    xs, ys = np.meshgrid(np.arange(50.0), np.arange(50.0), indexing="ij")
    plane = PLANE_A * xs + PLANE_B * ys + PLANE_C

    smoothed = smooth_grid_elevations(plane, window_cells=7)

    interior = smoothed[10:-10, 10:-10]
    expected = plane[10:-10, 10:-10]
    assert np.allclose(interior, expected, atol=1e-9)


def test_multi_structural_smoothing_reduces_background_spike_but_keeps_flat_pads():
    """«Многоструктурное сглаживание»: фон сглаживается 2D-скользящим средним
    (одиночный всплеск растра размывается по соседям), а плоская площадка
    здания остаётся ТОЧНОЙ — она не проходит через `smooth_grid_elevations`
    вовсе (докстринг модуля `relief.tin`). Здание намеренно далеко от
    всплеска — иначе всплеск попал бы в вырезанную зону здания и тест
    проверял бы не то, что заявлено."""
    half = 60.0
    pixel = 1.0
    size = int(2 * half / pixel)
    cx, cy = 500.0, 500.0
    transform = Affine(pixel, 0.0, cx - half, 0.0, -pixel, cy + half)
    grid = Grid(transform=transform, width=size, height=size, crs="EPSG:3857")

    values = np.full((size, size), 100.0)
    spike_row, spike_col = size // 2 + 20, size // 2 + 20  # вдали от здания у центра
    values[spike_row, spike_col] = 150.0

    spike_local_x = transform.a * (spike_col + 0.5) + transform.c - cx
    spike_local_y = transform.e * (spike_row + 0.5) + transform.f - cy

    building = SiteFeature(
        layer="osm_buildings", osm_id=1, osm_type="W",
        geometry=Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)]),
        attributes={}, confidence={},
    )
    tin = build_site_tin(values, grid, cx, cy, radius_m=50.0, features=[building])

    vertices = tin.vertices
    dist_to_spike = np.hypot(vertices[:, 0] - spike_local_x, vertices[:, 1] - spike_local_y)
    nearest_idx = np.argmin(dist_to_spike)
    assert dist_to_spike[nearest_idx] < 1.0  # фоновая сетка на шаге 1 м покрывает окрестность всплеска

    spike_z = vertices[nearest_idx, 2]
    assert 100.0 < spike_z < 150.0  # сглажено между фоном и сырым всплеском, не равно ни тому ни другому

    far_mask = dist_to_spike > 10.0  # вне окна сглаживания (радиус 2 м по умолчанию)
    assert np.allclose(vertices[far_mask, 2], 100.0, atol=1e-6)

    building_corner_z = tin.interpolate_z(-5.0, -5.0)
    assert building_corner_z == pytest.approx(100.0, abs=1e-9)  # площадка не сглажена — точная
