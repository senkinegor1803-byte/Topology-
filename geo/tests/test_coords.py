"""Тесты пересчёта координат (Шаг 0.4).

Контрольные точки здесь синтетические (сгенерированы самой функцией пересчёта
и известной моделью высот) — они проверяют, что код математически корректен и
самосогласован (round-trip, восстановление калибровки МНК). Они НЕ заменяют
проверку на 5-10 реальных контрольных точках (пункты ГГС, топосъёмка) с реальными
официальными параметрами МСК-59, которая по плану требует данных изыскателя.
"""

from __future__ import annotations

import pytest

from shapely.geometry import LineString, Point, Polygon

from topology_geo.coords import (
    MSK59_ZONES,
    ControlPoint,
    fit_height_offset,
    msk59_to_wgs84,
    pick_msk59_zone,
    transform_geometry_to_msk59,
    verify_heights,
    verify_planimetric,
    wgs84_to_msk59,
)

PERM_CENTER_LON = 56.2431
PERM_CENTER_LAT = 58.0105


def test_pick_zone_for_perm_is_zone_2():
    assert pick_msk59_zone(PERM_CENTER_LON) == 2


@pytest.mark.parametrize("zone", sorted(MSK59_ZONES))
def test_round_trip_wgs84_msk59(zone: int):
    lon0 = MSK59_ZONES[zone].lon0
    lon, lat = lon0 + 0.3, 58.0
    x, y, picked_zone = wgs84_to_msk59(lon, lat, zone=zone)
    assert picked_zone == zone

    lon_back, lat_back = msk59_to_wgs84(x, y, zone=zone)
    assert lon_back == pytest.approx(lon, abs=1e-7)
    assert lat_back == pytest.approx(lat, abs=1e-7)


def test_false_easting_lands_near_axial_meridian():
    """На осевом меридиане зоны x должен быть близок к false_easting.

    Точного совпадения нет: 7-параметрический сдвиг towgs84 (перенос+повороты
    Красовский<->WGS-84) сам по себе даёт отклонение порядка первых сотен метров -
    это ожидаемо и не является ошибкой пересчёта, допуск подобран с запасом.
    """
    params = MSK59_ZONES[2]
    x, _, _ = wgs84_to_msk59(params.lon0, 58.0, zone=2)
    assert x == pytest.approx(params.false_easting, abs=200.0)


def test_pick_zone_out_of_range_raises():
    with pytest.raises(ValueError):
        pick_msk59_zone(30.0)


def test_verify_planimetric_self_consistent():
    """Точки, сгенерированные самой функцией, должны давать ~0 расхождение."""
    zone = 2
    points = []
    for i, (dlon, dlat) in enumerate([(0.0, 0.0), (0.01, 0.0), (0.0, 0.01), (-0.01, 0.02)]):
        lon, lat = PERM_CENTER_LON + dlon, PERM_CENTER_LAT + dlat
        x, y, _ = wgs84_to_msk59(lon, lat, zone=zone)
        points.append(ControlPoint(name=f"p{i}", lon=lon, lat=lat, msk_x=x, msk_y=y))

    results = verify_planimetric(points, zone=zone)
    for r in results:
        assert r.error_m <= 0.1, r


def test_fit_and_verify_height_offset_recovers_plane():
    zone = 2
    true_offset, true_gx, true_gy = -14.7, 0.00002, -0.00001
    lon0, lat0 = PERM_CENTER_LON, PERM_CENTER_LAT
    x0, y0, _ = wgs84_to_msk59(lon0, lat0, zone=zone)

    points = []
    for i, (dlon, dlat) in enumerate(
        [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01), (-0.01, 0.01), (0.02, -0.01)]
    ):
        lon, lat = lon0 + dlon, lat0 + dlat
        x, y, _ = wgs84_to_msk59(lon, lat, zone=zone)
        h_source = 120.0 + i  # произвольная высота EGM96
        h_target = h_source + true_offset + true_gx * (x - x0) + true_gy * (y - y0)
        points.append(
            ControlPoint(name=f"cp{i}", lon=lon, lat=lat, h_source=h_source, h_target=h_target)
        )

    model = fit_height_offset(points, zone=zone)
    results = verify_heights(points, model, zone=zone)
    assert results, "верификация должна вернуть результаты по всем точкам с известной высотой"
    for r in results:
        assert r.error_m <= 0.05, r


def test_fit_height_offset_with_single_point_uses_constant():
    zone = 2
    point = ControlPoint(
        name="single", lon=PERM_CENTER_LON, lat=PERM_CENTER_LAT, h_source=100.0, h_target=85.3
    )
    model = fit_height_offset([point], zone=zone)
    assert model.offset == pytest.approx(-14.7)
    assert model.grad_x == 0.0
    assert model.grad_y == 0.0


def test_transform_geometry_point_matches_scalar_transform():
    zone = 2
    x, y, _ = wgs84_to_msk59(PERM_CENTER_LON, PERM_CENTER_LAT, zone=zone)
    transformed = transform_geometry_to_msk59(Point(PERM_CENTER_LON, PERM_CENTER_LAT), zone=zone)
    assert transformed.x == pytest.approx(x)
    assert transformed.y == pytest.approx(y)


def test_transform_geometry_line_preserves_vertex_count_and_order():
    zone = 2
    line = LineString([(PERM_CENTER_LON, PERM_CENTER_LAT), (PERM_CENTER_LON + 0.01, PERM_CENTER_LAT + 0.01)])
    transformed = transform_geometry_to_msk59(line, zone=zone)
    assert len(transformed.coords) == 2
    x0, y0, _ = wgs84_to_msk59(PERM_CENTER_LON, PERM_CENTER_LAT, zone=zone)
    assert transformed.coords[0] == pytest.approx((x0, y0))


def test_transform_geometry_polygon_stays_valid_and_closed():
    zone = 2
    poly = Polygon(
        [
            (PERM_CENTER_LON, PERM_CENTER_LAT),
            (PERM_CENTER_LON + 0.001, PERM_CENTER_LAT),
            (PERM_CENTER_LON + 0.001, PERM_CENTER_LAT + 0.001),
            (PERM_CENTER_LON, PERM_CENTER_LAT + 0.001),
        ]
    )
    transformed = transform_geometry_to_msk59(poly, zone=zone)
    assert transformed.is_valid
    assert transformed.exterior.coords[0] == transformed.exterior.coords[-1]
    assert transformed.area > 0
