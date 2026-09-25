"""Тесты обрезки по кругу и локализации координат (Шаг 1.4, п. 2). Чистая
геометрия (shapely) — Postgres не требуется."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, Point, Polygon

from topology_geo.selection.clip import clip_and_localize

CENTER_X, CENTER_Y = 2_300_000.0, 500_000.0
RADIUS = 100.0


def test_point_inside_circle_is_localized_to_relative_offset():
    point = Point(CENTER_X + 10, CENTER_Y + 5)
    result = clip_and_localize(point, CENTER_X, CENTER_Y, RADIUS)
    assert result is not None
    assert result.x == pytest.approx(10)
    assert result.y == pytest.approx(5)


def test_point_at_center_localizes_to_origin():
    point = Point(CENTER_X, CENTER_Y)
    result = clip_and_localize(point, CENTER_X, CENTER_Y, RADIUS)
    assert result.x == pytest.approx(0)
    assert result.y == pytest.approx(0)


def test_point_outside_circle_returns_none():
    point = Point(CENTER_X + 1000, CENTER_Y)
    result = clip_and_localize(point, CENTER_X, CENTER_Y, RADIUS)
    assert result is None


def test_line_crossing_boundary_is_clipped_and_shorter():
    line = LineString([(CENTER_X - 500, CENTER_Y), (CENTER_X + 500, CENTER_Y)])
    result = clip_and_localize(line, CENTER_X, CENTER_Y, RADIUS)
    assert result is not None
    assert result.length == pytest.approx(2 * RADIUS, rel=1e-3)
    # локализовано: середина отрезка проходит через (0,0)
    assert result.centroid.x == pytest.approx(0, abs=1e-6)


def test_line_entirely_outside_returns_none():
    line = LineString([(CENTER_X + 1000, CENTER_Y), (CENTER_X + 2000, CENTER_Y)])
    assert clip_and_localize(line, CENTER_X, CENTER_Y, RADIUS) is None


def test_polygon_fully_inside_keeps_area_but_shifts_position():
    poly = Polygon(
        [
            (CENTER_X - 10, CENTER_Y - 10),
            (CENTER_X + 10, CENTER_Y - 10),
            (CENTER_X + 10, CENTER_Y + 10),
            (CENTER_X - 10, CENTER_Y + 10),
        ]
    )
    result = clip_and_localize(poly, CENTER_X, CENTER_Y, RADIUS)
    assert result is not None
    assert result.area == pytest.approx(poly.area)
    assert result.centroid.x == pytest.approx(0, abs=1e-6)
    assert result.centroid.y == pytest.approx(0, abs=1e-6)


def test_polygon_partially_overlapping_is_clipped_smaller():
    poly = Polygon(
        [
            (CENTER_X + 50, CENTER_Y - 200),
            (CENTER_X + 250, CENTER_Y - 200),
            (CENTER_X + 250, CENTER_Y + 200),
            (CENTER_X + 50, CENTER_Y + 200),
        ]
    )
    result = clip_and_localize(poly, CENTER_X, CENTER_Y, RADIUS)
    assert result is not None
    assert result.area < poly.area


def test_polygon_fully_outside_returns_none():
    poly = Polygon(
        [
            (CENTER_X + 1000, CENTER_Y),
            (CENTER_X + 1010, CENTER_Y),
            (CENTER_X + 1010, CENTER_Y + 10),
            (CENTER_X + 1000, CENTER_Y + 10),
        ]
    )
    assert clip_and_localize(poly, CENTER_X, CENTER_Y, RADIUS) is None


def test_result_geometry_is_valid_when_present():
    poly = Polygon(
        [
            (CENTER_X + 50, CENTER_Y - 200),
            (CENTER_X + 250, CENTER_Y - 200),
            (CENTER_X + 250, CENTER_Y + 200),
            (CENTER_X + 50, CENTER_Y + 200),
        ]
    )
    result = clip_and_localize(poly, CENTER_X, CENTER_Y, RADIUS)
    assert result.is_valid
