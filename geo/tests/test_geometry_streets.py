"""Тесты полос через osm2streets (Шаг 2.3, п. 1). Реальный процесс Node.js
+ настоящий osm2streets-js (не мок) - пропускается, если Node.js или пакет
`osm2streets-js` недоступны (`geo/osm2streets/`, `npm install`)."""

from __future__ import annotations

import shutil

import pytest
from shapely.geometry import LineString

from topology_geo.geometry.streets import (
    OSM2STREETS_DIR,
    IntersectionArea,
    LaneRibbon,
    build_lane_network,
    is_osm2streets_available,
)
from topology_geo.osm.raw_roads import RawRoadWay

CENTER_LON, CENTER_LAT = 56.2430, 58.0105
ZONE = 2

pytestmark = pytest.mark.skipif(
    not is_osm2streets_available(),
    reason=f"требуется node и `npm install` в {OSM2STREETS_DIR} (пакет osm2streets-js)",
)


def _crossroads() -> list[RawRoadWay]:
    """4 улицы, сходящиеся в общем узле 1 - реальный перекрёсток."""
    return [
        RawRoadWay(
            osm_id=10, tags={"highway": "residential", "lanes": "2", "name": "North"},
            node_ids=[1, 2], geometry=LineString([(56.2430, 58.0105), (56.2430, 58.0125)]),
        ),
        RawRoadWay(
            osm_id=11, tags={"highway": "residential", "lanes": "2", "name": "East"},
            node_ids=[1, 3], geometry=LineString([(56.2430, 58.0105), (56.2460, 58.0105)]),
        ),
        RawRoadWay(
            osm_id=12, tags={"highway": "residential", "lanes": "2", "name": "South"},
            node_ids=[1, 4], geometry=LineString([(56.2430, 58.0105), (56.2430, 58.0085)]),
        ),
        RawRoadWay(
            osm_id=13, tags={"highway": "residential", "lanes": "2", "name": "West"},
            node_ids=[1, 5], geometry=LineString([(56.2430, 58.0105), (56.2400, 58.0105)]),
        ),
    ]


def test_empty_input_returns_empty_lists():
    lanes, intersections = build_lane_network([], CENTER_LON, CENTER_LAT, 500.0, ZONE)
    assert lanes == []
    assert intersections == []


def test_crossroads_produces_driving_and_sidewalk_lanes_per_road():
    lanes, intersections = build_lane_network(_crossroads(), CENTER_LON, CENTER_LAT, 500.0, ZONE)

    assert len(lanes) == 16  # 4 дороги x (2 проезжие полосы + 2 тротуара)
    assert all(isinstance(lane, LaneRibbon) for lane in lanes)

    types = {lane.lane_type for lane in lanes}
    assert types == {"Driving", "Sidewalk"}

    driving = [lane for lane in lanes if lane.lane_type == "Driving"]
    sidewalks = [lane for lane in lanes if lane.lane_type == "Sidewalk"]
    assert len(driving) == 8
    assert len(sidewalks) == 8
    assert all(lane.width_m == pytest.approx(3.0) for lane in driving)
    assert all(lane.width_m == pytest.approx(1.5) for lane in sidewalks)
    assert all(lane.direction in ("Fwd", "Back") for lane in lanes)

    way_ids_seen = {osm_id for lane in lanes for osm_id in lane.osm_way_ids}
    assert way_ids_seen == {10, 11, 12, 13}


def test_lane_polygons_are_valid_with_positive_area():
    lanes, _ = build_lane_network(_crossroads(), CENTER_LON, CENTER_LAT, 500.0, ZONE)
    for lane in lanes:
        assert lane.polygon.is_valid
        assert lane.polygon.area > 0


def test_lane_polygons_are_in_local_coordinates_near_origin():
    """Центр перекрёстка (узел 1) совпадает с центром буфера -> все полосы
    должны оказаться близко к (0, 0) в локальных координатах участка, а не
    где-то в районе исходных WGS-84 градусов или метров МСК-59."""
    lanes, _ = build_lane_network(_crossroads(), CENTER_LON, CENTER_LAT, 500.0, ZONE)
    for lane in lanes:
        cx, cy = lane.polygon.centroid.x, lane.polygon.centroid.y
        assert abs(cx) < 500.0
        assert abs(cy) < 500.0


def test_crossroads_produces_intersection_areas():
    _, intersections = build_lane_network(_crossroads(), CENTER_LON, CENTER_LAT, 500.0, ZONE)
    assert len(intersections) == 4
    assert all(isinstance(area, IntersectionArea) for area in intersections)
    assert all(area.polygon.is_valid and area.polygon.area > 0 for area in intersections)


def test_single_road_without_intersection_still_produces_lanes():
    single = [
        RawRoadWay(
            osm_id=20, tags={"highway": "residential", "lanes": "2"},
            node_ids=[100, 101], geometry=LineString([(56.2430, 58.0105), (56.2450, 58.0105)]),
        ),
    ]
    lanes, intersections = build_lane_network(single, 56.2440, 58.0105, 500.0, ZONE)
    assert len(lanes) == 4  # 2 проезжие + 2 тротуара
    assert intersections == []


def test_node_available_check_matches_shutil():
    assert is_osm2streets_available() == (
        shutil.which("node") is not None and (OSM2STREETS_DIR / "node_modules" / "osm2streets-js").is_dir()
    )
