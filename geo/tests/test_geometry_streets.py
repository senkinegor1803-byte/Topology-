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
    LaneMarking,
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
    network = build_lane_network([], CENTER_LON, CENTER_LAT, 500.0, ZONE)
    assert network.lanes == []
    assert network.intersections == []
    assert network.markings == []


def test_crossroads_produces_driving_and_sidewalk_lanes_per_road():
    lanes = build_lane_network(_crossroads(), CENTER_LON, CENTER_LAT, 500.0, ZONE).lanes
    assert all(isinstance(lane, LaneRibbon) for lane in lanes)

    driving = [lane for lane in lanes if lane.lane_type == "Driving"]
    sidewalks = [lane for lane in lanes if lane.lane_type == "Sidewalk"]
    assert len(driving) == 8
    assert len(sidewalks) == 8
    assert all(lane.width_m == pytest.approx(3.0) for lane in driving)
    assert all(lane.width_m == pytest.approx(1.5) for lane in sidewalks)
    assert all(lane.direction in ("Fwd", "Back") for lane in driving + sidewalks)

    way_ids_seen = {osm_id for lane in lanes for osm_id in lane.osm_way_ids}
    assert way_ids_seen == {10, 11, 12, 13}


def test_lane_polygons_are_valid_with_positive_area():
    lanes = build_lane_network(_crossroads(), CENTER_LON, CENTER_LAT, 500.0, ZONE).lanes
    for lane in lanes:
        assert lane.polygon.is_valid
        assert lane.polygon.area > 0


def test_lane_polygons_are_in_local_coordinates_near_origin():
    """Центр перекрёстка (узел 1) совпадает с центром буфера -> все полосы
    должны оказаться близко к (0, 0) в локальных координатах участка, а не
    где-то в районе исходных WGS-84 градусов или метров МСК-59."""
    lanes = build_lane_network(_crossroads(), CENTER_LON, CENTER_LAT, 500.0, ZONE).lanes
    for lane in lanes:
        cx, cy = lane.polygon.centroid.x, lane.polygon.centroid.y
        assert abs(cx) < 500.0
        assert abs(cy) < 500.0


def test_crossroads_produces_intersection_areas():
    intersections = build_lane_network(_crossroads(), CENTER_LON, CENTER_LAT, 500.0, ZONE).intersections
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
    network = build_lane_network(single, 56.2440, 58.0105, 500.0, ZONE)
    assert len(network.lanes) == 6  # 2 проезжие + 2 тротуара + 2 бордюра (по стороне)
    assert network.intersections == []


# --- разметка (Шаг 2.3, п. 3) -------------------------------------------


def test_crossroads_produces_center_line_and_arrow_markings():
    markings = build_lane_network(_crossroads(), CENTER_LON, CENTER_LAT, 500.0, ZONE).markings
    assert len(markings) > 0
    assert all(isinstance(m, LaneMarking) for m in markings)
    kinds = {m.kind for m in markings}
    assert "center line" in kinds


def test_marking_polygons_are_valid_with_positive_area():
    markings = build_lane_network(_crossroads(), CENTER_LON, CENTER_LAT, 500.0, ZONE).markings
    for marking in markings:
        assert marking.polygon.is_valid
        assert marking.polygon.area > 0


def test_turn_lanes_tag_produces_lane_arrow_markings():
    roads = [
        RawRoadWay(
            osm_id=10, tags={"highway": "residential", "lanes": "2", "turn:lanes": "left|right"},
            node_ids=[1, 2], geometry=LineString([(56.2430, 58.0105), (56.2430, 58.0125)]),
        ),
        RawRoadWay(
            osm_id=11, tags={"highway": "residential", "lanes": "2"},
            node_ids=[1, 3], geometry=LineString([(56.2430, 58.0105), (56.2460, 58.0105)]),
        ),
        RawRoadWay(
            osm_id=12, tags={"highway": "residential", "lanes": "2"},
            node_ids=[1, 4], geometry=LineString([(56.2430, 58.0105), (56.2430, 58.0085)]),
        ),
    ]
    markings = build_lane_network(roads, CENTER_LON, CENTER_LAT, 500.0, ZONE).markings
    kinds = {m.kind for m in markings}
    assert "lane arrow" in kinds


# --- бордюр (Шаг 2.3, п. 2, 4) ------------------------------------------


def test_paved_road_gets_curb_between_driving_and_sidewalk():
    single = [
        RawRoadWay(
            osm_id=20, tags={"highway": "residential", "lanes": "2"},
            node_ids=[100, 101], geometry=LineString([(56.2430, 58.0105), (56.2450, 58.0105)]),
        ),
    ]
    lanes = build_lane_network(single, 56.2440, 58.0105, 500.0, ZONE).lanes
    curbs = [lane for lane in lanes if lane.lane_type == "Curb"]

    assert len(curbs) == 2  # по бордюру на каждую сторону дороги
    assert all(curb.width_m == pytest.approx(0.15) for curb in curbs)
    assert all(curb.osm_way_ids == (20,) for curb in curbs)
    assert all(curb.polygon.is_valid and curb.polygon.area > 0 for curb in curbs)


def test_dirt_road_gets_no_curb():
    single = [
        RawRoadWay(
            osm_id=21, tags={"highway": "track", "surface": "ground"},
            node_ids=[100, 101], geometry=LineString([(56.2430, 58.0105), (56.2450, 58.0105)]),
        ),
    ]
    lanes = build_lane_network(single, 56.2440, 58.0105, 500.0, ZONE).lanes
    assert not any(lane.lane_type == "Curb" for lane in lanes)


def test_crossroads_curbs_only_on_paved_roads():
    roads = _crossroads()
    roads[-1] = RawRoadWay(
        osm_id=13, tags={"highway": "track", "surface": "unpaved", "name": "West"},
        node_ids=[1, 5], geometry=LineString([(56.2430, 58.0105), (56.2400, 58.0105)]),
    )
    lanes = build_lane_network(roads, CENTER_LON, CENTER_LAT, 500.0, ZONE).lanes
    curbs = [lane for lane in lanes if lane.lane_type == "Curb"]

    assert len(curbs) == 6  # 3 мощёные дороги x 2 бордюра, грунтовая (13) - без
    assert not any(13 in curb.osm_way_ids for curb in curbs)


# --- покрытие (Шаг 2.3, п. 4) -------------------------------------------


def test_lane_surface_matches_source_way_tag():
    single = [
        RawRoadWay(
            osm_id=30, tags={"highway": "residential", "lanes": "2", "surface": "asphalt"},
            node_ids=[100, 101], geometry=LineString([(56.2430, 58.0105), (56.2450, 58.0105)]),
        ),
    ]
    lanes = build_lane_network(single, 56.2440, 58.0105, 500.0, ZONE).lanes
    non_curb = [lane for lane in lanes if lane.lane_type != "Curb"]
    assert non_curb  # проезжая часть/тротуар нашлись
    assert all(lane.surface == "asphalt" for lane in non_curb)


def test_lane_surface_is_none_when_tag_absent():
    single = [
        RawRoadWay(
            osm_id=31, tags={"highway": "residential", "lanes": "2"},
            node_ids=[100, 101], geometry=LineString([(56.2430, 58.0105), (56.2450, 58.0105)]),
        ),
    ]
    lanes = build_lane_network(single, 56.2440, 58.0105, 500.0, ZONE).lanes
    assert all(lane.surface is None for lane in lanes if lane.lane_type != "Curb")


def test_curb_surface_is_none_not_a_real_tag():
    single = [
        RawRoadWay(
            osm_id=32, tags={"highway": "residential", "lanes": "2", "surface": "asphalt"},
            node_ids=[100, 101], geometry=LineString([(56.2430, 58.0105), (56.2450, 58.0105)]),
        ),
    ]
    lanes = build_lane_network(single, 56.2440, 58.0105, 500.0, ZONE).lanes
    curbs = [lane for lane in lanes if lane.lane_type == "Curb"]
    assert curbs
    assert all(curb.surface is None for curb in curbs)


def test_different_roads_keep_their_own_surface():
    # 2 дороги, встречающиеся только друг с другом (степень узла 2), для
    # osm2streets - не настоящий перекрёсток, они склеиваются в один "road"
    # (проверено эмпирически: тогда у объединённой полосы surface только от
    # первого way) - берём настоящий перекрёсток (степень 3), чтобы дороги
    # остались раздельными.
    roads = [
        RawRoadWay(
            osm_id=40, tags={"highway": "residential", "lanes": "2", "surface": "asphalt"},
            node_ids=[1, 2], geometry=LineString([(56.2430, 58.0105), (56.2430, 58.0125)]),
        ),
        RawRoadWay(
            osm_id=41, tags={"highway": "residential", "lanes": "2", "surface": "paving_stones"},
            node_ids=[1, 3], geometry=LineString([(56.2430, 58.0105), (56.2460, 58.0105)]),
        ),
        RawRoadWay(
            osm_id=42, tags={"highway": "residential", "lanes": "2", "surface": "asphalt"},
            node_ids=[1, 4], geometry=LineString([(56.2430, 58.0105), (56.2430, 58.0085)]),
        ),
    ]
    lanes = build_lane_network(roads, CENTER_LON, CENTER_LAT, 500.0, ZONE).lanes
    surfaces_by_way = {40: set(), 41: set(), 42: set()}
    for lane in lanes:
        for way_id in lane.osm_way_ids:
            if lane.surface is not None:
                surfaces_by_way[way_id].add(lane.surface)
    assert surfaces_by_way[40] == {"asphalt"}
    assert surfaces_by_way[41] == {"paving_stones"}
    assert surfaces_by_way[42] == {"asphalt"}


def test_node_available_check_matches_shutil():
    assert is_osm2streets_available() == (
        shutil.which("node") is not None and (OSM2STREETS_DIR / "node_modules" / "osm2streets-js").is_dir()
    )
