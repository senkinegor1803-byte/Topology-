"""Тесты Шага 3.8: очистка площадки и построение подъезда."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from shapely.geometry import LineString, Point, box

from topology_geo.geometry.road_network import NETWORK_BACKBONE, NETWORK_INTERNAL
from topology_geo.siting.site_access import build_access_road, clear_site


@dataclass
class _Feature:
    osm_id: int
    geometry: object


@dataclass
class _Road:
    osm_id: int
    geometry: object
    network: str


def test_clear_site_separates_intersecting_and_untouched_features():
    footprint = box(0, 0, 10, 10)
    inside = _Feature(osm_id=1, geometry=box(2, 2, 4, 4))
    overlapping = _Feature(osm_id=2, geometry=box(8, 8, 12, 12))
    outside = _Feature(osm_id=3, geometry=box(20, 20, 22, 22))

    result = clear_site([inside, overlapping, outside], footprint)

    assert {f.osm_id for f in result.removed} == {1, 2}
    assert {f.osm_id for f in result.kept} == {3}


def test_clear_site_empty_footprint_removes_nothing():
    footprint = box(100, 100, 101, 101)
    feature = _Feature(osm_id=1, geometry=box(0, 0, 1, 1))

    result = clear_site([feature], footprint)

    assert result.removed == []
    assert result.kept == [feature]


def test_build_access_road_connects_to_nearest_internal_road_no_confirmation():
    site_boundary = box(0, 0, 20, 20)
    internal_road = _Road(osm_id=1, geometry=LineString([(30, 10), (30, 30)]), network=NETWORK_INTERNAL)
    far_backbone = _Road(osm_id=2, geometry=LineString([(1000, 0), (1000, 100)]), network=NETWORK_BACKBONE)

    access = build_access_road(site_boundary, [internal_road, far_backbone], obstacles=[])

    assert access is not None
    assert access.connects_to.osm_id == 1
    assert access.requires_user_confirmation is False
    assert access.crosses_obstacle is False
    assert access.path.length == pytest.approx(10.0)  # от (20,10) до (30,10)


def test_build_access_road_to_backbone_requires_confirmation():
    site_boundary = box(0, 0, 20, 20)
    backbone_road = _Road(osm_id=1, geometry=LineString([(30, 10), (30, 30)]), network=NETWORK_BACKBONE)

    access = build_access_road(site_boundary, [backbone_road], obstacles=[])

    assert access.requires_user_confirmation is True


def test_build_access_road_detects_crossing_obstacle():
    site_boundary = box(0, 0, 20, 20)
    road = _Road(osm_id=1, geometry=LineString([(30, 10), (30, 30)]), network=NETWORK_INTERNAL)
    obstacle_building = box(22, 5, 28, 15)  # прямо на пути (20,10)->(30,10)

    access = build_access_road(site_boundary, [road], obstacles=[obstacle_building])

    assert access.crosses_obstacle is True


def test_build_access_road_no_roads_returns_none():
    site_boundary = box(0, 0, 20, 20)
    assert build_access_road(site_boundary, [], obstacles=[]) is None
