"""Тесты мостов/путепроводов (Шаг 2.5, п. 1-3): обнаружение по тегу,
интерполяция отметок между устоями (не рельеф), проверка габарита над
нижележащей дорогой/путями."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, MultiLineString

from topology_geo.geometry.bridges import (
    MIN_CLEARANCE_RAIL_M,
    MIN_CLEARANCE_ROAD_M,
    STATUS_CALCULATED,
    STATUS_OFFICIAL,
    BridgeRibbon,
    abutment_elevation_fn,
    build_bridge_ribbons,
    is_bridge,
)
from topology_geo.geometry.road_network import NETWORK_BACKBONE, NETWORK_INTERNAL
from topology_geo.geometry.roads import build_road_ribbons
from topology_geo.selection.service import SiteFeature


def _feature(layer, geometry, osm_id=1, attributes=None, raw_tags=None) -> SiteFeature:
    return SiteFeature(
        layer=layer, osm_id=osm_id, osm_type="W", geometry=geometry,
        attributes=attributes or {}, confidence={}, raw_tags=raw_tags or {},
    )


# --- is_bridge ---------------------------------------------------------


@pytest.mark.parametrize("value", ["yes", "viaduct", "movable", "covered", "YES"])
def test_is_bridge_true_for_real_values(value):
    assert is_bridge({"bridge": value}) is True


@pytest.mark.parametrize("tags", [{}, {"bridge": "no"}, {"bridge": ""}, {"bridge": "  "}])
def test_is_bridge_false_for_absent_or_no(tags):
    assert is_bridge(tags) is False


def test_build_road_ribbons_excludes_bridge_tagged_features():
    """Мостовой участок не должен попадать в build_road_ribbons (Шаг 2.5,
    п. 1) - он строится отдельно, build_bridge_ribbons."""
    bridge_feature = _feature(
        "osm_roads", LineString([(0, 0), (100, 0)]), osm_id=1,
        attributes={"highway_class": "primary"}, raw_tags={"highway": "primary", "bridge": "yes"},
    )
    plain_feature = _feature(
        "osm_roads", LineString([(200, 0), (300, 0)]), osm_id=2,
        attributes={"highway_class": "primary"}, raw_tags={"highway": "primary"},
    )
    ribbons = build_road_ribbons([bridge_feature, plain_feature])
    assert {r.osm_id for r in ribbons} == {2}


# --- abutment_elevation_fn ------------------------------------------------


def test_abutment_elevation_interpolates_linearly_between_ends():
    axis = LineString([(0, 0), (100, 0)])

    def terrain(x, y):
        # Ловушка: посередине рельеф совсем другой - профиль моста НЕ должен
        # его учитывать (см. следующий тест).
        if x == 0:
            return 100.0
        if x == 100:
            return 110.0
        return -1000.0

    deck_fn = abutment_elevation_fn(axis, terrain)
    assert deck_fn(0.0, 0.0) == pytest.approx(100.0)
    assert deck_fn(100.0, 0.0) == pytest.approx(110.0)
    assert deck_fn(50.0, 0.0) == pytest.approx(105.0)  # линейная середина, не -1000


def test_abutment_elevation_ignores_terrain_dip_between_ends():
    """Ключевое отличие от дороги/воды: пролётное строение жёсткое, его
    отметка НЕ следует за рельефом между устоями, только за отметками на
    самих устоях."""
    axis = LineString([(0, 0), (100, 0)])

    def terrain_flat_ends_deep_dip(x, y):
        return 100.0 if x in (0.0, 100.0) else 20.0

    deck_fn = abutment_elevation_fn(axis, terrain_flat_ends_deep_dip)
    assert deck_fn(50.0, 0.0) == pytest.approx(100.0)  # не проваливается к 20.0


# --- build_bridge_ribbons: базовая геометрия -----------------------------


def _flat_terrain(z: float):
    return lambda x, y: z


def test_build_bridge_ribbons_finds_bridge_and_builds_ribbon():
    feature = _feature(
        "osm_roads", LineString([(0, 0), (100, 0)]), osm_id=10,
        attributes={"highway_class": "primary", "surface": "asphalt"},
        raw_tags={"highway": "primary", "bridge": "yes", "width": "10"},
    )
    bridges = build_bridge_ribbons([feature], _flat_terrain(100.0))
    assert len(bridges) == 1
    bridge = bridges[0]
    assert isinstance(bridge, BridgeRibbon)
    assert bridge.osm_id == 10
    assert bridge.width_m == pytest.approx(10.0)
    assert bridge.surface == "asphalt"
    assert bridge.network == NETWORK_BACKBONE
    assert bridge.ribbon.area == pytest.approx(100.0 * 10.0, rel=1e-6)
    assert bridge.axis is not None
    assert bridge.status == STATUS_OFFICIAL
    assert bridge.clearance_m is None  # нет crossing_axes - проверка не выполнялась


def test_build_bridge_ribbons_ignores_non_bridge_features():
    feature = _feature(
        "osm_roads", LineString([(0, 0), (100, 0)]), attributes={"highway_class": "primary"},
        raw_tags={"highway": "primary"},
    )
    assert build_bridge_ribbons([feature], _flat_terrain(100.0)) == []


def test_build_bridge_ribbons_skips_multilinestring_geometry():
    """Составная геометрия - нет единственной оси для интерполяции устоев
    (тот же принцип, что у RoadRibbon.axis, Шаг 2.4)."""
    feature = _feature(
        "osm_roads", MultiLineString([[(0, 0), (50, 0)], [(60, 0), (100, 0)]]),
        attributes={"highway_class": "primary"}, raw_tags={"highway": "primary", "bridge": "yes"},
    )
    assert build_bridge_ribbons([feature], _flat_terrain(100.0)) == []


def test_build_bridge_ribbons_internal_network_for_residential():
    feature = _feature(
        "osm_roads", LineString([(0, 0), (50, 0)]), attributes={"highway_class": "residential"},
        raw_tags={"highway": "residential", "bridge": "yes"},
    )
    bridges = build_bridge_ribbons([feature], _flat_terrain(100.0))
    assert bridges[0].network == NETWORK_INTERNAL


# --- проверка габарита (п. 3) ---------------------------------------------


def _terrain_with_crossing_dip(abutment_z: float, crossing_z: float):
    """Плоские устои на 0 и 100, ровно посередине (x=50) - другая отметка
    (место пересечения с нижележащей дорогой/путями)."""

    def _fn(x, y):
        return crossing_z if x == 50.0 else abutment_z
    return _fn


def test_build_bridge_ribbons_lifts_span_when_road_clearance_violated():
    bridge_feature = _feature(
        "osm_roads", LineString([(0, 0), (100, 0)]), osm_id=20,
        attributes={"highway_class": "primary"}, raw_tags={"highway": "primary", "bridge": "yes"},
    )
    crossing_road_axis = LineString([(50, -10), (50, 10)])
    # Устои и место пересечения на одной отметке -> габарит 0 м, меньше нормы.
    terrain = _terrain_with_crossing_dip(abutment_z=100.0, crossing_z=100.0)

    bridges = build_bridge_ribbons(
        [bridge_feature], terrain, crossing_road_axes=[crossing_road_axis],
    )
    bridge = bridges[0]
    assert bridge.status == STATUS_CALCULATED
    assert bridge.clearance_m == pytest.approx(MIN_CLEARANCE_ROAD_M)
    assert bridge.deck_elevation_fn(50.0, 0.0) == pytest.approx(100.0 + MIN_CLEARANCE_ROAD_M)


def test_build_bridge_ribbons_stays_official_when_clearance_sufficient():
    bridge_feature = _feature(
        "osm_roads", LineString([(0, 0), (100, 0)]), osm_id=21,
        attributes={"highway_class": "primary"}, raw_tags={"highway": "primary", "bridge": "yes"},
    )
    crossing_road_axis = LineString([(50, -10), (50, 10)])
    # Устои на 110, пересечение на 100 -> габарит 10 м, больше нормы 5 м.
    terrain = _terrain_with_crossing_dip(abutment_z=110.0, crossing_z=100.0)

    bridges = build_bridge_ribbons(
        [bridge_feature], terrain, crossing_road_axes=[crossing_road_axis],
    )
    bridge = bridges[0]
    assert bridge.status == STATUS_OFFICIAL
    assert bridge.clearance_m == pytest.approx(10.0)
    assert bridge.deck_elevation_fn(50.0, 0.0) == pytest.approx(110.0)  # не поднят


def test_build_bridge_ribbons_uses_higher_clearance_norm_for_rail():
    bridge_feature = _feature(
        "osm_roads", LineString([(0, 0), (100, 0)]), osm_id=22,
        attributes={"highway_class": "primary"}, raw_tags={"highway": "primary", "bridge": "yes"},
    )
    crossing_rail_axis = LineString([(50, -10), (50, 10)])
    terrain = _terrain_with_crossing_dip(abutment_z=100.0, crossing_z=100.0)

    bridges = build_bridge_ribbons(
        [bridge_feature], terrain, crossing_rail_axes=[crossing_rail_axis],
    )
    bridge = bridges[0]
    assert bridge.status == STATUS_CALCULATED
    assert bridge.clearance_m == pytest.approx(MIN_CLEARANCE_RAIL_M)
    assert MIN_CLEARANCE_RAIL_M > MIN_CLEARANCE_ROAD_M  # норматив для пути строже


def test_build_bridge_ribbons_no_crossings_means_no_clearance_check():
    """Без осей других объектов - официальный статус, проверка не
    выполняется (честный водопад, а не изобретённое нарушение)."""
    bridge_feature = _feature(
        "osm_roads", LineString([(0, 0), (100, 0)]), osm_id=23,
        attributes={"highway_class": "primary"}, raw_tags={"highway": "primary", "bridge": "yes"},
    )
    bridges = build_bridge_ribbons([bridge_feature], _flat_terrain(100.0))
    bridge = bridges[0]
    assert bridge.status == STATUS_OFFICIAL
    assert bridge.clearance_m is None


def test_build_bridge_ribbons_no_intersection_means_no_clearance_check():
    """Ось нижележащего объекта не пересекает мост геометрически -> не
    учитывается вовсе."""
    bridge_feature = _feature(
        "osm_roads", LineString([(0, 0), (100, 0)]), osm_id=24,
        attributes={"highway_class": "primary"}, raw_tags={"highway": "primary", "bridge": "yes"},
    )
    far_away_axis = LineString([(1000, -10), (1000, 10)])
    bridges = build_bridge_ribbons(
        [bridge_feature], _flat_terrain(100.0), crossing_road_axes=[far_away_axis],
    )
    bridge = bridges[0]
    assert bridge.status == STATUS_OFFICIAL
    assert bridge.clearance_m is None
