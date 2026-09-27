"""Тесты Шага 2.7: опоры/башни ЛЭП, провода цепной линией, подстанции,
охранная зона по классу напряжения."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, Point, Polygon

from topology_geo.geometry.power import (
    CALCULATED_POLE_SPACING_M,
    CONFIDENCE_DEFAULT,
    CONFIDENCE_FACT,
    DEFAULT_CABLES,
    DEFAULT_POLE_HEIGHT_M,
    DEFAULT_TOWER_HEIGHT_M,
    PHASE_SPACING_M,
    POLE_RADIUS_M_BY_MATERIAL,
    SAG_RATIO,
    PoleTower,
    PowerSafetyZone,
    Substation,
    WireSpan,
    build_poles,
    build_power_safety_zones,
    build_substations,
    build_wire_spans,
    place_calculated_poles,
)
from topology_geo.selection.service import SiteFeature


def _feature(layer, geometry, osm_id=1, attributes=None, raw_tags=None) -> SiteFeature:
    return SiteFeature(
        layer=layer, osm_id=osm_id, osm_type="N", geometry=geometry,
        attributes=attributes or {}, confidence={}, raw_tags=raw_tags or {},
    )


# --- build_poles --------------------------------------------------------


def test_build_poles_reads_real_pole_with_height_tag():
    feature = _feature(
        "osm_power", Point(10, 20), osm_id=1,
        attributes={"voltage_kv": 10.0}, raw_tags={"power": "pole", "height": "9.5", "material": "concrete"},
    )
    poles = build_poles([feature])
    assert len(poles) == 1
    pole = poles[0]
    assert isinstance(pole, PoleTower)
    assert pole.osm_id == 1
    assert pole.height_m == pytest.approx(9.5)
    assert pole.height_confidence == CONFIDENCE_FACT
    assert pole.material == "concrete"
    assert pole.radius_base_m == pytest.approx(POLE_RADIUS_M_BY_MATERIAL["concrete"])
    assert pole.radius_base_m == pole.radius_top_m  # столб постоянного сечения
    assert pole.voltage_kv == pytest.approx(10.0)
    assert pole.source == CONFIDENCE_FACT


def test_build_poles_tower_defaults_and_tapers():
    feature = _feature("osm_power", Point(0, 0), raw_tags={"power": "tower"})
    pole = build_poles([feature])[0]
    assert pole.height_m == pytest.approx(DEFAULT_TOWER_HEIGHT_M)
    assert pole.height_confidence == CONFIDENCE_DEFAULT
    assert pole.radius_base_m > pole.radius_top_m  # сужается к вершине
    assert pole.material == "steel"


def test_build_poles_pole_defaults_wood():
    feature = _feature("osm_power", Point(0, 0), raw_tags={"power": "pole"})
    pole = build_poles([feature])[0]
    assert pole.height_m == pytest.approx(DEFAULT_POLE_HEIGHT_M)
    assert pole.material == "wood"


def test_build_poles_series_from_design_tag():
    feature = _feature("osm_power", Point(0, 0), raw_tags={"power": "tower", "design": "donau"})
    pole = build_poles([feature])[0]
    assert pole.series == "donau"


def test_build_poles_series_absent_when_no_tag():
    feature = _feature("osm_power", Point(0, 0), raw_tags={"power": "pole"})
    assert build_poles([feature])[0].series is None


def test_build_poles_ignores_non_point_and_wrong_power_value():
    line_feature = _feature("osm_power", LineString([(0, 0), (1, 1)]), raw_tags={"power": "line"})
    other_layer = _feature("osm_roads", Point(0, 0), raw_tags={"power": "pole"})
    assert build_poles([line_feature, other_layer]) == []


# --- place_calculated_poles ----------------------------------------------


def test_place_calculated_poles_fills_line_without_real_poles():
    length = 130.0
    feature = _feature("osm_power", LineString([(0, 0), (length, 0)]), raw_tags={"power": "line"})
    poles = place_calculated_poles([feature], real_poles=[])
    assert len(poles) >= 2
    assert all(p.source == CONFIDENCE_DEFAULT for p in poles)
    assert all(p.osm_id is None for p in poles)
    xs = sorted(p.x for p in poles)
    assert xs[0] == pytest.approx(0.0)
    assert xs[-1] == pytest.approx(length)


def test_place_calculated_poles_skips_line_with_nearby_real_pole():
    feature = _feature("osm_power", LineString([(0, 0), (100, 0)]), raw_tags={"power": "line"})
    real_pole = PoleTower(
        osm_id=5, x=50.0, y=0.5, height_m=9.0, height_confidence=CONFIDENCE_FACT,
        radius_base_m=0.1, radius_top_m=0.1, material="wood", series=None, voltage_kv=None, source=CONFIDENCE_FACT,
    )
    assert place_calculated_poles([feature], real_poles=[real_pole]) == []


def test_place_calculated_poles_ignores_non_line_power_values():
    feature = _feature("osm_power", Point(0, 0), raw_tags={"power": "pole"})
    assert place_calculated_poles([feature], real_poles=[]) == []


def test_place_calculated_poles_even_spacing_close_to_target():
    length = 185.0
    feature = _feature("osm_power", LineString([(0, 0), (length, 0)]), raw_tags={"power": "line"})
    poles = place_calculated_poles([feature], real_poles=[], spacing_m=CALCULATED_POLE_SPACING_M)
    xs = sorted(p.x for p in poles)
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    assert all(gap <= CALCULATED_POLE_SPACING_M + 1e-6 for gap in gaps)
    assert max(gaps) - min(gaps) < 1e-6  # равномерно, без обрубка последнего пролёта


# --- build_wire_spans ------------------------------------------------------


def _flat_terrain(z: float = 100.0):
    return lambda x, y: z


def _poles_pair(span_length=100.0, height=10.0):
    a = PoleTower(
        osm_id=1, x=0.0, y=0.0, height_m=height, height_confidence=CONFIDENCE_FACT,
        radius_base_m=0.1, radius_top_m=0.1, material="wood", series=None, voltage_kv=10.0, source=CONFIDENCE_FACT,
    )
    b = PoleTower(
        osm_id=2, x=span_length, y=0.0, height_m=height, height_confidence=CONFIDENCE_FACT,
        radius_base_m=0.1, radius_top_m=0.1, material="wood", series=None, voltage_kv=10.0, source=CONFIDENCE_FACT,
    )
    return a, b


def test_build_wire_spans_default_cable_count_and_sag():
    span_length = 100.0
    height = 10.0
    a, b = _poles_pair(span_length, height)
    feature = _feature("osm_power", LineString([(a.x, a.y), (b.x, b.y)]), osm_id=7, raw_tags={"power": "line"})
    spans = build_wire_spans([feature], [a, b], _flat_terrain(100.0))
    assert len({s.strand_index for s in spans}) == DEFAULT_CABLES
    assert all(isinstance(s, WireSpan) for s in spans)

    middle_strand = next(s for s in spans if s.strand_index == (DEFAULT_CABLES - 1) // 2 and abs(
        s.path[0][1]
    ) < 1e-6)
    z_attach = 100.0 + height * 0.9
    z_mid = middle_strand.path[len(middle_strand.path) // 2][2]
    assert z_mid < z_attach  # провис ниже линейной прямой между опорами
    expected_sag = span_length * SAG_RATIO
    assert z_attach - z_mid == pytest.approx(expected_sag, rel=1e-6)


def test_build_wire_spans_respects_cables_tag():
    a, b = _poles_pair()
    feature = _feature(
        "osm_power", LineString([(a.x, a.y), (b.x, b.y)]), osm_id=7, raw_tags={"power": "line", "cables": "6"}
    )
    spans = build_wire_spans([feature], [a, b], _flat_terrain(100.0))
    assert len({s.strand_index for s in spans}) == 6
    assert all(s.cables == 6 for s in spans)


def test_build_wire_spans_offsets_strands_laterally():
    a, b = _poles_pair()
    feature = _feature("osm_power", LineString([(a.x, a.y), (b.x, b.y)]), osm_id=7, raw_tags={"power": "line"})
    spans = build_wire_spans([feature], [a, b], _flat_terrain(100.0))
    start_ys = sorted({round(s.path[0][1], 6) for s in spans})
    assert len(start_ys) == DEFAULT_CABLES
    assert start_ys[0] == pytest.approx(-PHASE_SPACING_M)
    assert start_ys[-1] == pytest.approx(PHASE_SPACING_M)


def test_build_wire_spans_needs_at_least_two_poles():
    a, _ = _poles_pair()
    feature = _feature("osm_power", LineString([(0, 0), (100, 0)]), raw_tags={"power": "line"})
    assert build_wire_spans([feature], [a], _flat_terrain(100.0)) == []


def test_build_wire_spans_ignores_other_power_values():
    a, b = _poles_pair()
    feature = _feature("osm_power", Point(0, 0), raw_tags={"power": "pole"})
    assert build_wire_spans([feature], [a, b], _flat_terrain(100.0)) == []


# --- build_substations ------------------------------------------------------


def test_build_substations_uses_polygon_footprint():
    polygon = Polygon([(0, 0), (20, 0), (20, 10), (0, 10)])
    feature = _feature("osm_power", polygon, osm_id=9, raw_tags={"power": "substation"})
    substations = build_substations([feature], _flat_terrain(100.0))
    assert len(substations) == 1
    sub = substations[0]
    assert isinstance(sub, Substation)
    assert sub.base_z == pytest.approx(100.0)
    assert sub.height_m > 0
    assert sub.kind == "substation"


def test_build_substations_plant_also_handled():
    polygon = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    feature = _feature("osm_power", polygon, raw_tags={"power": "plant"})
    assert build_substations([feature], _flat_terrain(100.0))[0].kind == "plant"


def test_build_substations_skips_point_only_substation():
    feature = _feature("osm_power", Point(0, 0), raw_tags={"power": "substation"})
    assert build_substations([feature], _flat_terrain(100.0)) == []


# --- build_power_safety_zones -----------------------------------------------


def test_build_power_safety_zones_skips_without_voltage():
    feature = _feature("osm_power", LineString([(0, 0), (100, 0)]), raw_tags={"power": "line"})
    assert build_power_safety_zones([feature]) == []


def test_build_power_safety_zones_low_voltage_narrow():
    feature = _feature(
        "osm_power", LineString([(0, 0), (100, 0)]), attributes={"voltage_kv": 0.4}, raw_tags={"power": "line"}
    )
    zones = build_power_safety_zones([feature])
    assert len(zones) == 1
    zone = zones[0]
    assert isinstance(zone, PowerSafetyZone)
    assert zone.half_width_m == pytest.approx(2.0)
    assert zone.corridor.area == pytest.approx(100.0 * 2 * 2.0, rel=1e-6)


def test_build_power_safety_zones_high_voltage_wider():
    feature = _feature(
        "osm_power", LineString([(0, 0), (100, 0)]), attributes={"voltage_kv": 220.0}, raw_tags={"power": "line"}
    )
    zone = build_power_safety_zones([feature])[0]
    assert zone.half_width_m == pytest.approx(25.0)


def test_build_power_safety_zones_above_max_class_uses_largest_norm():
    feature = _feature(
        "osm_power", LineString([(0, 0), (100, 0)]), attributes={"voltage_kv": 1150.0}, raw_tags={"power": "line"}
    )
    zone = build_power_safety_zones([feature])[0]
    assert zone.half_width_m == pytest.approx(55.0)
