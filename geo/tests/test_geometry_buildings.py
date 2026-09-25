"""Тесты экструзии зданий (Шаг 1.6). Критерий приёмки шага: 100% контуров
превращаются в корректные тела; здесь же — проверка формулы высоты
(`docs/math-model.md` §2.5) и исправления геометрии."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from topology_geo.geometry.buildings import (
    CONFIDENCE_DEFAULT,
    CONFIDENCE_FACT,
    compute_height_m,
    extrude_buildings,
    repair_footprint,
)
from topology_geo.selection.service import SiteFeature

SQUARE = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def _feature(raw_tags=None, attributes=None, confidence=None, geometry=SQUARE, osm_id=1) -> SiteFeature:
    return SiteFeature(
        layer="osm_buildings", osm_id=osm_id, osm_type="W", geometry=geometry,
        attributes=attributes or {}, confidence=confidence or {}, raw_tags=raw_tags or {},
    )


# --- compute_height_m ---------------------------------------------------


def test_height_tag_takes_priority():
    feature = _feature(raw_tags={"height": "17.5"}, attributes={"levels": 3}, confidence={"levels": CONFIDENCE_FACT})
    height, confidence = compute_height_m(feature)
    assert height == pytest.approx(17.5)
    assert confidence == CONFIDENCE_FACT


def test_malformed_height_tag_falls_back_to_levels():
    feature = _feature(raw_tags={"height": "tall"}, attributes={"levels": 4}, confidence={"levels": CONFIDENCE_FACT})
    height, confidence = compute_height_m(feature)
    assert height == pytest.approx(4 * 3 + 1)
    assert confidence == CONFIDENCE_FACT


def test_negative_height_tag_falls_back_to_levels():
    feature = _feature(raw_tags={"height": "-5"}, attributes={"levels": 2}, confidence={"levels": CONFIDENCE_FACT})
    height, confidence = compute_height_m(feature)
    assert height == pytest.approx(2 * 3 + 1)
    assert confidence == CONFIDENCE_FACT


def test_levels_used_when_no_height_tag():
    feature = _feature(attributes={"levels": 5}, confidence={"levels": CONFIDENCE_FACT})
    height, confidence = compute_height_m(feature)
    assert height == pytest.approx(5 * 3 + 1)
    assert confidence == CONFIDENCE_FACT


def test_default_levels_confidence_does_not_count_as_fact():
    """levels=1 умолчанием (Шаг 1.4) не должен трактоваться как факт этажности."""
    feature = _feature(attributes={"type": "house", "levels": 1}, confidence={"levels": CONFIDENCE_DEFAULT})
    height, confidence = compute_height_m(feature)
    assert height == pytest.approx(8.0)  # default_height("house")
    assert confidence == CONFIDENCE_DEFAULT


def test_known_type_default_height():
    feature = _feature(attributes={"type": "apartments", "levels": 1}, confidence={"levels": CONFIDENCE_DEFAULT})
    height, confidence = compute_height_m(feature)
    assert height == pytest.approx(25.0)
    assert confidence == CONFIDENCE_DEFAULT


def test_unknown_type_uses_fallback_default():
    feature = _feature(attributes={"type": "some_unusual_tag", "levels": 1}, confidence={"levels": CONFIDENCE_DEFAULT})
    height, confidence = compute_height_m(feature)
    assert height == pytest.approx(9.0)
    assert confidence == CONFIDENCE_DEFAULT


# --- repair_footprint -----------------------------------------------------


def test_repair_simple_square_preserves_area():
    result = repair_footprint(SQUARE)
    assert result is not None
    assert result.area == pytest.approx(100.0)
    assert result.is_valid


def test_repair_forces_ccw_orientation():
    clockwise = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    result = repair_footprint(clockwise)
    assert result.exterior.is_ccw


def test_repair_fixes_self_intersecting_bowtie():
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    result = repair_footprint(bowtie)
    assert result is not None
    assert result.is_valid
    assert result.area > 0


def test_repair_removes_duplicate_consecutive_points():
    with_dupes = Polygon([(0, 0), (0, 0), (10, 0), (10, 0), (10, 10), (0, 10)])
    result = repair_footprint(with_dupes)
    assert result is not None
    assert result.area == pytest.approx(100.0)


def test_repair_rejects_degenerate_zero_area():
    degenerate = Polygon([(0, 0), (1, 0), (2, 0)])
    assert repair_footprint(degenerate) is None


def test_repair_rejects_empty_geometry():
    assert repair_footprint(Polygon()) is None


def test_repair_rejects_none():
    assert repair_footprint(None) is None


# --- extrude_buildings ------------------------------------------------------


def _flat_terrain(z: float):
    return lambda x, y: z


def test_extrude_produces_one_solid_per_valid_footprint():
    features = [
        _feature(osm_id=1, attributes={"type": "house", "levels": 2}, confidence={"levels": CONFIDENCE_FACT}),
        _feature(osm_id=2, geometry=Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])),
    ]
    solids = extrude_buildings(features, _flat_terrain(100.0))
    assert len(solids) == 2
    assert {s.osm_id for s in solids} == {1, 2}
    assert all(s.footprint.is_valid and s.footprint.area > 0 for s in solids)


def test_extrude_all_footprints_become_valid_solids_even_with_one_bad_input():
    """Критерий Шага 1.6: 100% контуров превращаются в корректные тела -
    невалидный контур отбрасывается, не ломая обработку остальных."""
    good_a = _feature(osm_id=1)
    good_b = _feature(osm_id=2, geometry=Polygon([(50, 50), (60, 50), (60, 60), (50, 60)]))
    bad = _feature(osm_id=3, geometry=Polygon([(0, 0), (1, 0), (2, 0)]))  # вырожденный

    solids = extrude_buildings([good_a, good_b, bad], _flat_terrain(100.0))

    assert {s.osm_id for s in solids} == {1, 2}
    assert all(s.footprint.is_valid for s in solids)


def test_extrude_base_z_is_minimum_terrain_at_footprint():
    def terrain(x, y):
        return 100.0 + x  # наклон - минимум на левой стороне контура (x=0)

    feature = _feature()
    solids = extrude_buildings([feature], terrain)
    assert solids[0].base_z == pytest.approx(100.0)  # min по контуру x=[0,10] -> x=0


def test_extrude_propagates_height_and_confidence():
    feature = _feature(raw_tags={"height": "12.3"})
    solids = extrude_buildings([feature], _flat_terrain(0.0))
    assert solids[0].height_m == pytest.approx(12.3)
    assert solids[0].height_confidence == CONFIDENCE_FACT


def test_extrude_ignores_non_building_layers():
    non_building = SiteFeature(
        layer="osm_vegetation", osm_id=99, osm_type="N", geometry=SQUARE,
        attributes={}, confidence={}, raw_tags={},
    )
    assert extrude_buildings([non_building], _flat_terrain(0.0)) == []
