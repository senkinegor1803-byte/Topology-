"""Тесты Шага 2.6: ширина насыпи по числу путей и колее, платформы, переезды,
опоры контактной сети упрощённо. Базовая геометрия ленты пути (Шаг 1.7) — в
`test_geometry_environment.py`."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, Point, Polygon

from topology_geo.geometry.rail import (
    BALLAST_WIDTH_M,
    CONFIDENCE_DEFAULT,
    CONFIDENCE_FACT,
    DEFAULT_GAUGE_M,
    POLE_OFFSET_FROM_AXIS_M,
    POLE_SPACING_M,
    TRACK_SPACING_M,
    LevelCrossing,
    PlatformArea,
    build_level_crossings,
    build_platform_areas,
    build_rail_ribbons,
    compute_ballast_width_m,
    place_catenary_poles,
)
from topology_geo.selection.service import SiteFeature


def _feature(layer, geometry, osm_id=1, attributes=None, raw_tags=None) -> SiteFeature:
    return SiteFeature(
        layer=layer, osm_id=osm_id, osm_type="W", geometry=geometry,
        attributes=attributes or {}, confidence={}, raw_tags=raw_tags or {},
    )


# --- compute_ballast_width_m ------------------------------------------------


def test_compute_ballast_width_defaults_to_single_track_legacy_width():
    width, tracks, gauge, confidence = compute_ballast_width_m({})
    assert width == pytest.approx(BALLAST_WIDTH_M)
    assert tracks == 1
    assert gauge == pytest.approx(DEFAULT_GAUGE_M)
    assert confidence == CONFIDENCE_DEFAULT


def test_compute_ballast_width_grows_with_track_count():
    width_1, *_ = compute_ballast_width_m({"tracks": "1"})
    width_2, tracks_2, _, confidence_2 = compute_ballast_width_m({"tracks": "2"})
    assert tracks_2 == 2
    assert confidence_2 == CONFIDENCE_FACT
    assert width_2 == pytest.approx(width_1 + TRACK_SPACING_M)


def test_compute_ballast_width_uses_gauge_tag_in_mm():
    width_default, *_ = compute_ballast_width_m({})
    width_narrow, tracks, gauge_m, confidence = compute_ballast_width_m({"gauge": "1000"})
    assert tracks == 1
    assert gauge_m == pytest.approx(1.0)
    assert confidence == CONFIDENCE_FACT
    assert width_narrow < width_default  # уже колея -> уже насыпь


@pytest.mark.parametrize("bad_tracks", ["0", "-1", "abc", ""])
def test_compute_ballast_width_tolerates_bad_tracks_tag(bad_tracks):
    width, tracks, _, confidence = compute_ballast_width_m({"tracks": bad_tracks})
    assert tracks == 1
    assert confidence == CONFIDENCE_DEFAULT
    assert width == pytest.approx(BALLAST_WIDTH_M)


@pytest.mark.parametrize("bad_gauge", ["0", "-100", "abc", ""])
def test_compute_ballast_width_tolerates_bad_gauge_tag(bad_gauge):
    width, _, gauge_m, confidence = compute_ballast_width_m({"gauge": bad_gauge})
    assert gauge_m == pytest.approx(DEFAULT_GAUGE_M)
    assert confidence == CONFIDENCE_DEFAULT
    assert width == pytest.approx(BALLAST_WIDTH_M)


def test_build_rail_ribbons_stores_tracks_gauge_and_width():
    feature = _feature(
        "osm_railways", LineString([(0, 0), (100, 0)]),
        raw_tags={"railway": "rail", "tracks": "2", "gauge": "1520"},
    )
    ribbon = build_rail_ribbons([feature])[0]
    assert ribbon.tracks == 2
    assert ribbon.gauge_m == pytest.approx(1.520)
    assert ribbon.width_confidence == CONFIDENCE_FACT
    assert ribbon.ballast.area == pytest.approx(100.0 * ribbon.width_m, rel=1e-6)
    assert ribbon.width_m > BALLAST_WIDTH_M  # два пути шире одного


# --- платформы ---------------------------------------------------------------


def test_build_platform_areas_uses_polygon_as_is():
    polygon = Polygon([(0, 0), (20, 0), (20, 4), (0, 4)])
    feature = _feature("osm_railway_platforms", polygon, osm_id=30, raw_tags={"railway": "platform"})
    platforms = build_platform_areas([feature])
    assert len(platforms) == 1
    assert isinstance(platforms[0], PlatformArea)
    assert platforms[0].footprint.area == pytest.approx(80.0)
    assert platforms[0].height_m > 0


def test_build_platform_areas_buffers_line_platform():
    line = LineString([(0, 0), (50, 0)])
    feature = _feature("osm_railway_platforms", line, osm_id=31, raw_tags={"railway": "platform"})
    platforms = build_platform_areas([feature])
    assert len(platforms) == 1
    assert platforms[0].footprint.area > 0  # линия сама по себе площади не имеет - буферизована


def test_build_platform_areas_ignores_other_layers():
    feature = _feature("osm_railways", LineString([(0, 0), (10, 0)]), raw_tags={"railway": "rail"})
    assert build_platform_areas([feature]) == []


# --- переезды -----------------------------------------------------------------


def test_build_level_crossings_uses_nearest_rail_width():
    rail_feature = _feature(
        "osm_railways", LineString([(0, 0), (100, 0)]), osm_id=1, raw_tags={"railway": "rail", "tracks": "2"},
    )
    rail = build_rail_ribbons([rail_feature])
    crossing_feature = _feature(
        "osm_railway_crossings", Point(50, 0), osm_id=40, raw_tags={"railway": "level_crossing"},
    )
    crossings = build_level_crossings([crossing_feature], rail)
    assert len(crossings) == 1
    crossing = crossings[0]
    assert isinstance(crossing, LevelCrossing)
    assert crossing.crossing_type == "level_crossing"
    assert crossing.size_m == pytest.approx(rail[0].width_m)


def test_build_level_crossings_falls_back_without_rail_ribbons():
    crossing_feature = _feature(
        "osm_railway_crossings", Point(0, 0), osm_id=41, raw_tags={"railway": "crossing"},
    )
    crossings = build_level_crossings([crossing_feature], [])
    assert crossings[0].size_m == pytest.approx(BALLAST_WIDTH_M)
    assert crossings[0].crossing_type == "crossing"


def test_build_level_crossings_ignores_non_point_geometry():
    feature = _feature(
        "osm_railway_crossings", LineString([(0, 0), (1, 1)]), raw_tags={"railway": "level_crossing"},
    )
    assert build_level_crossings([feature], []) == []


# --- опоры контактной сети -----------------------------------------------------


def test_place_catenary_poles_only_for_electrified():
    electrified = _feature(
        "osm_railways", LineString([(0, 0), (100, 0)]), osm_id=1,
        raw_tags={"railway": "rail", "electrified": "contact_line"},
    )
    not_electrified = _feature(
        "osm_railways", LineString([(200, 0), (300, 0)]), osm_id=2, raw_tags={"railway": "rail"},
    )
    poles = place_catenary_poles([electrified, not_electrified])
    assert all(p.osm_id == 1 for p in poles)
    assert len(poles) > 0


def test_place_catenary_poles_respects_spacing_and_offset():
    length = 149.0
    feature = _feature(
        "osm_railways", LineString([(0, 0), (length, 0)]), osm_id=1,
        raw_tags={"railway": "rail", "electrified": "yes"},
    )
    poles = place_catenary_poles([feature], spacing_m=POLE_SPACING_M, offset_m=POLE_OFFSET_FROM_AXIS_M)
    expected_count = int(length // POLE_SPACING_M) + 1
    assert len(poles) == expected_count
    # ось вдоль X -> смещение опоры по Y на POLE_OFFSET_FROM_AXIS_M
    for pole in poles:
        assert abs(pole.y) == pytest.approx(POLE_OFFSET_FROM_AXIS_M)


def test_place_catenary_poles_electrified_no_is_not_electrified():
    feature = _feature(
        "osm_railways", LineString([(0, 0), (100, 0)]), raw_tags={"railway": "rail", "electrified": "no"},
    )
    assert place_catenary_poles([feature]) == []
