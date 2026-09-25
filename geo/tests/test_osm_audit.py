"""Тесты аудита полноты OSM (Шаг 0.6). Элементы синтетические (нет реального пилота)."""

from __future__ import annotations

from shapely.geometry import LineString, Point, Polygon

from topology_geo.osm.audit import (
    OsmElement,
    build_completeness_report,
    compare_to_reference,
    is_courtyard_driveway,
    suggest_decisions,
)

PILOT = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])


def _building(id_: int, x: float, y: float, with_levels: bool) -> OsmElement:
    tags = {"building": "yes"}
    if with_levels:
        tags["building:levels"] = "5"
    return OsmElement(id=id_, kind="way", tags=tags, geometry=Point(x, y).buffer(1))


def _highway(id_: int, x0: float, y0: float, x1: float, y1: float, surface: bool, lanes: bool) -> OsmElement:
    tags = {"highway": "residential"}
    if surface:
        tags["surface"] = "asphalt"
    if lanes:
        tags["lanes"] = "2"
    return OsmElement(id=id_, kind="way", tags=tags, geometry=LineString([(x0, y0), (x1, y1)]))


def make_sample_elements() -> list[OsmElement]:
    elements = [
        _building(1, 10, 10, with_levels=True),
        _building(2, 20, 10, with_levels=True),
        _building(3, 30, 10, with_levels=False),
        _building(4, 40, 10, with_levels=False),
        _highway(10, 0, 50, 10, 50, surface=True, lanes=True),
        _highway(11, 10, 50, 20, 50, surface=True, lanes=False),
        _highway(12, 20, 50, 30, 50, surface=False, lanes=False),
        OsmElement(13, "way", {"highway": "service"}, LineString([(5, 5), (6, 6)])),
        OsmElement(14, "way", {"highway": "service", "service": "driveway"}, LineString([(6, 6), (7, 7)])),
        OsmElement(15, "way", {"highway": "service", "service": "parking_aisle"}, LineString([(7, 7), (8, 8)])),
        OsmElement(16, "node", {"power": "pole"}, Point(50, 50)),
        OsmElement(17, "node", {"power": "tower"}, Point(51, 51)),
        OsmElement(18, "node", {"natural": "tree"}, Point(60, 60)),
        OsmElement(19, "node", {"natural": "tree"}, Point(61, 61)),
        OsmElement(20, "node", {"natural": "tree"}, Point(62, 62)),
        # Вне пилота — не должен учитываться.
        _building(99, 1000, 1000, with_levels=True),
    ]
    return elements


def test_buildings_with_levels_ratio():
    report = build_completeness_report(make_sample_elements(), PILOT)
    assert report.buildings_with_levels.total == 4
    assert report.buildings_with_levels.complete == 2
    assert report.buildings_with_levels.ratio == 0.5


def test_roads_with_surface_and_lanes_ratio():
    report = build_completeness_report(make_sample_elements(), PILOT)
    # 3 highway=residential + 3 highway=service = 6 highways всего
    assert report.roads_with_surface.total == 6
    assert report.roads_with_surface.complete == 2
    assert report.roads_with_lanes.total == 6
    assert report.roads_with_lanes.complete == 1


def test_courtyard_driveways_counted_correctly():
    elements = make_sample_elements()
    driveways = [e for e in elements if is_courtyard_driveway(e)]
    assert len(driveways) == 3  # id 13 (без service), 14 (driveway), 15 (parking_aisle)


def test_power_poles_and_trees_counts():
    report = build_completeness_report(make_sample_elements(), PILOT)
    assert report.power_poles == 2
    assert report.trees == 3


def test_out_of_boundary_elements_excluded():
    report = build_completeness_report(make_sample_elements(), PILOT)
    # Здание #99 далеко за пределами пилота не должно попасть в total.
    assert report.buildings_with_levels.total == 4


def test_suggest_decisions_thresholds():
    report = build_completeness_report(make_sample_elements(), PILOT)
    decisions = suggest_decisions(report, good_threshold=0.8, poor_threshold=0.4)
    assert decisions["buildings_with_levels"] == "дополнить вручную"  # ratio 0.5
    assert decisions["roads_with_lanes"] == "нужен другой источник"  # ratio ~0.17


def test_compare_to_reference_reports_missing_and_coverage():
    report = build_completeness_report(make_sample_elements(), PILOT)
    comparison = compare_to_reference(report, {"power_poles": 4, "trees": 3})
    assert comparison["power_poles"]["found"] == 2
    assert comparison["power_poles"]["missing"] == 2
    assert comparison["power_poles"]["coverage"] == 0.5
    assert comparison["trees"]["coverage"] == 1.0
    assert comparison["trees"]["missing"] == 0


def test_report_without_boundary_uses_all_elements():
    elements = make_sample_elements()
    report = build_completeness_report(elements, boundary=None)
    assert report.buildings_with_levels.total == 5  # включая элемент вне PILOT
