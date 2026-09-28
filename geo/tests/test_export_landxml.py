"""Тесты Шага 2.10, п. 2: экспорт LandXML (поверхность рельефа + оси дорог/путей)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial import Delaunay
from shapely.geometry import LineString

from topology_geo.export.landxml import LANDXML_NAMESPACE, build_landxml
from topology_geo.geometry.rail import RailRibbon
from topology_geo.geometry.roads import RoadRibbon
from topology_geo.relief.tin import SiteTin

NS = {"lx": LANDXML_NAMESPACE}


def _make_flat_tin() -> SiteTin:
    xs, ys = np.meshgrid(np.linspace(0, 10, 3), np.linspace(0, 10, 3))
    zs = 100.0 + 0.1 * xs
    vertices = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    delaunay = Delaunay(vertices[:, :2])
    return SiteTin(vertices=vertices, triangles=delaunay.simplices, _delaunay=delaunay)


def _road(osm_id, axis) -> RoadRibbon:
    return RoadRibbon(
        osm_id=osm_id, ribbon=axis.buffer(3.0) if axis is not None else None,
        width_m=6.0, width_confidence="умолчание", surface="asphalt",
        highway_class="residential", network="внутриквартальная", axis=axis,
    )


def test_build_landxml_is_well_formed_and_namespaced():
    xml_bytes = build_landxml(None, [], [])
    assert xml_bytes.startswith(b"<?xml")
    root = ET.fromstring(xml_bytes)
    assert root.tag == f"{{{LANDXML_NAMESPACE}}}LandXML"


def test_build_landxml_without_tin_or_roads_has_no_surfaces_or_alignments():
    root = ET.fromstring(build_landxml(None, [], []))
    assert root.find("lx:Surfaces", NS) is None
    assert root.find("lx:Alignments", NS) is None


def test_build_landxml_surface_matches_tin_point_and_face_count():
    tin = _make_flat_tin()
    root = ET.fromstring(build_landxml(tin, [], []))
    pnts = root.findall(".//lx:Surfaces/lx:Surface/lx:Definition/lx:Pnts/lx:P", NS)
    faces = root.findall(".//lx:Surfaces/lx:Surface/lx:Definition/lx:Faces/lx:F", NS)
    assert len(pnts) == len(tin.vertices)
    assert len(faces) == len(tin.triangles)


def test_build_landxml_surface_point_coordinates_match_tin():
    tin = _make_flat_tin()
    root = ET.fromstring(build_landxml(tin, [], []))
    first_p = root.find(".//lx:Surfaces/lx:Surface/lx:Definition/lx:Pnts/lx:P", NS)
    x, y, z = (float(v) for v in first_p.text.split())
    assert (x, y, z) == pytest.approx(tuple(tin.vertices[0]))


def test_build_landxml_alignment_has_line_per_axis_segment():
    axis = LineString([(0, 0), (10, 0), (10, 10)])  # 2 сегмента
    road = _road(5, axis)
    root = ET.fromstring(build_landxml(None, [road], []))
    alignment = root.find(".//lx:Alignments/lx:Alignment", NS)
    assert alignment is not None
    assert "5" in alignment.get("name")
    assert float(alignment.get("length")) == pytest.approx(axis.length)
    lines = alignment.findall("lx:CoordGeom/lx:Line", NS)
    assert len(lines) == 2


def test_build_landxml_skips_roads_without_axis():
    road_no_axis = _road(6, None)
    root = ET.fromstring(build_landxml(None, [road_no_axis], []))
    assert root.find("lx:Alignments", NS) is None


def test_build_landxml_includes_rail_axis_with_its_own_prefix():
    axis = LineString([(0, 0), (50, 0)])
    track = RailRibbon(osm_id=9, ballast=axis.buffer(2.0), rail_type="rail", axis=axis)
    root = ET.fromstring(build_landxml(None, [], [track]))
    alignment = root.find(".//lx:Alignments/lx:Alignment", NS)
    assert alignment is not None
    assert alignment.get("name").startswith("Путь")
