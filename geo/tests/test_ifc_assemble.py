"""Тесты сборки `site.ifc` (Шаг 1.8). Геометрия проверяется точными
аналитическими формулами (площадь треугольников, объём через теорему о
дивергенции для замкнутого меша); сборка IFC — через ifcopenshell.validate
на реальной модели (не мок)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import Delaunay
from shapely.geometry import LineString, Polygon

from topology_geo.geometry.buildings import BuildingSolid
from topology_geo.geometry.rail import RailRibbon
from topology_geo.geometry.roads import RoadRibbon
from topology_geo.geometry.vegetation import TreePoint
from topology_geo.geometry.water import WaterArea, WaterwayRibbon
from topology_geo.ifc.assemble import (
    BasePoint,
    SiteModel,
    build_site_ifc,
    extrude_polygon_mesh,
    flat_polygon_mesh,
    mesh_cylinder,
    triangulate_polygon,
)
from topology_geo.ifc.generate_test_ifc import validate_model
from topology_geo.relief.tin import SiteTin


def _triangle_area_3d(v0, v1, v2) -> float:
    a = np.array(v1) - np.array(v0)
    b = np.array(v2) - np.array(v0)
    return float(np.linalg.norm(np.cross(a, b)) / 2.0)


def _signed_volume(vertices, faces) -> float:
    vol = 0.0
    for a, b, c in faces:
        v0, v1, v2 = vertices[a], vertices[b], vertices[c]
        vol += (
            v0[0] * (v1[1] * v2[2] - v1[2] * v2[1])
            - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0])
            + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0])
        )
    return vol / 6.0


# --- triangulate_polygon / flat_polygon_mesh ---------------------------------


def test_triangulate_polygon_area_matches_shapely_for_simple_square():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    vertices, triangles = triangulate_polygon(poly)
    total = sum(_triangle_area_3d((*vertices[a], 0), (*vertices[b], 0), (*vertices[c], 0)) for a, b, c in triangles)
    assert total == pytest.approx(poly.area, rel=1e-9)


def test_triangulate_polygon_area_excludes_hole():
    poly = Polygon(
        [(-10, -10), (10, -10), (10, 10), (-10, 10)],
        [[(-4, -4), (4, -4), (4, 4), (-4, 4)][::-1]],
    )
    vertices, triangles = triangulate_polygon(poly)
    total = sum(_triangle_area_3d((*vertices[a], 0), (*vertices[b], 0), (*vertices[c], 0)) for a, b, c in triangles)
    assert total == pytest.approx(poly.area, rel=1e-9)
    assert poly.area == pytest.approx(400.0 - 64.0)


def test_flat_polygon_mesh_samples_elevation_at_each_vertex():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    vertices, _ = flat_polygon_mesh(poly, lambda x, y: 100.0 + x + 2 * y)
    for x, y, z in vertices:
        assert z == pytest.approx(100.0 + x + 2 * y)


def test_flat_polygon_mesh_uses_default_when_elevation_fn_returns_none():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    vertices, _ = flat_polygon_mesh(poly, lambda x, y: None, default_z=42.0)
    assert all(z == pytest.approx(42.0) for _, _, z in vertices)


# --- extrude_polygon_mesh (объём через теорему о дивергенции) ----------------


@pytest.mark.parametrize(
    "polygon,base_z,top_z",
    [
        (Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), 0.0, 5.0),
        (Polygon([(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]), 0.0, 3.0),  # невыпуклый (Г-образный)
        (
            Polygon(
                [(-10, -10), (10, -10), (10, 10), (-10, 10)],
                [[(-4, -4), (4, -4), (4, 4), (-4, 4)][::-1]],
            ),
            100.0,
            112.0,
        ),  # с двором
    ],
)
def test_extrude_polygon_mesh_volume_matches_footprint_area_times_height(polygon, base_z, top_z):
    vertices, faces = extrude_polygon_mesh(polygon, base_z, top_z)
    expected = polygon.area * (top_z - base_z)
    assert _signed_volume(vertices, faces) == pytest.approx(expected, rel=1e-9)


def test_extrude_polygon_mesh_is_watertight_manifold():
    # каждое ребро встречается ровно в двух треугольниках (в противоположных
    # направлениях) - необходимое условие замкнутости меша.
    polygon = Polygon(
        [(-10, -10), (10, -10), (10, 10), (-10, 10)],
        [[(-4, -4), (4, -4), (4, 4), (-4, 4)][::-1]],
    )
    _, faces = extrude_polygon_mesh(polygon, 0.0, 5.0)
    edge_count: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edge_count[(u, v)] = edge_count.get((u, v), 0) + 1
    for (u, v), count in edge_count.items():
        assert count == 1, f"ребро {(u, v)} встречается {count} раз(а)"
        assert edge_count.get((v, u), 0) == 1, f"обратное ребро {(v, u)} отсутствует"


def test_mesh_cylinder_has_expected_vertex_and_face_counts():
    verts, faces = mesh_cylinder(0.3, 6.0, segments=6)
    assert len(verts) == 12
    assert all(len(f) == 3 for f in faces)


# --- build_site_ifc -----------------------------------------------------------


def _make_flat_tin(half_extent: float = 50.0, n: int = 6) -> SiteTin:
    xs, ys = np.meshgrid(np.linspace(-half_extent, half_extent, n), np.linspace(-half_extent, half_extent, n))
    zs = 100.0 + 0.01 * xs + 0.02 * ys
    vertices = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    delaunay = Delaunay(vertices[:, :2])
    return SiteTin(vertices=vertices, triangles=delaunay.simplices, _delaunay=delaunay)


def _make_site_model() -> SiteModel:
    building = BuildingSolid(
        osm_id=1, footprint=Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
        height_m=12.0, height_confidence="факт", height_source="OSM", base_z=100.0, building_type="жилой",
    )
    road = RoadRibbon(
        osm_id=2, ribbon=LineString([(-40, 0), (40, 0)]).buffer(3.0, cap_style="flat"),
        width_m=6.0, width_confidence="умолчание", surface="asphalt", highway_class="residential",
    )
    water = WaterArea(osm_id=3, polygon=Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]), level_z=99.5)
    waterway = WaterwayRibbon(osm_id=4, ribbon=LineString([(0, -40), (0, 40)]).buffer(1.5, cap_style="flat"), width_m=3.0)
    rail = RailRibbon(osm_id=5, ballast=LineString([(-40, -20), (40, -20)]).buffer(2.0, cap_style="flat"), rail_type="tram")
    tree = TreePoint(x=15.0, y=-15.0, species="Betula pendula", confidence="факт", source_osm_id=6)
    return SiteModel(
        tin=_make_flat_tin(), buildings=[building], roads=[road],
        water_areas=[water], waterways=[waterway], rail=[rail], trees=[tree],
    )


BASE_POINT = BasePoint(lon=56.25, lat=58.0, zone=2, x=2_310_450.0, y=-5_857_320.0, height=150.0)


@pytest.mark.parametrize("schema", ["IFC4", "IFC4X3"])
def test_build_site_ifc_validates_without_issues(schema):
    f, _ = build_site_ifc(schema, _make_site_model(), BASE_POINT)
    assert validate_model(f) == []


def test_build_site_ifc_registers_one_global_id_per_source_object():
    f, registry = build_site_ifc("IFC4", _make_site_model(), BASE_POINT)
    layers = {layer for layer, _, _ in registry}
    assert layers == {"osm_buildings", "osm_roads", "osm_water_areas", "osm_waterways", "osm_railways", "osm_vegetation"}
    assert len(registry) == 6
    global_ids = [gid for _, _, gid in registry]
    assert len(set(global_ids)) == len(global_ids)  # уникальны
    all_global_ids_in_file = {p.GlobalId for p in f.by_type("IfcRoot")}
    assert set(global_ids) <= all_global_ids_in_file


def test_build_site_ifc_uses_native_ifcroad_in_ifc43():
    f, registry = build_site_ifc("IFC4X3", _make_site_model(), BASE_POINT)
    assert len(f.by_type("IfcRoad")) == 1
    assert len(f.by_type("IfcBuildingElementProxy")) == 2  # здание + ж/д (без нативного класса)
    road_gid = next(gid for layer, _, gid in registry if layer == "osm_roads")
    assert f.by_id(f.by_type("IfcRoad")[0].id()).GlobalId == road_gid


def test_build_site_ifc_uses_proxy_road_in_ifc4():
    # IFC4 не знает класс IfcRoad вовсе (не просто "нет экземпляров") -
    # by_type с несуществующим в схеме именем поднимает RuntimeError.
    f, _ = build_site_ifc("IFC4", _make_site_model(), BASE_POINT)
    assert len(f.by_type("IfcBuildingElementProxy")) == 3  # здание + дорога-прокси + ж/д


def test_build_site_ifc_georeferencing_matches_base_point():
    f, _ = build_site_ifc("IFC4", _make_site_model(), BASE_POINT)
    conversions = f.by_type("IfcMapConversion")
    assert len(conversions) == 1
    conv = conversions[0]
    assert conv.Eastings == pytest.approx(BASE_POINT.x)
    assert conv.Northings == pytest.approx(BASE_POINT.y)
    assert conv.OrthogonalHeight == pytest.approx(BASE_POINT.height)


def test_build_site_ifc_handles_empty_site_model():
    f, registry = build_site_ifc("IFC4", SiteModel(), BASE_POINT)
    assert registry == []
    assert validate_model(f) == []
    assert len(f.by_type("IfcSite")) == 1


def test_build_site_ifc_terrain_matches_tin_triangle_count():
    model = SiteModel(tin=_make_flat_tin())
    f, _ = build_site_ifc("IFC4", model, BASE_POINT)
    terrain = f.by_type("IfcGeographicElement")
    assert len(terrain) == 1
    assert terrain[0].PredefinedType == "TERRAIN"
