"""Тесты конвертации IFC -> GLB (Шаг 1.9, п. 1). Собираем реальный site.ifc
через assemble.build_site_ifc (Шаг 1.8), конвертируем и проверяем структуру
через настоящую загрузку `pygltflib.GLTF2.load_from_bytes` (не мок)."""

from __future__ import annotations

import numpy as np
import pytest
from pygltflib import GLTF2
from scipy.spatial import Delaunay
from shapely.geometry import LineString, Polygon

from topology_geo.geometry.buildings import BuildingSolid
from topology_geo.geometry.rail import RailRibbon
from topology_geo.geometry.roads import RoadRibbon
from topology_geo.geometry.vegetation import TreePoint
from topology_geo.geometry.water import WaterArea, WaterwayRibbon
from topology_geo.ifc.assemble import BasePoint, SiteModel, build_site_ifc
from topology_geo.ifc.to_glb import (
    CATEGORY_BUILDINGS,
    CATEGORY_RAIL,
    CATEGORY_ROADS,
    CATEGORY_TERRAIN,
    CATEGORY_TREES,
    CATEGORY_WATER,
    convert_ifc_to_glb,
)
from topology_geo.relief.tin import SiteTin

BASE_POINT = BasePoint(lon=56.25, lat=58.0, zone=2, x=2_310_450.0, y=-5_857_320.0, height=150.0)


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


def _scene_root(gltf: GLTF2):
    return gltf.nodes[gltf.scenes[gltf.scene].nodes[0]]


def _category_names(gltf: GLTF2) -> dict[str, list[str]]:
    root = _scene_root(gltf)
    return {
        gltf.nodes[cat_idx].name: [gltf.nodes[leaf].name for leaf in gltf.nodes[cat_idx].children]
        for cat_idx in root.children
    }


@pytest.mark.parametrize("schema", ["IFC4", "IFC4X3"])
def test_convert_ifc_to_glb_roundtrips_and_has_all_categories(schema):
    model, _ = build_site_ifc(schema, _make_site_model(), BASE_POINT)
    data = convert_ifc_to_glb(model)

    gltf = GLTF2.load_from_bytes(data)
    categories = _category_names(gltf)
    assert set(categories) == {
        CATEGORY_TERRAIN, CATEGORY_BUILDINGS, CATEGORY_ROADS,
        CATEGORY_WATER, CATEGORY_RAIL, CATEGORY_TREES,
    }
    assert categories[CATEGORY_BUILDINGS] == ["Здание 1"]
    assert categories[CATEGORY_WATER] == ["Водоём 3", "Водоток 4"]
    assert len(gltf.meshes) == len(gltf.accessors) // 2  # один меш = позиции + индексы


def test_convert_ifc_to_glb_node_extras_carry_globalid_class_and_psets():
    model, registry = build_site_ifc("IFC4", _make_site_model(), BASE_POINT)
    data = convert_ifc_to_glb(model)
    gltf = GLTF2.load_from_bytes(data)

    building_global_id = next(gid for layer, _, gid in registry if layer == "osm_buildings")
    building_node = next(n for n in gltf.nodes if n.extras and n.extras.get("globalId") == building_global_id)

    assert building_node.extras["ifcClass"] == "IfcBuildingElementProxy"
    assert building_node.extras["psets"]["Pset_Здание"]["Высота_м"] == pytest.approx(12.0)
    assert building_node.extras["psets"]["Pset_Здание"]["Источник_высоты"] == "OSM"
    assert "id" not in building_node.extras["psets"]["Pset_Здание"]


def test_convert_ifc_to_glb_mesh_geometry_matches_source_vertex_count():
    model, _ = build_site_ifc("IFC4", _make_site_model(), BASE_POINT)
    data = convert_ifc_to_glb(model)
    gltf = GLTF2.load_from_bytes(data)
    blob = gltf.binary_blob()

    tree_node = next(n for n in gltf.nodes if n.name and n.name.startswith("Дерево"))
    mesh = gltf.meshes[tree_node.mesh]
    pos_accessor = gltf.accessors[mesh.primitives[0].attributes.POSITION]
    idx_accessor = gltf.accessors[mesh.primitives[0].indices]

    pos_view = gltf.bufferViews[pos_accessor.bufferView]
    positions = np.frombuffer(blob, dtype=np.float32, count=pos_accessor.count * 3, offset=pos_view.byteOffset)
    assert positions.reshape(-1, 3).shape[0] == pos_accessor.count
    assert pos_accessor.count > 0
    assert idx_accessor.count % 3 == 0  # треугольники


def test_convert_ifc_to_glb_handles_empty_site_model():
    model, _ = build_site_ifc("IFC4", SiteModel(), BASE_POINT)
    data = convert_ifc_to_glb(model)
    gltf = GLTF2.load_from_bytes(data)
    root = _scene_root(gltf)
    assert root.children == []
    assert gltf.meshes == []
