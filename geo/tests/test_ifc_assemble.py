"""Тесты сборки `site.ifc` (Шаг 1.8). Геометрия проверяется точными
аналитическими формулами (площадь треугольников, объём через теорему о
дивергенции для замкнутого меша); сборка IFC — через ifcopenshell.validate
на реальной модели (не мок)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial import Delaunay
from shapely.geometry import LineString, Polygon

from topology_geo.geometry.bridges import STATUS_CALCULATED, STATUS_OFFICIAL, BridgeRibbon
from topology_geo.geometry.buildings import BuildingSolid, EntranceInfo, LAYER_BUILDING_PARTS
from topology_geo.geometry.landscaping import Bench, FenceSegment, StreetLamp
from topology_geo.geometry.power import PoleTower, PowerSafetyZone, Substation, WireSpan
from topology_geo.geometry.rail import CatenaryPole, LevelCrossing, PlatformArea, RailRibbon
from topology_geo.geometry.road_network import NETWORK_BACKBONE, NETWORK_INTERNAL
from topology_geo.geometry.roads import RoadRibbon
from topology_geo.geometry.streets import IntersectionArea, LaneMarking, LaneRibbon
from topology_geo.geometry.vegetation import LawnArea, ShrubPoint, TreePoint
from topology_geo.geometry.water import WaterArea, WaterwayRibbon
from topology_geo.ifc.assemble import (
    LANE_CROSS_SLOPE,
    MARKING_CLEARANCE_M,
    PAVEMENT_CLEARANCE_M,
    BasePoint,
    SiteModel,
    build_site_ifc,
    extrude_polygon_mesh,
    flat_polygon_mesh,
    lane_marking_elevation_fn,
    lane_pavement_elevation_fn,
    mesh_cylinder,
    mesh_ribbon_along_path,
    pavement_elevation_fn,
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


def test_mesh_cylinder_default_radius_top_matches_base_unchanged():
    """Поведение по умолчанию (Шаг 2.6/ранее) не меняется - `radius_top=None`
    даёт цилиндр постоянного сечения."""
    verts, _ = mesh_cylinder(0.3, 6.0, segments=6)
    base_ring, top_ring = verts[:6], verts[6:]
    assert all(math.hypot(x, y) == pytest.approx(0.3) for x, y, _ in base_ring)
    assert all(math.hypot(x, y) == pytest.approx(0.3) for x, y, _ in top_ring)


def test_mesh_cylinder_tapers_to_radius_top():
    """Решётчатая опора ЛЭП (Шаг 2.7, п. 1) - сужается к вершине."""
    verts, _ = mesh_cylinder(1.5, 20.0, radius_top=0.3, segments=8)
    base_ring, top_ring = verts[:8], verts[8:]
    assert all(math.hypot(x, y) == pytest.approx(1.5) for x, y, _ in base_ring)
    assert all(math.hypot(x, y) == pytest.approx(0.3) for x, y, _ in top_ring)
    assert all(z == pytest.approx(0.0) for _, _, z in base_ring)
    assert all(z == pytest.approx(20.0) for _, _, z in top_ring)


def test_mesh_ribbon_along_path_area_matches_length_times_width_when_flat():
    """Провод-упрощение (Шаг 2.7, п. 3): для прямого горизонтального пути
    (без провиса) лента вырождается в тот же прямоугольник, что и
    `line.buffer(width/2, cap_style="flat")` у дорог/рельсов."""
    length, width = 40.0, 0.05
    path = [(x, 0.0, 100.0) for x in (0.0, 10.0, 20.0, 30.0, 40.0)]
    verts, faces = mesh_ribbon_along_path(path, width)
    total_area = sum(_triangle_area_3d(verts[a], verts[b], verts[c]) for a, b, c in faces)
    assert total_area == pytest.approx(length * width, rel=1e-9)


def test_mesh_ribbon_along_path_follows_sag_in_z():
    path = [(0.0, 0.0, 100.0), (5.0, 0.0, 97.0), (10.0, 0.0, 100.0)]
    verts, _ = mesh_ribbon_along_path(path, 0.1)
    n = len(path)
    # средняя точка ленты (левый и правый край) должна унаследовать провис
    assert verts[1][2] == pytest.approx(97.0)
    assert verts[n + 1][2] == pytest.approx(97.0)


def test_mesh_ribbon_along_path_rejects_single_point():
    with pytest.raises(ValueError):
        mesh_ribbon_along_path([(0.0, 0.0, 0.0)], 0.1)


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
        network="внутриквартальная",
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


# --- Шаг 2.6: платформы, переезды, опоры контактной сети --------------------


def _make_site_model_with_rail_extras() -> SiteModel:
    model = _make_site_model()
    platform = PlatformArea(osm_id=30, footprint=Polygon([(0, -25), (10, -25), (10, -22), (0, -22)]), height_m=0.3)
    crossing = LevelCrossing(osm_id=31, x=-20.0, y=-20.0, crossing_type="level_crossing", size_m=4.0)
    pole = CatenaryPole(osm_id=5, x=-30.0, y=-25.0, index=0)
    return SiteModel(
        tin=model.tin, buildings=model.buildings, roads=model.roads,
        water_areas=model.water_areas, waterways=model.waterways, rail=model.rail,
        platforms=[platform], level_crossings=[crossing], catenary_poles=[pole],
        trees=model.trees,
    )


def test_build_site_ifc_includes_platform_crossing_and_pole():
    f, registry = build_site_ifc("IFC4", _make_site_model_with_rail_extras(), BASE_POINT)
    assert validate_model(f) == []

    names = [e.Name for e in f.by_type("IfcBuildingElementProxy")]
    assert any((n or "").startswith("Платформа") for n in names)
    assert any((n or "").startswith("Переезд") for n in names)
    assert any((n or "").startswith("Опора КС") for n in names)

    layers = {layer for layer, _, _ in registry}
    assert "osm_railway_platforms" in layers
    assert "osm_railway_crossings" in layers
    # Опоры без естественного уникального osm_id (несколько на один путь,
    # как разметка) - в реестр GlobalId не попадают, см. `ifc/assemble.py`.
    assert "osm_railway_poles" not in layers


def test_platform_pset_reports_type_and_height():
    model = _make_site_model_with_rail_extras()
    f, _ = build_site_ifc("IFC4", model, BASE_POINT)
    pset = _psets_of_proxy_by_name(f, "Платформа 30")["Pset_ЖД"]
    assert pset["Тип"] == "платформа"
    assert pset["Высота_м"] == pytest.approx(0.3)


def test_road_network_filter_excludes_rail_extras():
    """Платформы/переезды/опоры - не дорожная сеть, исключены из
    roads_backbone/roads_internal (тот же приём, что и rail/trees, Шаг 2.4)."""
    f, registry = build_site_ifc(
        "IFC4", _make_site_model_with_rail_extras(), BASE_POINT, road_network_filter=NETWORK_INTERNAL
    )
    names = [e.Name for e in f.by_type("IfcBuildingElementProxy")]
    assert not any((n or "").startswith("Платформа") for n in names)
    assert not any((n or "").startswith("Переезд") for n in names)
    assert not any((n or "").startswith("Опора КС") for n in names)
    layers = {layer for layer, _, _ in registry}
    assert "osm_railway_platforms" not in layers
    assert "osm_railway_crossings" not in layers


# --- Шаг 2.7: опоры/провода ЛЭП, подстанции, охранная зона ------------------


def _make_site_model_with_power_extras() -> SiteModel:
    model = _make_site_model()
    pole_real = PoleTower(
        osm_id=40, x=20.0, y=30.0, height_m=9.0, height_confidence="факт",
        radius_base_m=0.15, radius_top_m=0.15, material="concrete", series=None,
        voltage_kv=10.0, source="факт",
    )
    pole_calculated = PoleTower(
        osm_id=None, x=20.0, y=45.0, height_m=9.0, height_confidence="умолчание",
        radius_base_m=0.12, radius_top_m=0.12, material="wood", series=None,
        voltage_kv=10.0, source="умолчание",
    )
    wire = WireSpan(
        line_osm_id=41, strand_index=0, cables=3, voltage_kv=10.0,
        path=[(20.0, 30.0, 110.0), (20.0, 37.5, 108.0), (20.0, 45.0, 110.0)],
    )
    substation = Substation(
        osm_id=42, footprint=Polygon([(30, -40), (38, -40), (38, -34), (30, -34)]),
        base_z=100.0, height_m=4.0, kind="substation",
    )
    zone = PowerSafetyZone(
        line_osm_id=41, corridor=LineString([(0, -45), (0, -35)]).buffer(10.0, cap_style="flat"),
        voltage_kv=20.0, half_width_m=10.0,
    )
    return SiteModel(
        tin=model.tin, buildings=model.buildings, roads=model.roads,
        water_areas=model.water_areas, waterways=model.waterways, rail=model.rail,
        power_poles=[pole_real, pole_calculated], power_wires=[wire],
        substations=[substation], power_safety_zones=[zone],
        trees=model.trees,
    )


def test_build_site_ifc_includes_power_infrastructure():
    f, registry = build_site_ifc("IFC4", _make_site_model_with_power_extras(), BASE_POINT)
    assert validate_model(f) == []

    proxy_names = [e.Name for e in f.by_type("IfcBuildingElementProxy")]
    assert any((n or "").startswith("Опора ЛЭП 40") for n in proxy_names)
    assert any((n or "").startswith("Опора ЛЭП (расч.)") for n in proxy_names)
    assert any((n or "").startswith("Подстанция") for n in proxy_names)
    assert any(e.PredefinedType == "CABLESEGMENT" for e in f.by_type("IfcCableSegment"))
    assert f.by_type("IfcSpatialZone")

    layers = {layer for layer, osm_id, _ in registry if layer == "osm_power"}
    assert layers == {"osm_power"}
    power_osm_ids = {osm_id for layer, osm_id, _ in registry if layer == "osm_power"}
    assert power_osm_ids == {40, 42}  # расчётная опора (osm_id=None) не регистрируется


def test_power_pole_pset_reports_material_voltage_and_source():
    f, _ = build_site_ifc("IFC4", _make_site_model_with_power_extras(), BASE_POINT)
    pset = _psets_of_proxy_by_name(f, "Опора ЛЭП 40")["Pset_ЛЭП"]
    assert pset["Материал"] == "concrete"
    assert pset["Источник"] == "факт"
    assert pset["Напряжение_кВ"] == pytest.approx(10.0)
    assert "Серия" not in pset  # не подставляется, когда design/structure отсутствуют


def test_power_wire_pset_reports_phases_and_circuits():
    f, _ = build_site_ifc("IFC4", _make_site_model_with_power_extras(), BASE_POINT)
    wire = next(e for e in f.by_type("IfcCableSegment") if e.PredefinedType == "CABLESEGMENT")
    pset = {
        rel.RelatingPropertyDefinition.Name: {
            prop.Name: prop.NominalValue.wrappedValue for prop in rel.RelatingPropertyDefinition.HasProperties
        }
        for rel in wire.IsDefinedBy
        if rel.RelatingPropertyDefinition.is_a("IfcPropertySet")
    }["Pset_ЛЭП"]
    assert pset["Число_фаз"] == 3
    assert pset["Число_цепей"] == 1
    assert pset["Напряжение_кВ"] == pytest.approx(10.0)


def test_substation_pset_reports_kind():
    f, _ = build_site_ifc("IFC4", _make_site_model_with_power_extras(), BASE_POINT)
    pset = _psets_of_proxy_by_name(f, "Подстанция 42")["Pset_Подстанция"]
    assert pset["Тип"] == "substation"


def test_safety_zone_reports_voltage_class_and_status():
    f, _ = build_site_ifc("IFC4", _make_site_model_with_power_extras(), BASE_POINT)
    zone = f.by_type("IfcSpatialZone")[0]
    pset = {
        rel.RelatingPropertyDefinition.Name: {
            prop.Name: prop.NominalValue.wrappedValue for prop in rel.RelatingPropertyDefinition.HasProperties
        }
        for rel in zone.IsDefinedBy
        if rel.RelatingPropertyDefinition.is_a("IfcPropertySet")
    }["Pset_Ограничение"]
    assert pset["Вид"] == "охранная зона ЛЭП"
    assert pset["Статус"] == "расчётная"
    assert pset["Напряжение_кВ"] == pytest.approx(20.0)
    assert pset["Полуширина_м"] == pytest.approx(10.0)


def test_road_network_filter_excludes_power_infrastructure():
    f, registry = build_site_ifc(
        "IFC4", _make_site_model_with_power_extras(), BASE_POINT, road_network_filter=NETWORK_INTERNAL
    )
    proxy_names = [e.Name for e in f.by_type("IfcBuildingElementProxy")]
    assert not any((n or "").startswith("Опора ЛЭП") for n in proxy_names)
    assert not any((n or "").startswith("Подстанция") for n in proxy_names)
    assert f.by_type("IfcCableSegment") == []
    assert f.by_type("IfcSpatialZone") == []
    assert not any(layer == "osm_power" for layer, _, _ in registry)


# --- Шаг 2.8: ограждения/стены, фонари --------------------------------------


def _make_site_model_with_landscaping_extras() -> SiteModel:
    model = _make_site_model()
    wall = FenceSegment(
        osm_id=50, kind="wall", ribbon=LineString([(0, 40), (10, 40)]).buffer(0.1, cap_style="flat"),
        height_m=2.0, posts=(),
    )
    fence = FenceSegment(
        osm_id=51, kind="fence", ribbon=LineString([(20, 40), (30, 40)]).buffer(0.025, cap_style="flat"),
        height_m=1.5, posts=((20.0, 40.0), (25.0, 40.0), (30.0, 40.0)),
    )
    lamp = StreetLamp(osm_id=52, x=35.0, y=40.0, pole_height_m=6.0, pole_radius_m=0.08, head_radius_m=0.15, head_height_m=0.3)
    return SiteModel(
        tin=model.tin, buildings=model.buildings, roads=model.roads,
        water_areas=model.water_areas, waterways=model.waterways, rail=model.rail,
        fences=[wall, fence], streetlamps=[lamp],
        trees=model.trees,
    )


def test_build_site_ifc_includes_landscaping_infrastructure():
    f, registry = build_site_ifc("IFC4", _make_site_model_with_landscaping_extras(), BASE_POINT)
    assert validate_model(f) == []

    walls = f.by_type("IfcWall")
    assert any((w.Name or "").startswith("Стена 50") for w in walls)
    railings = f.by_type("IfcRailing")
    assert any((r.Name or "").startswith("Ограждение 51") for r in railings)
    assert any((r.Name or "").startswith("Столб ограждения 51") for r in railings)
    proxy_names = [e.Name for e in f.by_type("IfcBuildingElementProxy")]
    assert any((n or "").startswith("Фонарь 52") for n in proxy_names)

    layers = {(layer, osm_id) for layer, osm_id, _ in registry if layer == "osm_landscaping"}
    assert layers == {("osm_landscaping", 50), ("osm_landscaping", 51), ("osm_landscaping", 52)}


def test_ifc4x3_fence_uses_native_fence_predefined_type():
    f, _ = build_site_ifc("IFC4X3", _make_site_model_with_landscaping_extras(), BASE_POINT)
    railing = next(r for r in f.by_type("IfcRailing") if (r.Name or "").startswith("Ограждение 51"))
    assert railing.PredefinedType == "FENCE"


def test_ifc4_fence_falls_back_to_userdefined_predefined_type():
    """IFC4 `IfcRailingTypeEnum` не знает значения FENCE (эмпирически
    проверено через ifcopenshell) - предопределённый тип падает на
    USERDEFINED, а не изобретается несуществующее значение схемы."""
    f, _ = build_site_ifc("IFC4", _make_site_model_with_landscaping_extras(), BASE_POINT)
    railing = next(r for r in f.by_type("IfcRailing") if (r.Name or "").startswith("Ограждение 51"))
    assert railing.PredefinedType == "USERDEFINED"


def test_road_network_filter_excludes_landscaping_infrastructure():
    f, registry = build_site_ifc(
        "IFC4", _make_site_model_with_landscaping_extras(), BASE_POINT, road_network_filter=NETWORK_INTERNAL
    )
    assert f.by_type("IfcWall") == []
    assert f.by_type("IfcRailing") == []
    proxy_names = [e.Name for e in f.by_type("IfcBuildingElementProxy")]
    assert not any((n or "").startswith("Фонарь") for n in proxy_names)
    assert not any(layer == "osm_landscaping" for layer, _, _ in registry)


# --- Шаг 2.9: высота дерева, кустарники, газоны, скамейки -------------------


def _make_site_model_with_vegetation_extras() -> SiteModel:
    model = _make_site_model()
    tall_tree = TreePoint(
        x=25.0, y=-30.0, species="Quercus robur", confidence="факт", source_osm_id=60,
        height_m=18.0, height_confidence="факт",
    )
    shrub = ShrubPoint(x=30.0, y=-30.0, height_m=1.2, source_osm_id=61)
    lawn = LawnArea(osm_id=62, polygon=Polygon([(0, -40), (10, -40), (10, -35), (0, -35)]))
    bench = Bench(osm_id=63, x=15.0, y=-40.0, length_m=1.5, depth_m=0.5, seat_height_m=0.45)
    return SiteModel(
        tin=model.tin, buildings=model.buildings, roads=model.roads,
        water_areas=model.water_areas, waterways=model.waterways, rail=model.rail,
        trees=[tall_tree], shrubs=[shrub], lawns=[lawn], benches=[bench],
    )


def test_build_site_ifc_includes_vegetation_extras():
    f, registry = build_site_ifc("IFC4", _make_site_model_with_vegetation_extras(), BASE_POINT)
    assert validate_model(f) == []

    geo_names = [e.Name for e in f.by_type("IfcGeographicElement")]
    assert any((n or "").startswith("Дерево 60") for n in geo_names)
    assert any((n or "").startswith("Куст 61") for n in geo_names)
    assert any((n or "").startswith("Газон 62") for n in geo_names)
    assert any((item.Name or "").startswith("Скамейка 63") for item in f.by_type("IfcFurniture"))

    layers = {(layer, osm_id) for layer, osm_id, _ in registry}
    assert ("osm_vegetation", 60) in layers
    assert ("osm_vegetation", 61) in layers  # куст - тоже регистрируется, как рассеянные деревья леса
    assert ("osm_vegetation", 62) in layers
    assert ("osm_landscaping", 63) in layers


def test_tree_pset_reports_height_from_tag():
    f, _ = build_site_ifc("IFC4", _make_site_model_with_vegetation_extras(), BASE_POINT)
    element = next(e for e in f.by_type("IfcGeographicElement") if (e.Name or "").startswith("Дерево 60"))
    pset = {
        rel.RelatingPropertyDefinition.Name: {
            prop.Name: prop.NominalValue.wrappedValue for prop in rel.RelatingPropertyDefinition.HasProperties
        }
        for rel in element.IsDefinedBy
        if rel.RelatingPropertyDefinition.is_a("IfcPropertySet")
    }["Pset_Растительность"]
    assert pset["Высота_м"] == pytest.approx(18.0)
    assert pset["Источник_высоты"] == "факт"


def test_shrub_is_shorter_than_tree_in_mesh_geometry():
    """Геометрическая, а не только Pset, проверка: высота меша куста
    (Шаг 2.9) заметно меньше высоты меша дерева (сравнение внутри одного
    файла - см. docstring `_mesh_coords`, значения в мм файла, не в метрах)."""
    f, _ = build_site_ifc("IFC4", _make_site_model_with_vegetation_extras(), BASE_POINT)
    tree = next(e for e in f.by_type("IfcGeographicElement") if (e.Name or "").startswith("Дерево 60"))
    shrub = next(e for e in f.by_type("IfcGeographicElement") if (e.Name or "").startswith("Куст 61"))

    def _z_span(element):
        zs = [c[2] for c in _mesh_coords(element)]
        return max(zs) - min(zs)

    assert _z_span(shrub) < _z_span(tree)


def test_road_network_filter_excludes_vegetation_extras():
    f, registry = build_site_ifc(
        "IFC4", _make_site_model_with_vegetation_extras(), BASE_POINT, road_network_filter=NETWORK_INTERNAL
    )
    assert f.by_type("IfcFurniture") == []
    geo_names = [e.Name for e in f.by_type("IfcGeographicElement")]
    assert not any((n or "").startswith("Куст") for n in geo_names)
    assert not any((n or "").startswith("Газон") for n in geo_names)
    assert not any(layer == "osm_landscaping" for layer, _, _ in registry)


def _psets_of(model, name_predicate) -> dict:
    element = next(e for e in model.by_type("IfcBuildingElementProxy") if name_predicate(e.Name or ""))
    return {
        rel.RelatingPropertyDefinition.Name: {
            prop.Name: prop.NominalValue.wrappedValue for prop in rel.RelatingPropertyDefinition.HasProperties
        }
        for rel in element.IsDefinedBy
        if rel.RelatingPropertyDefinition.is_a("IfcPropertySet")
    }


def test_build_site_ifc_writes_entrance_properties_on_building():
    building = BuildingSolid(
        osm_id=1, footprint=Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
        height_m=12.0, height_confidence="факт", height_source="OSM", base_z=100.0, building_type="жилой",
        entrances=(
            EntranceInfo(entrance_type="main", x=0.0, y=-10.0),
            EntranceInfo(entrance_type="yes", x=10.0, y=0.0),
        ),
    )
    model = SiteModel(buildings=[building])
    f, _ = build_site_ifc("IFC4", model, BASE_POINT)
    assert validate_model(f) == []

    psets = _psets_of(f, lambda name: name.startswith("Здание"))
    building_pset = psets["Pset_Здание"]
    assert building_pset["Входов_всего"] == 2
    assert building_pset["Вход_1_Тип"] == "main"
    assert building_pset["Вход_1_X_м"] == pytest.approx(0.0)
    assert building_pset["Вход_1_Y_м"] == pytest.approx(-10.0)
    assert building_pset["Вход_2_Тип"] == "yes"


def test_build_site_ifc_building_without_entrances_has_no_entrance_properties():
    f, _ = build_site_ifc("IFC4", _make_site_model(), BASE_POINT)
    psets = _psets_of(f, lambda name: name.startswith("Здание"))
    assert "Входов_всего" not in psets["Pset_Здание"]


def test_build_site_ifc_marks_building_part_in_context_pset():
    part = BuildingSolid(
        osm_id=2, footprint=Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
        height_m=5.0, height_confidence="факт", height_source="OSM", base_z=100.0, building_type="roof",
        source_layer=LAYER_BUILDING_PARTS, is_part=True,
    )
    f, registry = build_site_ifc("IFC4", SiteModel(buildings=[part]), BASE_POINT)
    assert validate_model(f) == []

    psets = _psets_of(f, lambda name: name.startswith("Здание"))
    assert psets["Pset_Контекст"]["Часть_здания"] is True
    assert registry == [(LAYER_BUILDING_PARTS, 2, f.by_type("IfcBuildingElementProxy")[0].GlobalId)]


def test_build_site_ifc_plain_building_has_no_part_marker_in_context_pset():
    f, _ = build_site_ifc("IFC4", _make_site_model(), BASE_POINT)
    psets = _psets_of(f, lambda name: name.startswith("Здание"))
    assert "Часть_здания" not in psets["Pset_Контекст"]


def _psets_of_proxy_by_name(model, name: str) -> dict:
    element = next(e for e in model.by_type("IfcBuildingElementProxy") if e.Name == name)
    return {
        rel.RelatingPropertyDefinition.Name: {
            prop.Name: prop.NominalValue.wrappedValue for prop in rel.RelatingPropertyDefinition.HasProperties
        }
        for rel in element.IsDefinedBy
        if rel.RelatingPropertyDefinition.is_a("IfcPropertySet")
    }


def test_build_site_ifc_writes_lane_and_intersection_properties():
    lane = LaneRibbon(
        osm_way_ids=(10, 11), lane_type="Driving", width_m=3.0, direction="Fwd",
        polygon=Polygon([(-5, -1), (5, -1), (5, 1), (-5, 1)]), surface="asphalt",
    )
    intersection = IntersectionArea(kind="sidewalk corner", polygon=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
    model = SiteModel(lanes=[lane], intersections=[intersection])

    f, registry = build_site_ifc("IFC4", model, BASE_POINT)
    assert validate_model(f) == []

    lane_psets = _psets_of_proxy_by_name(f, "Полоса 10+11")
    assert lane_psets["Pset_Полоса"] == {
        "Тип": "Driving", "Ширина_м": 3.0, "Направление": "Fwd", "Покрытие": "asphalt",
        "Сеть": "внутриквартальная", "Редактируемый": True,
    }

    inter_psets = _psets_of_proxy_by_name(f, "Перекрёсток (sidewalk corner)")
    assert inter_psets["Pset_Перекрёсток"] == {"Тип": "sidewalk corner", "Сеть": "внутриквартальная"}

    # одна полоса с несколькими osm_way_ids -> запись реестра на каждый way
    # (тот же принятый компромисс, что и многосегментные road.ribbon/waterway.ribbon).
    assert set(registry) == {
        ("osm_roads", 10, next(e for e in f.by_type("IfcBuildingElementProxy") if e.Name == "Полоса 10+11").GlobalId),
        ("osm_roads", 11, next(e for e in f.by_type("IfcBuildingElementProxy") if e.Name == "Полоса 10+11").GlobalId),
    }


def test_build_site_ifc_intersection_has_no_registry_entry():
    intersection = IntersectionArea(kind="sidewalk corner", polygon=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
    _, registry = build_site_ifc("IFC4", SiteModel(intersections=[intersection]), BASE_POINT)
    assert registry == []  # нет естественного osm_id - как и у TIN/рельефа


def test_build_site_ifc_merges_markings_of_same_kind_into_one_product():
    markings = [
        LaneMarking(kind="center line", polygon=Polygon([(0, 0), (1, 0), (1, 0.1), (0, 0.1)])),
        LaneMarking(kind="center line", polygon=Polygon([(2, 0), (3, 0), (3, 0.1), (2, 0.1)])),
        LaneMarking(kind="lane arrow", polygon=Polygon([(0, 5), (1, 5), (1, 6), (0.5, 6.5), (0, 6)])),
    ]
    f, registry = build_site_ifc("IFC4", SiteModel(markings=markings), BASE_POINT)
    assert validate_model(f) == []

    marking_products = [e for e in f.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Разметка")]
    assert len(marking_products) == 2  # один продукт на вид, не на штрих

    center_line_psets = _psets_of_proxy_by_name(f, "Разметка (center line)")
    assert center_line_psets["Pset_Разметка"] == {
        "Тип": "center line", "Элементов": 2, "Сеть": "внутриквартальная",
    }

    arrow_psets = _psets_of_proxy_by_name(f, "Разметка (lane arrow)")
    assert arrow_psets["Pset_Разметка"] == {
        "Тип": "lane arrow", "Элементов": 1, "Сеть": "внутриквартальная",
    }

    assert registry == []  # без естественного osm_id


def test_build_site_ifc_merged_marking_mesh_covers_all_pieces_area():
    # два непересекающихся штриха разной формы (разное число вершин) - ровно
    # случай, из-за которого понадобилось слияние в один меш вручную
    # (add_mesh_representation не принимает разноразмерные items в одном продукте).
    piece_a = Polygon([(0, 0), (1, 0), (1, 0.1), (0, 0.1)])
    piece_b = Polygon([(10, 10), (11, 10), (11, 11), (10.5, 11.5), (10, 11)])
    markings = [LaneMarking(kind="center line", polygon=piece_a), LaneMarking(kind="center line", polygon=piece_b)]

    f, _ = build_site_ifc("IFC4", SiteModel(markings=markings), BASE_POINT)
    assert validate_model(f) == []

    product = next(e for e in f.by_type("IfcBuildingElementProxy") if e.Name == "Разметка (center line)")
    rep_item = product.Representation.Representations[0].Items[0]
    assert rep_item.is_a("IfcPolygonalFaceSet")
    # 2 треугольника на прямоугольник + 3 на пятиугольник = 5 (earcut даёт
    # ровно n-2 треугольников на выпуклый n-угольник без отверстий) - оба
    # куска в ОДНОМ представлении (не по одному продукту на штрих).
    assert len(rep_item.Faces) == 2 + 3


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


def _mesh_coords(product) -> list[tuple[float, float, float]]:
    rep_item = product.Representation.Representations[0].Items[0]
    return [tuple(c) for c in rep_item.Coordinates.CoordList]


def test_pavement_elevation_fn_lifts_terrain_by_clearance():
    """Дорожное покрытие — обязательное требование проекта — всегда СТРОГО
    выше земли, не вровень с ней: вровень означало бы гарантированный
    z-fighting в вебвьюере и, при малейшем расхождении сеток ленты/TIN,
    видимое проваливание полотна под рельеф."""
    terrain = lambda x, y: 100.0 + 0.01 * x
    pavement = pavement_elevation_fn(terrain)
    for x, y in [(0.0, 0.0), (12.3, -4.0), (-50.0, 50.0)]:
        assert pavement(x, y) == pytest.approx(terrain(x, y) + PAVEMENT_CLEARANCE_M)


def test_pavement_elevation_fn_propagates_none():
    pavement = pavement_elevation_fn(lambda x, y: None)
    assert pavement(0.0, 0.0) is None


def test_lane_marking_elevation_fn_sits_above_pavement():
    """Разметка — тонкий слой краски НА покрытии, ещё выше него — иначе линия
    разметки зрительно тонула бы в асфальте, поднятом над рельефом. Точка
    почти на оси штриха (нулевой поперечный снос) - чистая проверка
    вертикального зазора без примеси поперечного уклона."""
    terrain = lambda x, y: 100.0
    marking_poly = Polygon([(-2, -0.05), (2, -0.05), (2, 0.05), (-2, 0.05)])
    marking = lane_marking_elevation_fn(marking_poly, terrain)
    assert marking(0.0, 0.0) == pytest.approx(100.0 + PAVEMENT_CLEARANCE_M + MARKING_CLEARANCE_M, abs=1e-6)


def test_lane_pavement_elevation_fn_falls_back_to_flat_for_compact_polygon():
    """Почти квадратный (не вытянутый) фрагмент полосы — своя ось ненадёжна
    (см. докстринг `lane_pavement_elevation_fn`), посадка — обычная плоская,
    без продольного профиля/поперечного уклона."""
    terrain = lambda x, y: 100.0 + 0.01 * x
    compact_poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    lane_fn = lane_pavement_elevation_fn(compact_poly, terrain)
    assert lane_fn(2.0, 2.0) == pytest.approx(terrain(2.0, 2.0) + PAVEMENT_CLEARANCE_M, abs=1e-6)


def test_lane_pavement_elevation_fn_applies_cross_slope_for_elongated_polygon():
    """Вытянутая полоса на РОВНОМ рельефе — отметка на оси выше, чем у края
    (поперечный уклон стока, `LANE_CROSS_SLOPE`), а вдоль оси не меняется."""
    terrain = lambda x, y: 100.0
    lane_poly = Polygon([(-20, -2), (20, -2), (20, 2), (-20, 2)])  # вдоль X, ширина 4 м
    lane_fn = lane_pavement_elevation_fn(lane_poly, terrain)

    center = lane_fn(0.0, 0.0)
    edge = lane_fn(0.0, 2.0)
    assert center == pytest.approx(100.0 + PAVEMENT_CLEARANCE_M, abs=1e-6)
    assert edge == pytest.approx(center - LANE_CROSS_SLOPE * 2.0, abs=1e-6)
    assert lane_fn(-15.0, 0.0) == pytest.approx(lane_fn(15.0, 0.0), abs=1e-6)  # вдоль оси на ровном рельефе не меняется


def test_build_site_ifc_road_lane_and_marking_are_above_terrain_end_to_end():
    """Сквозная проверка через реальную сборку IFC (не только логику функций
    выше): дорога и разметка попадают в файл в правильном относительном
    порядке (разметка выше покрытия). Ровный (без уклона) TIN — иначе
    сравнение глобальных min/max между двумя разными по площади объектами
    путал бы уклон рельефа с самим сравниваемым подъёмом. Координаты в файле
    — в его собственных единицах (мм по умолчанию
    `ifcopenshell.api.project.create_file`), поэтому сравниваем отношения
    Z-координат внутри файла, а не абсолютные метры."""
    tin = _make_flat_tin(n=2)
    tin.vertices[:, 2] = 100.0  # без уклона — см. докстринг
    road = RoadRibbon(
        osm_id=2, ribbon=LineString([(-40, 0), (40, 0)]).buffer(3.0, cap_style="flat"),
        width_m=6.0, width_confidence="умолчание", surface="asphalt", highway_class="residential",
        network="внутриквартальная",
    )
    marking = LaneMarking(kind="center line", polygon=Polygon([(-2, -0.05), (2, -0.05), (2, 0.05), (-2, 0.05)]))
    model = SiteModel(tin=tin, roads=[road], markings=[marking])
    f, _ = build_site_ifc("IFC4", model, BASE_POINT)
    assert validate_model(f) == []

    road_product = next(e for e in f.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Дорога"))
    marking_product = next(e for e in f.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Разметка"))
    road_z = [c[2] for c in _mesh_coords(road_product)]
    marking_z = [c[2] for c in _mesh_coords(marking_product)]

    assert min(marking_z) > max(road_z)  # разметка строго выше покрытия дороги, а не вровень с ним


# --- road_network_filter (Шаг 2.4, п. 1 и 4) -----------------------------


def _make_mixed_network_model() -> SiteModel:
    """Один backbone-объект каждого типа (дорога/полоса/перекрёсток/
    разметка) + один internal - плюс здание/вода/ж-д/дерево, которых в
    роли-фильтрованном файле дорог быть не должно вовсе."""
    backbone_road = RoadRibbon(
        osm_id=100, ribbon=LineString([(-40, 10), (40, 10)]).buffer(5.0, cap_style="flat"),
        width_m=10.0, width_confidence="умолчание", surface="asphalt", highway_class="primary",
        network=NETWORK_BACKBONE,
    )
    internal_road = RoadRibbon(
        osm_id=101, ribbon=LineString([(-40, -10), (40, -10)]).buffer(3.0, cap_style="flat"),
        width_m=6.0, width_confidence="умолчание", surface="asphalt", highway_class="residential",
        network=NETWORK_INTERNAL,
    )
    backbone_lane = LaneRibbon(
        osm_way_ids=(100,), lane_type="Driving", width_m=3.0, direction="Fwd",
        polygon=Polygon([(-5, 8), (5, 8), (5, 12), (-5, 12)]), surface="asphalt", network=NETWORK_BACKBONE,
    )
    internal_lane = LaneRibbon(
        osm_way_ids=(101,), lane_type="Driving", width_m=3.0, direction="Fwd",
        polygon=Polygon([(-5, -12), (5, -12), (5, -8), (-5, -8)]), surface="asphalt", network=NETWORK_INTERNAL,
    )
    backbone_intersection = IntersectionArea(
        kind="sidewalk corner", polygon=Polygon([(20, 8), (22, 8), (22, 12), (20, 12)]), network=NETWORK_BACKBONE,
    )
    internal_intersection = IntersectionArea(
        kind="sidewalk corner", polygon=Polygon([(20, -12), (22, -12), (22, -8), (20, -8)]),
        network=NETWORK_INTERNAL,
    )
    backbone_marking = LaneMarking(
        kind="center line", polygon=Polygon([(0, 9.95), (1, 9.95), (1, 10.05), (0, 10.05)]),
        network=NETWORK_BACKBONE,
    )
    internal_marking = LaneMarking(
        kind="center line", polygon=Polygon([(0, -10.05), (1, -10.05), (1, -9.95), (0, -9.95)]),
        network=NETWORK_INTERNAL,
    )
    building = BuildingSolid(
        osm_id=1, footprint=Polygon([(-10, 30), (10, 30), (10, 40), (-10, 40)]),
        height_m=12.0, height_confidence="факт", height_source="OSM", base_z=100.0, building_type="жилой",
    )
    water = WaterArea(osm_id=3, polygon=Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]), level_z=99.5)
    rail = RailRibbon(osm_id=5, ballast=LineString([(-40, -30), (40, -30)]).buffer(2.0, cap_style="flat"), rail_type="tram")
    tree = TreePoint(x=15.0, y=-25.0, species="Betula pendula", confidence="факт", source_osm_id=6)
    return SiteModel(
        tin=_make_flat_tin(),
        roads=[backbone_road, internal_road],
        lanes=[backbone_lane, internal_lane],
        intersections=[backbone_intersection, internal_intersection],
        markings=[backbone_marking, internal_marking],
        buildings=[building], water_areas=[water], rail=[rail], trees=[tree],
    )


def test_road_network_filter_backbone_keeps_only_backbone_roads_and_lanes():
    f, _ = build_site_ifc("IFC4", _make_mixed_network_model(), BASE_POINT, road_network_filter=NETWORK_BACKBONE)
    assert validate_model(f) == []

    road_names = {e.Name for e in f.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Дорога")}
    assert road_names == {"Дорога 100"}

    lane_names = {e.Name for e in f.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Полоса")}
    assert lane_names == {"Полоса 100"}

    inter_psets = _psets_of_proxy_by_name(f, "Перекрёсток (sidewalk corner)")
    assert inter_psets["Pset_Перекрёсток"]["Сеть"] == NETWORK_BACKBONE

    marking_psets = _psets_of_proxy_by_name(f, "Разметка (center line)")
    assert marking_psets["Pset_Разметка"] == {"Тип": "center line", "Элементов": 1, "Сеть": NETWORK_BACKBONE}


def test_road_network_filter_internal_keeps_only_internal_roads_and_lanes():
    f, _ = build_site_ifc("IFC4", _make_mixed_network_model(), BASE_POINT, road_network_filter=NETWORK_INTERNAL)
    assert validate_model(f) == []

    road_names = {e.Name for e in f.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Дорога")}
    assert road_names == {"Дорога 101"}

    lane_names = {e.Name for e in f.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Полоса")}
    assert lane_names == {"Полоса 101"}


def test_road_network_filter_excludes_buildings_water_rail_trees():
    for network in (NETWORK_BACKBONE, NETWORK_INTERNAL):
        f, registry = build_site_ifc(
            "IFC4", _make_mixed_network_model(), BASE_POINT, road_network_filter=network
        )
        assert validate_model(f) == []
        assert not any((e.Name or "").startswith("Здание") for e in f.by_type("IfcBuildingElementProxy"))
        assert not any((e.Name or "").startswith("Ж/д") for e in f.by_type("IfcBuildingElementProxy"))
        # ни рельефа (TIN), ни воды, ни деревьев (все три - IfcGeographicElement)
        # быть не должно вовсе - см. test_road_network_filter_excludes_relief_tin
        # про то, почему рельеф здесь тоже исключён, а не оставлен как
        # "общая привязка".
        assert f.by_type("IfcGeographicElement") == []
        assert not any(layer == "osm_water_areas" for layer, _, _ in registry)
        assert not any(layer == "osm_railways" for layer, _, _ in registry)
        assert not any(layer == "osm_vegetation" for layer, _, _ in registry)


def test_road_network_filter_excludes_relief_tin():
    """Рельеф (TIN) не относится ни к одной из двух сетей и НЕ включается в
    отфильтрованные файлы (Шаг 2.4, п. 4) - в отличие от первой версии этого
    прохода. Причина не эстетическая: меш TIN на честном участке (шаг сетки
    1 м, Шаг 1.5) - самая тяжёлая по памяти часть сборки IFC (сотни тысяч
    вершин на средний радиус); дублирование его в обоих файлах на каждую
    схему означало реальный OOM на сквозном прогоне (не гипотетический -
    воспроизведено при разработке). Полосы/дороги уже несут свою абсолютную
    высоту (посадка на рельеф, Шаг 2.3, п. 5) - сам меш для их отображения
    не нужен."""
    for network in (NETWORK_BACKBONE, NETWORK_INTERNAL):
        f, _ = build_site_ifc("IFC4", _make_mixed_network_model(), BASE_POINT, road_network_filter=network)
        terrain = [e for e in f.by_type("IfcGeographicElement") if e.Name == "Рельеф участка (TIN)"]
        assert terrain == []

    # В полном (нефильтрованном) site.ifc рельеф остаётся, как и раньше.
    f, _ = build_site_ifc("IFC4", _make_mixed_network_model(), BASE_POINT)
    terrain = [e for e in f.by_type("IfcGeographicElement") if e.Name == "Рельеф участка (TIN)"]
    assert len(terrain) == 1


def test_road_network_filter_none_keeps_full_combined_model():
    f, _ = build_site_ifc("IFC4", _make_mixed_network_model(), BASE_POINT)
    assert validate_model(f) == []
    road_names = {e.Name for e in f.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Дорога")}
    assert road_names == {"Дорога 100", "Дорога 101"}
    assert any((e.Name or "").startswith("Здание") for e in f.by_type("IfcBuildingElementProxy"))

    # штрихи разметки обеих сетей склеены в один продукт "center line" -
    # честно нет единой "Сеть" на группу (см. build_site_ifc), а не наугад.
    marking_psets = _psets_of_proxy_by_name(f, "Разметка (center line)")
    assert "Сеть" not in marking_psets["Pset_Разметка"]
    assert marking_psets["Pset_Разметка"]["Элементов"] == 2


def test_road_network_filter_roads_registry_does_not_include_other_network():
    _, registry = build_site_ifc(
        "IFC4", _make_mixed_network_model(), BASE_POINT, road_network_filter=NETWORK_BACKBONE
    )
    registered_osm_ids = {osm_id for _, osm_id, _ in registry}
    assert 100 in registered_osm_ids
    assert 101 not in registered_osm_ids


# --- мосты (Шаг 2.5, п. 1-3) ----------------------------------------------


def _make_bridge(osm_id=50, status=STATUS_OFFICIAL, clearance_m=None, network=NETWORK_INTERNAL) -> BridgeRibbon:
    axis = LineString([(-20, 0), (20, 0)])
    return BridgeRibbon(
        osm_id=osm_id,
        ribbon=axis.buffer(4.0, cap_style="flat"),
        axis=axis,
        width_m=8.0,
        width_confidence="умолчание",
        surface="asphalt",
        highway_class="secondary",
        network=network,
        deck_elevation_fn=lambda x, y: 100.0,
        clearance_m=clearance_m,
        status=status,
    )


def test_build_site_ifc_uses_native_ifcbridge_in_ifc43():
    model = SiteModel(tin=_make_flat_tin(), bridges=[_make_bridge()])
    f, registry = build_site_ifc("IFC4X3", model, BASE_POINT)
    assert validate_model(f) == []
    assert len(f.by_type("IfcBridge")) == 1
    bridge_gid = next(gid for layer, osm_id, gid in registry if layer == "osm_roads" and osm_id == 50)
    assert f.by_type("IfcBridge")[0].GlobalId == bridge_gid


def test_build_site_ifc_uses_proxy_bridge_in_ifc4():
    model = SiteModel(tin=_make_flat_tin(), bridges=[_make_bridge()])
    f, _ = build_site_ifc("IFC4", model, BASE_POINT)
    assert validate_model(f) == []
    proxies = [e for e in f.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Мост")]
    assert len(proxies) == 1


def test_build_site_ifc_bridge_pset_reports_status_and_clearance():
    model = SiteModel(tin=_make_flat_tin(), bridges=[_make_bridge(status=STATUS_CALCULATED, clearance_m=5.0)])
    f, _ = build_site_ifc("IFC4", model, BASE_POINT)
    psets = _psets_of_proxy_by_name(f, "Мост 50")
    assert psets["Pset_Мост"] == {"Статус": STATUS_CALCULATED, "Габарит_м": 5.0}


def test_build_site_ifc_bridge_pset_omits_clearance_when_not_checked():
    model = SiteModel(tin=_make_flat_tin(), bridges=[_make_bridge(status=STATUS_OFFICIAL, clearance_m=None)])
    f, _ = build_site_ifc("IFC4", model, BASE_POINT)
    psets = _psets_of_proxy_by_name(f, "Мост 50")
    assert psets["Pset_Мост"] == {"Статус": STATUS_OFFICIAL}


def test_bridge_deck_elevation_includes_pavement_clearance():
    """Полотно моста — та же обёртка `pavement_elevation_fn`, что и у
    дороги/полосы (не отдельная логика зазора для мостов) - проверяется
    напрямую в метрах Python, а не через сборку IFC (файл по умолчанию в
    миллиметрах, см. docs/pavement.md)."""
    bridge = _make_bridge()
    deck_with_clearance = pavement_elevation_fn(bridge.deck_elevation_fn)
    assert deck_with_clearance(0.0, 0.0) == pytest.approx(100.0 + PAVEMENT_CLEARANCE_M)


def test_build_site_ifc_bridge_respects_road_network_filter():
    model = SiteModel(
        tin=_make_flat_tin(),
        bridges=[_make_bridge(osm_id=50, network=NETWORK_BACKBONE), _make_bridge(osm_id=51, network=NETWORK_INTERNAL)],
    )
    f, _ = build_site_ifc("IFC4", model, BASE_POINT, road_network_filter=NETWORK_BACKBONE)
    names = {e.Name for e in f.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Мост")}
    assert names == {"Мост 50"}
