"""Тесты сборки `site.ifc` (Шаг 1.8). Геометрия проверяется точными
аналитическими формулами (площадь треугольников, объём через теорему о
дивергенции для замкнутого меша); сборка IFC — через ifcopenshell.validate
на реальной модели (не мок)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import Delaunay
from shapely.geometry import LineString, Polygon

from topology_geo.geometry.buildings import BuildingSolid, EntranceInfo, LAYER_BUILDING_PARTS
from topology_geo.geometry.rail import RailRibbon
from topology_geo.geometry.roads import RoadRibbon
from topology_geo.geometry.streets import IntersectionArea, LaneMarking, LaneRibbon
from topology_geo.geometry.vegetation import TreePoint
from topology_geo.geometry.water import WaterArea, WaterwayRibbon
from topology_geo.ifc.assemble import (
    MARKING_CLEARANCE_M,
    PAVEMENT_CLEARANCE_M,
    BasePoint,
    SiteModel,
    build_site_ifc,
    extrude_polygon_mesh,
    flat_polygon_mesh,
    marking_elevation_fn,
    mesh_cylinder,
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
    }

    inter_psets = _psets_of_proxy_by_name(f, "Перекрёсток (sidewalk corner)")
    assert inter_psets["Pset_Перекрёсток"] == {"Тип": "sidewalk corner"}

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
    assert center_line_psets["Pset_Разметка"] == {"Тип": "center line", "Элементов": 2}

    arrow_psets = _psets_of_proxy_by_name(f, "Разметка (lane arrow)")
    assert arrow_psets["Pset_Разметка"] == {"Тип": "lane arrow", "Элементов": 1}

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


def test_marking_elevation_fn_sits_above_pavement():
    """Разметка — тонкий слой краски НА покрытии, ещё выше него — иначе линия
    разметки зрительно тонула бы в асфальте, поднятом над рельефом."""
    terrain = lambda x, y: 100.0
    marking = marking_elevation_fn(terrain)
    assert marking(0.0, 0.0) == pytest.approx(100.0 + PAVEMENT_CLEARANCE_M + MARKING_CLEARANCE_M)


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
