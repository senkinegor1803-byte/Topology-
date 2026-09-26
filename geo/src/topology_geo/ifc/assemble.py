"""Сборка `site.ifc` участка (Шаг 1.8).

`IfcProject` → `IfcSite` с `IfcMapConversion` и базовой точкой (п. 1);
каждый объект Шагов 1.5-1.7 — класс IFC и наборы свойств по словарю данных
(`docs/data-dictionary.md`), уникальный `GlobalId` (генерируется
ifcopenshell, `registry.py` сохраняет связь `osm_id -> GlobalId` в PostGIS,
п. 2); экспорт в схему, выбранную на Шаге 0.2 (п. 3); валидация —
`ifcopenshell.validate` (п. 4, переиспользует
`topology_geo.ifc.generate_test_ifc.validate_model`).

Полигон → меш (треугольники) через `mapbox_earcut` — терпит невыпуклые
контуры (частые у реальных зданий), в отличие от `shapely` (там нет
триангуляции, ограниченной контуром полигона).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.geometry
import ifcopenshell.api.georeference
import ifcopenshell.api.project
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.unit
import mapbox_earcut as earcut
import numpy as np

from topology_geo.geometry.buildings import BuildingSolid
from topology_geo.geometry.rail import RailRibbon
from topology_geo.geometry.roads import RoadRibbon
from topology_geo.geometry.roofs import (
    ROOF_FLAT,
    RoofParams,
    build_pitched_building_mesh,
    oriented_bounding_box,
)
from topology_geo.geometry.streets import IntersectionArea, LaneMarking, LaneRibbon
from topology_geo.geometry.vegetation import TreePoint
from topology_geo.geometry.water import WaterArea, WaterwayRibbon
from topology_geo.relief.tin import SiteTin

Vertex = tuple[float, float, float]
Face = tuple[int, ...]

ElevationFn = Callable[[float, float], float | None]
GlobalIdRegistry = list[tuple[str, int, str]]  # (layer, osm_id, GlobalId)


@dataclass(frozen=True)
class SiteModel:
    """Геометрические слои участка, готовые к сборке в IFC (выход Шагов 1.5-1.7)."""

    tin: SiteTin | None = None
    buildings: list[BuildingSolid] = field(default_factory=list)
    roads: list[RoadRibbon] = field(default_factory=list)
    water_areas: list[WaterArea] = field(default_factory=list)
    waterways: list[WaterwayRibbon] = field(default_factory=list)
    rail: list[RailRibbon] = field(default_factory=list)
    trees: list[TreePoint] = field(default_factory=list)
    lanes: list[LaneRibbon] = field(default_factory=list)
    intersections: list[IntersectionArea] = field(default_factory=list)
    markings: list[LaneMarking] = field(default_factory=list)


@dataclass(frozen=True)
class BasePoint:
    """Базовая точка проекта (Шаг 0.4, п. 4): передаётся в IFC через `IfcMapConversion`."""

    lon: float
    lat: float
    zone: int
    x: float  # МСК-59
    y: float
    height: float = 0.0


def _ring_coords(ring) -> list[tuple[float, float]]:
    coords = list(ring.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    return coords


def triangulate_polygon(polygon) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    """Триангулировать (возможно, невыпуклый, возможно, с отверстиями)
    полигон. Возвращает плоские вершины (в исходной 2D-плоскости объекта) и
    треугольники по их индексам."""
    rings = [_ring_coords(polygon.exterior)] + [_ring_coords(r) for r in polygon.interiors]
    flat_vertices: list[tuple[float, float]] = []
    ring_ends = []
    for ring in rings:
        flat_vertices.extend(ring)
        ring_ends.append(len(flat_vertices))

    verts_array = np.array(flat_vertices, dtype=np.float64)
    ring_ends_array = np.array(ring_ends, dtype=np.uint32)
    indices = earcut.triangulate_float64(verts_array, ring_ends_array)
    triangles = [tuple(int(i) for i in indices[i : i + 3]) for i in range(0, len(indices), 3)]
    return flat_vertices, triangles


def flat_polygon_mesh(polygon, elevation_fn: ElevationFn, *, default_z: float = 0.0) -> tuple[list[Vertex], list[Face]]:
    """Плоский (однослойный) меш полигона на высоте `elevation_fn(x, y)` в
    каждой вершине — для «наклеенных» на рельеф объектов (лента дороги,
    водоём, балласт ж/д): не солид, а поверхность."""
    flat_vertices, triangles = triangulate_polygon(polygon)
    vertices = [(x, y, elevation_fn(x, y) or default_z) for x, y in flat_vertices]
    return vertices, [tuple(t) for t in triangles]


def _combine_meshes(meshes: list[tuple[list[Vertex], list[Face]]]) -> tuple[list[Vertex], list[Face]]:
    """Слить несколько независимых мешей в один (вершины подряд, грани со
    сдвигом индексов) - один продукт IFC вместо одного на каждый элемент.
    Нужно для разметки (Шаг 2.3, п. 3): у одного перекрёстка её сотни мелких
    штрихов/стрелок одного вида, `ifcopenshell.api.geometry.add_mesh_representation`
    не поддерживает несколько представлений с разным числом вершин в одном
    продукте (`numpy.array` требует прямоугольную форму) - проще и надёжнее
    склеить геометрию заранее, чем упираться в это ограничение."""
    vertices: list[Vertex] = []
    faces: list[Face] = []
    offset = 0
    for verts, tris in meshes:
        vertices.extend(verts)
        faces.extend(tuple(i + offset for i in tri) for tri in tris)
        offset += len(verts)
    return vertices, faces


def extrude_polygon_mesh(polygon, base_z: float, top_z: float) -> tuple[list[Vertex], list[Face]]:
    """Призма: контур полигона на `base_z`, крыша на `top_z`, стены по
    внешнему контуру И по контурам внутренних отверстий (дворов) — иначе меш
    не замкнут (проверено расчётом объёма по теореме о дивергенции: без стен
    двора объём получался завышен на объём "трубы" между дном и крышей
    отверстия)."""
    flat_vertices, triangles = triangulate_polygon(polygon)
    n = len(flat_vertices)

    bottom = [(x, y, base_z) for x, y in flat_vertices]
    top = [(x, y, top_z) for x, y in flat_vertices]
    vertices = bottom + top

    faces: list[Face] = []
    for a, b, c in triangles:
        faces.append((a, c, b))  # низ смотрит вниз
        faces.append((a + n, b + n, c + n))  # верх смотрит вверх

    all_rings = [_ring_coords(polygon.exterior)] + [_ring_coords(r) for r in polygon.interiors]
    offset = 0
    for ring in all_rings:
        ring_n = len(ring)
        for i in range(ring_n):
            j = (i + 1) % ring_n
            b0, b1, t0, t1 = offset + i, offset + j, offset + i + n, offset + j + n
            faces.append((b0, b1, t1))
            faces.append((b0, t1, t0))
        offset += ring_n

    return vertices, faces


def mesh_cylinder(radius: float, length: float, segments: int = 8) -> tuple[list[Vertex], list[Face]]:
    """Упрощённый ствол дерева — переиспользует ту же форму, что тестовый
    генератор Шага 0.2 (`ifc.generate_test_ifc.mesh_cylinder`)."""
    verts: list[Vertex] = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        verts.append((radius * math.cos(angle), radius * math.sin(angle), 0.0))
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        verts.append((radius * math.cos(angle), radius * math.sin(angle), length))

    faces: list[Face] = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((i, j, segments + j))
        faces.append((i, segments + j, segments + i))
    for i in range(1, segments - 1):
        faces.append((0, i + 1, i))
    base = segments
    for i in range(1, segments - 1):
        faces.append((base, base + i, base + i + 1))
    return verts, faces


def _add_mesh_product(
    f: ifcopenshell.file,
    body_context,
    ifc_class: str,
    name: str,
    predefined_type: str | None,
    mesh: tuple[list[Vertex], list[Face]],
    psets: dict[str, dict],
) -> ifcopenshell.entity_instance.entity_instance:
    verts, faces = mesh
    product = ifcopenshell.api.root.create_entity(f, ifc_class=ifc_class, name=name, predefined_type=predefined_type)
    rep = ifcopenshell.api.geometry.add_mesh_representation(f, context=body_context, vertices=[verts], faces=[faces])
    ifcopenshell.api.geometry.assign_representation(f, product=product, representation=rep)
    for pset_name, properties in psets.items():
        if not properties:
            continue
        pset = ifcopenshell.api.pset.add_pset(f, product=product, name=pset_name)
        ifcopenshell.api.pset.edit_pset(f, pset=pset, properties=properties)
    return product


def build_site_ifc(
    schema: str,
    site_model: SiteModel,
    base_point: BasePoint,
    *,
    relief_resolution_note: str = "",
) -> tuple[ifcopenshell.file, GlobalIdRegistry]:
    """Собрать `site.ifc` по Шагу 1.8 (п. 1-2). Возвращает модель и список
    `(слой, osm_id, GlobalId)` для сохранения в PostGIS (`registry.py`, п. 2).
    Экспорт/валидация — у вызывающего кода (`write` /
    `topology_geo.ifc.generate_test_ifc.validate_model`)."""
    f = ifcopenshell.api.project.create_file(version=schema)
    ifcopenshell.api.root.create_entity(f, ifc_class="IfcProject", name="Топология — участок")
    ifcopenshell.api.unit.assign_unit(f)

    model_context = ifcopenshell.api.context.add_context(f, context_type="Model")
    body_context = ifcopenshell.api.context.add_context(
        f, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=model_context
    )

    site = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSite", name="Участок")
    ifcopenshell.api.aggregate.assign_object(f, products=[site], relating_object=f.by_type("IfcProject")[0])

    ifcopenshell.api.georeference.add_georeferencing(f)
    ifcopenshell.api.georeference.edit_georeferencing(
        f,
        projected_crs={"Name": f"MSK-59 zone {base_point.zone}"},
        coordinate_operation={
            "Eastings": base_point.x,
            "Northings": base_point.y,
            "OrthogonalHeight": base_point.height,
            "XAxisAbscissa": 1.0,
            "XAxisOrdinate": 0.0,
            "Scale": 1.0,
        },
    )

    products: list = []  # обычные IfcElement -> spatial.assign_container
    spatial_children: list = []  # IfcSpatialElement (нативный IfcRoad в IFC4X3) -> aggregate.assign_object
    registry: GlobalIdRegistry = []
    is_ifc43 = schema == "IFC4X3"

    if site_model.tin is not None:
        tin = site_model.tin
        verts = [tuple(float(c) for c in v) for v in tin.vertices]
        faces = [tuple(int(i) for i in tri) for tri in tin.triangles]
        terrain = _add_mesh_product(
            f, body_context, "IfcGeographicElement", "Рельеф участка (TIN)", "TERRAIN",
            (verts, faces),
            {"Pset_Контекст": {"Источник": "Топология: TIN участка (Шаг 1.5)", "Примечание": relief_resolution_note}},
        )
        products.append(terrain)

    for building in site_model.buildings:
        if building.roof_shape == ROOF_FLAT:
            mesh = extrude_polygon_mesh(building.footprint, building.base_z, building.base_z + building.height_m)
        else:
            obb = oriented_bounding_box(building.footprint)
            roof_params = RoofParams(
                shape=building.roof_shape,
                shape_confidence=building.roof_height_confidence,  # не используется при сборке меша
                height_m=building.roof_height_m,
                height_confidence=building.roof_height_confidence,
                ridge_along_long_axis=building.roof_ridge_along_long_axis,
                direction=building.roof_direction,
            )
            eave_z = building.base_z + building.height_m - building.roof_height_m
            mesh = build_pitched_building_mesh(obb, roof_params, building.base_z, eave_z)

        building_pset = {
            "Тип": building.building_type,
            "Высота_м": building.height_m,
            "Источник_высоты": building.height_source,
        }
        if building.entrances:
            building_pset["Входов_всего"] = len(building.entrances)
            for i, entrance in enumerate(building.entrances, start=1):
                building_pset[f"Вход_{i}_Тип"] = entrance.entrance_type
                building_pset[f"Вход_{i}_X_м"] = round(entrance.x, 3)
                building_pset[f"Вход_{i}_Y_м"] = round(entrance.y, 3)

        roof_pset = {}
        if building.roof_shape != ROOF_FLAT:
            roof_pset["Форма"] = building.roof_shape
            roof_pset["Высота_конька_м"] = building.roof_height_m
            roof_pset["Источник_высоты"] = building.roof_height_confidence
            if building.roof_direction is not None:
                dx, dy = building.roof_direction
                roof_pset["Направление_град"] = round(math.degrees(math.atan2(dx, dy)) % 360.0, 1)

        context_pset = {"Источник": "OSM (Шаг 1.1)"}
        if building.is_part:
            context_pset["Часть_здания"] = True  # building:part (Шаг 2.2, п. 1) - контур пропущен, см. geometry/buildings.py

        product = _add_mesh_product(
            f, body_context, "IfcBuildingElementProxy", f"Здание {building.osm_id}", "USERDEFINED",
            mesh,
            {
                "Pset_Здание": building_pset,
                "Pset_Крыша": roof_pset,
                "Pset_Контекст": context_pset,
            },
        )
        products.append(product)
        registry.append((building.source_layer, building.osm_id, product.GlobalId))

    def _terrain_elevation(x: float, y: float) -> float | None:
        return site_model.tin.interpolate_z(x, y) if site_model.tin is not None else 0.0

    # IfcRoad — нативно в IFC4X3 (IfcSpatialElement, роднится с сайтом через
    # aggregate.assign_object, как в Шаге 0.2); в IFC4 схема его не знает,
    # используется прокси IfcBuildingElementProxy (обычный IfcElement,
    # роднится через spatial.assign_container) с пометкой в Pset_Контекст —
    # тот же приём совместимости, что и в generate_test_ifc.py.
    for road in site_model.roads:
        polys = road.ribbon.geoms if road.ribbon.geom_type.startswith("Multi") else [road.ribbon]
        for poly in polys:
            mesh = flat_polygon_mesh(poly, _terrain_elevation)
            road_pset = {
                "Класс": road.highway_class,
                "Покрытие": road.surface or "",
                "Ширина_м": road.width_m,
            }
            if is_ifc43:
                product = _add_mesh_product(
                    f, body_context, "IfcRoad", f"Дорога {road.osm_id}", None,
                    mesh, {"Pset_Дорога": road_pset, "Pset_Контекст": {"Источник": "OSM (Шаг 1.1)"}},
                )
                spatial_children.append(product)
            else:
                product = _add_mesh_product(
                    f, body_context, "IfcBuildingElementProxy", f"Дорога {road.osm_id}", "USERDEFINED",
                    mesh,
                    {
                        "Pset_Дорога": road_pset,
                        "Pset_Контекст": {"Заменяет_класс": "IfcRoad", "Источник": "OSM (Шаг 1.1)"},
                    },
                )
                products.append(product)
            registry.append(("osm_roads", road.osm_id, product.GlobalId))

    # Полосы (Шаг 2.3, п. 1, `geometry.streets`) - независимый от `roads`
    # слой ДЕТАЛИЗАЦИИ того же `highway=*`: несколько полос на один
    # `osm_way_id` (проезжая часть х2 + тротуары), поэтому, как и для
    # многосегментных `road.ribbon`/`waterway.ribbon` выше, в реестр попадает
    # только последний GlobalId на такой `osm_id` - тот же принятый компромисс
    # (реестр рассчитан на 1 запись на исходный объект, не на под-объекты).
    for lane in site_model.lanes:
        mesh = flat_polygon_mesh(lane.polygon, _terrain_elevation)
        lane_name = "Полоса " + "+".join(str(i) for i in lane.osm_way_ids)
        product = _add_mesh_product(
            f, body_context, "IfcBuildingElementProxy", lane_name, "USERDEFINED",
            mesh,
            {
                "Pset_Полоса": {
                    "Тип": lane.lane_type, "Ширина_м": lane.width_m, "Направление": lane.direction,
                    "Покрытие": lane.surface or "",
                },
                "Pset_Контекст": {"Источник": "osm2streets (Шаг 2.3, п. 1)"},
            },
        )
        products.append(product)
        for osm_id in lane.osm_way_ids:
            registry.append(("osm_roads", osm_id, product.GlobalId))

    for intersection in site_model.intersections:
        mesh = flat_polygon_mesh(intersection.polygon, _terrain_elevation)
        product = _add_mesh_product(
            f, body_context, "IfcBuildingElementProxy", f"Перекрёсток ({intersection.kind})", "USERDEFINED",
            mesh,
            {
                "Pset_Перекрёсток": {"Тип": intersection.kind},
                "Pset_Контекст": {"Источник": "osm2streets (Шаг 2.3, п. 1)"},
            },
        )
        products.append(product)
        # не из одного OSM-объекта (перекрёсток собран из нескольких way) -
        # нет естественного osm_id, в реестр не попадает (как и рельеф/TIN выше).

    # Один продукт НА ВИД разметки (не на штрих/стрелку) - их у одного
    # перекрёстка могут быть сотни (дискретные отрезки центральной линии),
    # склеены в один меш (`_combine_meshes`), иначе site.ifc распухает от
    # тысяч тривиальных объектов на честный участок 3 км.
    markings_by_kind: dict[str, list[LaneMarking]] = {}
    for marking in site_model.markings:
        markings_by_kind.setdefault(marking.kind, []).append(marking)
    for kind, markings in markings_by_kind.items():
        mesh = _combine_meshes([flat_polygon_mesh(m.polygon, _terrain_elevation) for m in markings])
        product = _add_mesh_product(
            f, body_context, "IfcBuildingElementProxy", f"Разметка ({kind})", "USERDEFINED",
            mesh,
            {
                "Pset_Разметка": {"Тип": kind, "Элементов": len(markings)},
                "Pset_Контекст": {"Источник": "osm2streets (Шаг 2.3, п. 3)"},
            },
        )
        products.append(product)
        # без естественного osm_id (штрих/стрелка не привязаны к одному way,
        # да и склеены по несколько в один продукт) - не попадает в реестр,
        # как перекрёстки выше.

    for water in site_model.water_areas:
        mesh = flat_polygon_mesh(water.polygon, lambda x, y, z=water.level_z: z)
        product = _add_mesh_product(
            f, body_context, "IfcGeographicElement", f"Водоём {water.osm_id}", "USERDEFINED",
            mesh, {"Pset_Вода": {"Тип": "водоём", "Отметка_уреза_м": water.level_z}},
        )
        products.append(product)
        registry.append(("osm_water_areas", water.osm_id, product.GlobalId))

    for waterway in site_model.waterways:
        polys = waterway.ribbon.geoms if waterway.ribbon.geom_type.startswith("Multi") else [waterway.ribbon]
        for poly in polys:
            mesh = flat_polygon_mesh(poly, _terrain_elevation)
            product = _add_mesh_product(
                f, body_context, "IfcGeographicElement", f"Водоток {waterway.osm_id}", "USERDEFINED",
                mesh, {"Pset_Вода": {"Тип": "водоток", "Ширина_м": waterway.width_m}},
            )
            products.append(product)
            registry.append(("osm_waterways", waterway.osm_id, product.GlobalId))

    for rail in site_model.rail:
        polys = rail.ballast.geoms if rail.ballast.geom_type.startswith("Multi") else [rail.ballast]
        for poly in polys:
            mesh = flat_polygon_mesh(poly, _terrain_elevation)
            product = _add_mesh_product(
                f, body_context, "IfcBuildingElementProxy", f"Ж/д {rail.osm_id}", "USERDEFINED",
                mesh, {"Pset_ЖД": {"Тип": rail.rail_type}, "Pset_Контекст": {"Заменяет_класс": "IfcRail"}},
            )
            products.append(product)
            registry.append(("osm_railways", rail.osm_id, product.GlobalId))

    for i, tree in enumerate(site_model.trees):
        z = _terrain_elevation(tree.x, tree.y) or 0.0
        verts, faces = mesh_cylinder(0.15, 6.0, segments=6)
        verts = [(x + tree.x, y + tree.y, zz + z) for x, y, zz in verts]
        product = _add_mesh_product(
            f, body_context, "IfcGeographicElement", f"Дерево {tree.source_osm_id}-{i}", "USERDEFINED",
            (verts, faces),
            {"Pset_Растительность": {"Порода": tree.species, "Источник": tree.confidence}},
        )
        products.append(product)
        registry.append(("osm_vegetation", tree.source_osm_id, product.GlobalId))

    if products:
        ifcopenshell.api.spatial.assign_container(f, products=products, relating_structure=site)
    if spatial_children:
        ifcopenshell.api.aggregate.assign_object(f, products=spatial_children, relating_object=site)

    return f, registry
