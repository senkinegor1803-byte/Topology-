"""Приём и привязка ЖК (Шаг 3.6): чтение геопривязки из IFC, мастер
размещения при её отсутствии, пятно застройки по секциям.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import ifcopenshell
import ifcopenshell.geom
import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import MultiPoint
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class MapConversion:
    """Геопривязка `IfcMapConversion` (та же схема, что пишет
    `ifc.generate_test_ifc`/`ifc.assemble` для собственных моделей проекта,
    Шаг 0.4/1.8 — читается симметрично тому, как записывается)."""

    crs_name: str
    eastings: float
    northings: float
    height: float
    rotation_deg: float  # atan2(XAxisOrdinate, XAxisAbscissa)
    scale: float


def read_map_conversion(model: ifcopenshell.file) -> MapConversion | None:
    """Действие п. 1: «чтение геопривязки» — `IfcMapConversion` (IFC4/
    IFC4X3, тот же механизм что использует сам проект для СВОИХ моделей).
    Renga экспортирует IFC через тот же штатный механизм схемы, не
    собственный формат — читаем его напрямую, без специального парсера под
    конкретный экспортёр."""
    conversions = model.by_type("IfcMapConversion")
    if not conversions:
        return None
    mc = conversions[0]
    crs = mc.TargetCRS
    crs_name = crs.Name if crs and crs.Name else "неизвестная СК"
    rotation_deg = math.degrees(math.atan2(mc.XAxisOrdinate or 0.0, mc.XAxisAbscissa or 1.0))
    return MapConversion(
        crs_name=crs_name, eastings=mc.Eastings, northings=mc.Northings,
        height=mc.OrthogonalHeight or 0.0, rotation_deg=rotation_deg, scale=mc.Scale or 1.0,
    )


@dataclass(frozen=True)
class PlacementResult:
    footprint: BaseGeometry  # размещённое пятно, локальные метрические координаты (та же СК, что site_boundary)
    is_within_site: bool
    overlap_ratio: float  # доля площади пятна, попавшая внутрь границы участка (1.0 - полностью внутри)


def place_building_manually(
    footprint_local: BaseGeometry, *, origin_x_m: float, origin_y_m: float,
    rotation_deg: float, site_boundary: BaseGeometry,
) -> PlacementResult:
    """Действие п. 2: «мастер размещения — точка, угол, проверка по
    контуру участка», когда геопривязки в IFC нет. `footprint_local` —
    контур ЖК в СОБСТВЕННЫХ координатах модели ЖК (например, (0,0) —
    условное начало из Renga); результат и `site_boundary` — в ОДНОЙ
    метрической системе координат (например, МСК-59, Шаг 0.4 — перевод
    делает вызывающая сторона через `coords.transform_geometry_to_msk59`,
    как и весь остальной проект)."""
    rotated = rotate(footprint_local, rotation_deg, origin=(0, 0))
    placed = translate(rotated, xoff=origin_x_m, yoff=origin_y_m)

    intersection = placed.intersection(site_boundary)
    overlap_ratio = (intersection.area / placed.area) if placed.area > 0 else 0.0
    is_within_site = site_boundary.contains(placed)

    return PlacementResult(footprint=placed, is_within_site=is_within_site, overlap_ratio=overlap_ratio)


def building_footprints_by_building(model: ifcopenshell.file) -> dict[str, BaseGeometry]:
    """Действие п. 3: «пятно застройки по секциям». IFC не имеет
    стандартного класса «секция» — в реальных проектах секция ЖК обычно
    оформляется отдельным `IfcBuilding` (несколько корпусов/секций в одном
    файле) — здесь пятно строится НА КАЖДЫЙ `IfcBuilding`, ключ словаря —
    его `Name` (или `GlobalId`, если имя не задано).

    Пятно — выпуклая оболочка (convex hull) всех точек геометрии элементов
    здания в плане (XY, мировые координаты) — честное упрощение: точный
    контур потребовал бы объединения (union) всех отдельных footprint'ов
    элементов, что при упрощённых/пересекающихся телах модели ЖК не всегда
    даёт единый простой полигон, а выпуклая оболочка — всегда."""
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    iterator = ifcopenshell.geom.iterator(settings, model)

    points_by_building: dict[str, list[tuple[float, float]]] = {}

    if iterator.initialize():
        while True:
            shape = iterator.get()
            element = model.by_id(shape.id)
            building = _find_containing_building(element)
            if building is not None:
                key = building.Name or building.GlobalId
                verts = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
                points_by_building.setdefault(key, []).extend((x, y) for x, y, _z in verts)
            if not iterator.next():
                break

    return {key: MultiPoint(points).convex_hull for key, points in points_by_building.items() if points}


def _find_containing_building(element) -> ifcopenshell.entity_instance | None:
    """Поднимается от элемента до ближайшего `IfcBuilding` по РЕАЛЬНЫМ
    связям схемы IFC (`IfcRelContainedInSpatialStructure` для непосредственного
    контейнера элемента, затем `IfcRelAggregates` вверх по пространственной
    иерархии) — не через `ifcopenshell.util.element`, чьё поведение на
    гранях (элемент напрямую в `IfcSite`, без этажа) не гарантировано
    задокументированным способом без выхода в сеть за документацией."""
    node = None
    for rel in getattr(element, "ContainedInStructure", ()) or ():
        node = rel.RelatingStructure
        break
    if node is None:
        node = element

    while node is not None and not node.is_a("IfcBuilding"):
        parent = None
        for rel in getattr(node, "Decomposes", ()) or ():
            parent = rel.RelatingObject
            break
        if parent is None or parent.is_a("IfcProject"):
            return None
        node = parent
    return node
