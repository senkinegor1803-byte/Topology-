"""Здания (упрощённо): контуры -> призмы (Шаг 1.6), формы крыш (Шаг 2.2).

Формула высоты — `docs/math-model.md` §2.5, расширена источником Overture
(Шаг 2.2, п. 2):

    height = tag(height) -> tag(building:levels)*3+1 -> Overture -> default_height(type)

Источник отмечается явно (`height_source`: OSM/Overture/тип), не только
confidence факт/умолчание — OSM и Overture оба «факт», но это разные
источники и разное доверие на практике (план явно требует «отметку
источника в свойствах», не только да/нет).

`tag(height)`/`tag(building:levels)` читаются из `SiteFeature.raw_tags`
(Шаг 1.4 нормализует только `levels` как число этажей, не саму высоту в
метрах — по формуле это разные вещи: `levels` в метры переводится здесь).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.validation import make_valid

from topology_geo.geometry.roofs import (
    ROOF_FLAT,
    RoofParams,
    compute_roof_params,
    oriented_bounding_box,
)
from topology_geo.selection.service import SiteFeature

CONFIDENCE_FACT = "факт"
CONFIDENCE_DEFAULT = "умолчание"

SOURCE_OSM = "OSM"
SOURCE_OVERTURE = "Overture"
SOURCE_DEFAULT_BY_TYPE = "тип (умолчание)"

# Высота по умолчанию (м), когда нет ни height, ни строго заданной этажности -
# используется только как последний, явно помеченный "умолчание", резерв.
DEFAULT_HEIGHT_BY_TYPE: dict[str, float] = {
    "house": 8.0,
    "detached": 8.0,
    "apartments": 25.0,
    "residential": 18.0,
    "industrial": 10.0,
    "warehouse": 8.0,
    "retail": 6.0,
    "commercial": 9.0,
    "garage": 3.0,
    "garages": 3.0,
    "shed": 3.0,
}
DEFAULT_HEIGHT_FALLBACK_M = 9.0

MIN_FOOTPRINT_AREA_M2 = 1.0


class OvertureHeightSource(Protocol):
    """Источник высоты Overture Buildings (Шаг 2.2, п. 2) - второй уровень
    водопада после OSM, перед дефолтом по типу."""

    def lookup(self, osm_id: int, footprint: Polygon) -> float | None: ...


class NullOvertureSource:
    """Реального доступа к датасету Overture Buildings в этой среде нет (не
    настроено сетевое/BigQuery-подключение к его хранилищу) - водопад
    всегда падает дальше, на дефолт по типу. Подключение реального источника
    (реализовать этот же протокол `OvertureHeightSource`) не требует
    изменений в `compute_height_m`/`extrude_buildings`."""

    def lookup(self, osm_id: int, footprint: Polygon) -> float | None:
        return None


@dataclass(frozen=True)
class BuildingSolid:
    osm_id: int
    footprint: Polygon  # исправленный, единообразно ориентированный контур (локальные координаты)
    height_m: float  # ПОЛНАЯ высота (до конька/шатра для скатной крыши, до плоского верха для flat)
    height_confidence: str
    height_source: str  # SOURCE_OSM | SOURCE_OVERTURE | SOURCE_DEFAULT_BY_TYPE
    base_z: float
    building_type: str
    roof_shape: str = ROOF_FLAT
    roof_height_m: float = 0.0
    roof_height_confidence: str = CONFIDENCE_FACT
    roof_ridge_along_long_axis: bool = True
    roof_direction: tuple[float, float] | None = None


def compute_height_m(
    feature: SiteFeature, overture_source: OvertureHeightSource | None = None
) -> tuple[float, str, str]:
    """Высота здания в метрах по водопаду OSM -> Overture -> тип (см.
    docstring модуля). Возвращает `(height_m, confidence, source)`."""
    height_tag = feature.raw_tags.get("height")
    if height_tag:
        try:
            height = float(str(height_tag).strip())
        except ValueError:
            height = None
        if height is not None and height > 0:
            return height, CONFIDENCE_FACT, SOURCE_OSM

    levels = feature.attributes.get("levels")
    if feature.confidence.get("levels") == CONFIDENCE_FACT and levels:
        return float(levels) * 3.0 + 1.0, CONFIDENCE_FACT, SOURCE_OSM

    if overture_source is not None:
        overture_height = overture_source.lookup(feature.osm_id, feature.geometry)
        if overture_height is not None and overture_height > 0:
            return overture_height, CONFIDENCE_FACT, SOURCE_OVERTURE

    building_type = feature.attributes.get("type", "yes")
    default_height = DEFAULT_HEIGHT_BY_TYPE.get(building_type, DEFAULT_HEIGHT_FALLBACK_M)
    return default_height, CONFIDENCE_DEFAULT, SOURCE_DEFAULT_BY_TYPE


def repair_footprint(geom) -> Polygon | None:
    """Исправить самопересечения (`make_valid`), убрать дубли соседних точек,
    привести внешний контур к единой ориентации (против часовой стрелки).

    Возвращает `None`, если после исправления не осталось валидного полигона
    достаточной площади (`MIN_FOOTPRINT_AREA_M2`) — вызывающий код должен
    отбросить такой объект, а не падать на нём.
    """
    if geom is None or geom.is_empty:
        return None

    fixed = make_valid(geom)
    if fixed.is_empty:
        return None

    if fixed.geom_type == "MultiPolygon":
        fixed = max(fixed.geoms, key=lambda p: p.area)
    elif fixed.geom_type == "GeometryCollection":
        polys = [g for g in fixed.geoms if g.geom_type == "Polygon"]
        if not polys:
            return None
        fixed = max(polys, key=lambda p: p.area)

    if fixed.geom_type != "Polygon" or fixed.is_empty:
        return None

    coords = list(fixed.exterior.coords)
    deduped = [coords[0]]
    for c in coords[1:]:
        if abs(c[0] - deduped[-1][0]) > 1e-9 or abs(c[1] - deduped[-1][1]) > 1e-9:
            deduped.append(c)
    if len(deduped) < 4:  # меньше 3 уникальных вершин + замыкание
        return None

    polygon = Polygon(deduped)
    if not polygon.is_valid or polygon.area < MIN_FOOTPRINT_AREA_M2:
        return None

    return orient(polygon, sign=1.0)


def extrude_buildings(
    features: list[SiteFeature],
    terrain_elevation_fn: Callable[[float, float], float | None],
    overture_source: OvertureHeightSource | None = None,
) -> list[BuildingSolid]:
    """Построить призмы зданий: контур -> исправленная геометрия, высота по
    водопаду OSM->Overture->тип (Шаг 1.6/2.2), низ — минимальная отметка
    рельефа по контуру (площадка, как и в `relief.tin`, Шаг 1.5), форма
    крыши по `roof:*` (Шаг 2.2, п. 1) поверх OBBox контура."""
    solids: list[BuildingSolid] = []
    for feature in features:
        if feature.layer != "osm_buildings":
            continue

        footprint = repair_footprint(feature.geometry)
        if footprint is None:
            continue

        height_m, height_confidence, height_source = compute_height_m(feature, overture_source)

        elevations = [terrain_elevation_fn(x, y) for x, y in footprint.exterior.coords]
        elevations = [z for z in elevations if z is not None]
        base_z = min(elevations) if elevations else 0.0

        obb = oriented_bounding_box(footprint)
        roof = compute_roof_params(feature, obb)
        # карниз (верх стен) = height_m - roof_height_m должен остаться
        # положительным - иначе исходные теги противоречивы (roof:height
        # больше или равен полной высоте здания), откатываемся на плоскую
        # крышу, а не строим вывернутую геометрию.
        if roof.shape != ROOF_FLAT and roof.height_m >= height_m:
            roof = RoofParams(ROOF_FLAT, CONFIDENCE_DEFAULT, 0.0, CONFIDENCE_FACT, True, None)

        solids.append(
            BuildingSolid(
                osm_id=feature.osm_id,
                footprint=footprint,
                height_m=height_m,
                height_confidence=height_confidence,
                height_source=height_source,
                base_z=base_z,
                building_type=feature.attributes.get("type", "yes"),
                roof_shape=roof.shape,
                roof_height_m=roof.height_m,
                roof_height_confidence=roof.height_confidence,
                roof_ridge_along_long_axis=roof.ridge_along_long_axis,
                roof_direction=roof.direction,
            )
        )
    return solids
