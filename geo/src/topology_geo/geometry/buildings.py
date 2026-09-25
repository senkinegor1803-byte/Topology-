"""Здания (упрощённо): контуры -> призмы (Шаг 1.6).

Формула высоты — `docs/math-model.md` §2.5:

    height = tag(height) -> tag(building:levels)*3+1 -> default_height(type)  [confidence="умолчание"]

`tag(height)`/`tag(building:levels)` читаются из `SiteFeature.raw_tags`
(Шаг 1.4 нормализует только `levels` как число этажей, не саму высоту в
метрах — по формуле это разные вещи: `levels` в метры переводится здесь).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.validation import make_valid

from topology_geo.selection.service import SiteFeature

CONFIDENCE_FACT = "факт"
CONFIDENCE_DEFAULT = "умолчание"

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


@dataclass(frozen=True)
class BuildingSolid:
    osm_id: int
    footprint: Polygon  # исправленный, единообразно ориентированный контур (локальные координаты)
    height_m: float
    height_confidence: str
    base_z: float
    building_type: str


def compute_height_m(feature: SiteFeature) -> tuple[float, str]:
    """Высота здания в метрах по формуле Шага 1.6 (см. docstring модуля)."""
    height_tag = feature.raw_tags.get("height")
    if height_tag:
        try:
            height = float(str(height_tag).strip())
        except ValueError:
            height = None
        if height is not None and height > 0:
            return height, CONFIDENCE_FACT

    levels = feature.attributes.get("levels")
    if feature.confidence.get("levels") == CONFIDENCE_FACT and levels:
        return float(levels) * 3.0 + 1.0, CONFIDENCE_FACT

    building_type = feature.attributes.get("type", "yes")
    default_height = DEFAULT_HEIGHT_BY_TYPE.get(building_type, DEFAULT_HEIGHT_FALLBACK_M)
    return default_height, CONFIDENCE_DEFAULT


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
    features: list[SiteFeature], terrain_elevation_fn: Callable[[float, float], float | None]
) -> list[BuildingSolid]:
    """Построить призмы зданий: контур -> исправленная геометрия, высота по
    формуле Шага 1.6, низ — минимальная отметка рельефа по контуру (площадка,
    как и в `relief.tin` для той же цели, Шаг 1.5)."""
    solids: list[BuildingSolid] = []
    for feature in features:
        if feature.layer != "osm_buildings":
            continue

        footprint = repair_footprint(feature.geometry)
        if footprint is None:
            continue

        height_m, height_confidence = compute_height_m(feature)

        elevations = [terrain_elevation_fn(x, y) for x, y in footprint.exterior.coords]
        elevations = [z for z in elevations if z is not None]
        base_z = min(elevations) if elevations else 0.0

        solids.append(
            BuildingSolid(
                osm_id=feature.osm_id,
                footprint=footprint,
                height_m=height_m,
                height_confidence=height_confidence,
                base_z=base_z,
                building_type=feature.attributes.get("type", "yes"),
            )
        )
    return solids
