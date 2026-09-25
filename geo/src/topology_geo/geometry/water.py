"""Вода: полигоны на отметке уреза, реки/ручьи — ленты (Шаг 1.7, п. 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from shapely.ops import unary_union

from topology_geo.selection.service import SiteFeature

DEFAULT_WATERWAY_WIDTH_M = 3.0


@dataclass(frozen=True)
class WaterArea:
    osm_id: int
    polygon: object  # shapely Polygon, локальные координаты
    level_z: float  # отметка уреза - min рельефа по границе (как площадка здания, Шаг 1.5/1.6)


@dataclass(frozen=True)
class WaterwayRibbon:
    osm_id: int
    ribbon: object
    width_m: float


def _as_polygons(geom):
    if geom.geom_type.startswith("Multi"):
        return [g for g in geom.geoms if not g.is_empty]
    return [] if geom.is_empty else [geom]


def _as_lines(geom):
    if geom.geom_type.startswith("Multi"):
        return [g for g in geom.geoms if g.length > 0]
    return [geom] if geom.length > 0 else []


def build_water_areas(
    features: list[SiteFeature], terrain_elevation_fn: Callable[[float, float], float | None]
) -> list[WaterArea]:
    """Водоёмы (`osm_water_areas`) на единой отметке уреза (min рельефа по границе)."""
    result: list[WaterArea] = []
    for feature in features:
        if feature.layer != "osm_water_areas":
            continue
        for polygon in _as_polygons(feature.geometry):
            elevations = [terrain_elevation_fn(x, y) for x, y in polygon.exterior.coords]
            elevations = [z for z in elevations if z is not None]
            level_z = min(elevations) if elevations else 0.0
            result.append(WaterArea(osm_id=feature.osm_id, polygon=polygon, level_z=level_z))
    return result


def build_waterway_ribbons(
    features: list[SiteFeature], *, width_m: float = DEFAULT_WATERWAY_WIDTH_M
) -> list[WaterwayRibbon]:
    """Реки/ручьи (`osm_waterways`) -> лента постоянной ширины (Этап 2 уточнит по расходу/классу)."""
    result: list[WaterwayRibbon] = []
    for feature in features:
        if feature.layer != "osm_waterways":
            continue
        lines = _as_lines(feature.geometry)
        if not lines:
            continue
        ribbon = unary_union([line.buffer(width_m / 2, cap_style="flat") for line in lines])
        if ribbon.is_empty:
            continue
        result.append(WaterwayRibbon(osm_id=feature.osm_id, ribbon=ribbon, width_m=width_m))
    return result
