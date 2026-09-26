"""Железная дорога и трамвай: лента с балластом, рельсы упрощённо (Шаг 1.7, п. 3)."""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import LineString
from shapely.ops import unary_union

from topology_geo.selection.service import SiteFeature

# Упрощённо — одна ширина насыпи независимо от числа путей; разбивка по
# числу путей и колее с реальной шириной — уточнение Этапа 2 (шаг 2.6).
BALLAST_WIDTH_M = 4.0
DEFAULT_RAIL_TYPE = "rail"


@dataclass(frozen=True)
class RailRibbon:
    osm_id: int
    ballast: object  # shapely Polygon/MultiPolygon, локальные координаты
    rail_type: str  # rail | tram | light_rail и т.п. (тег railway=*)
    # Ось пути (Шаг 2.5, п. 3: проверка габарита моста над нижележащими
    # путями нужна ось для точки пересечения) - только для ОДНОГО цельного
    # сегмента линии, тот же принцип, что и у `RoadRibbon.axis` (Шаг 2.4).
    axis: LineString | None = None


def _as_lines(geom):
    if geom.geom_type.startswith("Multi"):
        return [g for g in geom.geoms if g.length > 0]
    return [geom] if geom.length > 0 else []


def build_rail_ribbons(features: list[SiteFeature], *, ballast_width_m: float = BALLAST_WIDTH_M) -> list[RailRibbon]:
    result: list[RailRibbon] = []
    for feature in features:
        if feature.layer != "osm_railways":
            continue
        lines = _as_lines(feature.geometry)
        if not lines:
            continue
        ballast = unary_union([line.buffer(ballast_width_m / 2, cap_style="flat") for line in lines])
        if ballast.is_empty:
            continue
        rail_type = feature.raw_tags.get("railway") or DEFAULT_RAIL_TYPE
        result.append(
            RailRibbon(
                osm_id=feature.osm_id, ballast=ballast, rail_type=rail_type,
                axis=lines[0] if len(lines) == 1 else None,
            )
        )
    return result
