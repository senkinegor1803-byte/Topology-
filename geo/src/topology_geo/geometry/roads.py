"""Дороги: ленты по оси (Шаг 1.7, п. 1)."""

from __future__ import annotations

from dataclasses import dataclass

from shapely.ops import unary_union

from topology_geo.selection.service import SiteFeature

CONFIDENCE_FACT = "факт"
CONFIDENCE_DEFAULT = "умолчание"

WIDTH_BY_HIGHWAY_CLASS: dict[str, float] = {
    "motorway": 15.0,
    "trunk": 12.0,
    "primary": 10.0,
    "secondary": 9.0,
    "tertiary": 7.0,
    "residential": 6.0,
    "living_street": 5.0,
    "service": 4.0,
    "track": 3.0,
    "pedestrian": 4.0,
    "footway": 1.5,
    "cycleway": 2.0,
    "path": 1.0,
}
DEFAULT_WIDTH_M = 5.0


@dataclass(frozen=True)
class RoadRibbon:
    osm_id: int
    ribbon: object  # shapely Polygon/MultiPolygon, локальные координаты
    width_m: float
    width_confidence: str
    surface: str | None
    highway_class: str


def compute_width_m(feature: SiteFeature) -> tuple[float, str]:
    """Ширина проезжей части: `width` тег -> по классу дороги (Шаг 1.7, п. 1)."""
    width_tag = feature.raw_tags.get("width")
    if width_tag:
        try:
            width = float(str(width_tag).strip())
        except ValueError:
            width = None
        if width is not None and width > 0:
            return width, CONFIDENCE_FACT

    highway_class = feature.attributes.get("highway_class", "unclassified")
    return WIDTH_BY_HIGHWAY_CLASS.get(highway_class, DEFAULT_WIDTH_M), CONFIDENCE_DEFAULT


def _as_lines(geom):
    if geom.geom_type.startswith("Multi"):
        return [g for g in geom.geoms if g.length > 0]
    return [geom] if geom.length > 0 else []


def build_road_ribbons(features: list[SiteFeature]) -> list[RoadRibbon]:
    """Дорога (ось) -> лента (`LineString.buffer` с плоскими торцами, п. 1)."""
    ribbons: list[RoadRibbon] = []
    for feature in features:
        if feature.layer != "osm_roads":
            continue

        lines = _as_lines(feature.geometry)
        if not lines:
            continue

        width_m, width_confidence = compute_width_m(feature)
        ribbon = unary_union([line.buffer(width_m / 2, cap_style="flat") for line in lines])
        if ribbon.is_empty:
            continue

        ribbons.append(
            RoadRibbon(
                osm_id=feature.osm_id,
                ribbon=ribbon,
                width_m=width_m,
                width_confidence=width_confidence,
                surface=feature.attributes.get("surface"),
                highway_class=feature.attributes.get("highway_class", "unclassified"),
            )
        )
    return ribbons
