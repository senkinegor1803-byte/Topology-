"""Дороги: ленты по оси (Шаг 1.7, п. 1); классификация каркасная/
внутриквартальная (Шаг 2.4, п. 1) — `topology_geo.geometry.road_network`;
параметрическая перестройка внутриквартальной ленты по новой оси (Шаг 2.4,
п. 5) — `rebuild_road_ribbon`."""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import LineString
from shapely.ops import unary_union

from topology_geo.geometry.road_network import NETWORK_INTERNAL, classify_road_network
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
    network: str  # каркасная/внутриквартальная (Шаг 2.4, п. 1, classify_road_network)
    # Ось дороги (Шаг 2.4, п. 5: «внутриквартальная сеть — параметрическая:
    # ось + профиль») - только для дороги с ОДНИМ цельным сегментом линии
    # (см. build_road_ribbons); `None` для MultiLineString-геометрии
    # (несколько разрозненных кусков одного osm_id - какой из них "ось" для
    # редактирования, неочевидно, известное ограничение, см. rebuild_road_ribbon).
    axis: LineString | None = None


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

        highway_class = feature.attributes.get("highway_class", "unclassified")
        ribbons.append(
            RoadRibbon(
                osm_id=feature.osm_id,
                ribbon=ribbon,
                width_m=width_m,
                width_confidence=width_confidence,
                surface=feature.attributes.get("surface"),
                highway_class=highway_class,
                network=classify_road_network(highway_class),
                axis=lines[0] if len(lines) == 1 else None,
            )
        )
    return ribbons


def rebuild_road_ribbon(road: RoadRibbon, new_axis: LineString) -> RoadRibbon:
    """Перестроить ленту дороги по новой оси (Шаг 2.4, п. 5: «внутриквартальная
    сеть — параметрическая: ось + профиль, перестраивается при изменении
    оси»). Профиль (ширина, покрытие, класс) не меняется — переносится
    как есть, меняется только геометрия ленты (`new_axis.buffer`, тот же
    способ, что и в `build_road_ribbons`).

    Каркасная сеть НЕредактируема (Шаг 2.4, п. 4) - вызов на дороге с
    `network != NETWORK_INTERNAL` запрещён на уровне этой функции, а не
    только пометкой в свойствах IFC (`Pset_Дорога.Редактируемый`); дорога без
    сохранённой оси (`road.axis is None` - разрозненная MultiLineString-
    геометрия, см. `RoadRibbon.axis`) тоже не может быть перестроена этим
    способом."""
    if road.network != NETWORK_INTERNAL:
        raise ValueError(f"дорога {road.osm_id} принадлежит каркасной сети - нередактируема (Шаг 2.4, п. 4)")
    if road.axis is None:
        raise ValueError(f"у дороги {road.osm_id} нет сохранённой оси (составная геометрия) - перестроить нельзя")
    if new_axis.length <= 0:
        raise ValueError("новая ось должна иметь ненулевую длину")

    new_ribbon = new_axis.buffer(road.width_m / 2, cap_style="flat")
    return RoadRibbon(
        osm_id=road.osm_id,
        ribbon=new_ribbon,
        width_m=road.width_m,
        width_confidence=road.width_confidence,
        surface=road.surface,
        highway_class=road.highway_class,
        network=road.network,
        axis=new_axis,
    )
