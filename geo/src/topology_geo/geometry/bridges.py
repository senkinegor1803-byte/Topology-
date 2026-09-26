"""Мосты, путепроводы (Шаг 2.5, п. 1-3).

Действия по плану, п. 1-3: найти мостовые участки (`bridge=*`); построить
пролётное строение по оси с интерполяцией отметок между устоями; проверить
габарит над нижележащей дорогой/путями, при нарушении — поднять пролёт и
пометить «расчётно».

Пролётное строение — жёсткая конструкция, поэтому отметка вдоль оси не
следует рельефу (в отличие от дороги, `relief.tin.build_profile_elevation_fn`,
или воды): она определяется ЛИНЕЙНОЙ интерполяцией между отметками рельефа
на двух концах оси (устои). Габарит проверяется в точках, где ось моста
геометрически пересекает ось другой (немостовой) дороги или пути — не
выборкой по площади: пересечение осей и есть место проверки.

Опоры по шагу для типа моста и насыпи подходов (п. 4), тоннели (п. 5) — в
этом проходе НЕ реализованы, отдельная задача (см. `docs/bridges.md`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from shapely.geometry import LineString, Point

from topology_geo.geometry.road_network import classify_road_network
from topology_geo.geometry.roads import compute_width_m, is_bridge
from topology_geo.selection.service import SiteFeature

ElevationFn = Callable[[float, float], float | None]

STATUS_OFFICIAL = "официальный"
STATUS_CALCULATED = "расчётный"

# Минимальный вертикальный габарит моста/путепровода над нижележащей дорогой
# или путями (Шаг 2.5, п. 3) — типовое значение по нормам проектирования
# (СП 34.13330 «Автомобильные дороги» для автодорог, габарит приближения
# строений с учётом контактной сети для электрифицированных ж/д путей),
# единое для всех классов дороги/типов пути — упрощение, не полноценный
# расчёт по нагрузке/скорости/негабаритным грузам.
MIN_CLEARANCE_ROAD_M = 5.0
MIN_CLEARANCE_RAIL_M = 6.0


@dataclass(frozen=True)
class BridgeRibbon:
    osm_id: int
    ribbon: object  # shapely Polygon/MultiPolygon, локальные координаты
    axis: LineString | None
    width_m: float
    width_confidence: str
    surface: str | None
    highway_class: str
    network: str
    deck_elevation_fn: ElevationFn
    clearance_m: float | None  # минимальный найденный габарит; None — пересечений с нижележащими объектами не найдено
    status: str  # STATUS_OFFICIAL | STATUS_CALCULATED


def _as_line(geom) -> LineString | None:
    if geom.geom_type == "LineString" and geom.length > 0:
        return geom
    return None


def _lifted(fn: ElevationFn, lift_m: float) -> ElevationFn:
    """Обёртка над отметкой пролёта — поднять на `lift_m` (Шаг 2.5, п. 3,
    нарушение габарита)."""

    def _fn(x: float, y: float) -> float | None:
        z = fn(x, y)
        return None if z is None else z + lift_m

    return _fn


def abutment_elevation_fn(axis: LineString, terrain_elevation_fn: ElevationFn) -> ElevationFn:
    """Интерполяция отметок между устоями (Шаг 2.5, п. 2) — линейная по
    длине оси между отметкой рельефа НА ЕЁ КОНЦАХ (устои), а не сглаженный
    профиль вдоль оси, как у дороги/воды: пролётное строение — жёсткая
    конструкция, её отметка определяется только двумя опорными точками, а
    не рельефом под ней."""
    x0, y0 = axis.coords[0]
    x1, y1 = axis.coords[-1]
    z0 = terrain_elevation_fn(x0, y0) or 0.0
    z1 = terrain_elevation_fn(x1, y1) or 0.0
    length = axis.length

    def _fn(x: float, y: float) -> float:
        t = axis.project(Point(x, y)) / length if length > 0 else 0.0
        return z0 + (z1 - z0) * t

    return _fn


def _clearances_at_crossings(
    axis: LineString, deck_fn: ElevationFn, terrain_elevation_fn: ElevationFn, crossing_axes: list[LineString]
) -> list[float]:
    """Габариты (Шаг 2.5, п. 3) в точках пересечения оси моста с осями
    `crossing_axes` (не мостов — крест мост-над-мостом, например развязка в
    два уровня, в этом проходе не разбирается, известное ограничение)."""
    clearances: list[float] = []
    for other_axis in crossing_axes:
        if not axis.intersects(other_axis):
            continue
        intersection = axis.intersection(other_axis)
        points = [intersection] if intersection.geom_type == "Point" else list(getattr(intersection, "geoms", []))
        for point in points:
            if point.geom_type != "Point":
                continue
            deck_z = deck_fn(point.x, point.y)
            ground_z = terrain_elevation_fn(point.x, point.y)
            if deck_z is None or ground_z is None:
                continue
            clearances.append(deck_z - ground_z)
    return clearances


def build_bridge_ribbons(
    features: list[SiteFeature],
    terrain_elevation_fn: ElevationFn,
    *,
    crossing_road_axes: list[LineString] | None = None,
    crossing_rail_axes: list[LineString] | None = None,
    min_clearance_road_m: float = MIN_CLEARANCE_ROAD_M,
    min_clearance_rail_m: float = MIN_CLEARANCE_RAIL_M,
) -> list[BridgeRibbon]:
    """Построить мостовые участки (Шаг 2.5, п. 1-3) из дорог с тегом
    `bridge=*` (`is_bridge`, `geometry.roads`).

    `crossing_road_axes`/`crossing_rail_axes` — оси других (немостовых)
    дорог и путей для проверки габарита (п. 3), у каждого типа свой
    норматив (`MIN_CLEARANCE_ROAD_M`/`MIN_CLEARANCE_RAIL_M` — путь требует
    больше из-за контактной сети). Без них (умолчание — пустые списки)
    проверка не выполняется и пролёт остаётся официальным — обоснованный
    честный водопад: без осей других объектов проверить нечего, нарушение
    не изобретается. Составная (`MultiLineString`) геометрия моста не
    обрабатывается — нет единственной оси для интерполяции устоев (тот же
    принцип, что у `RoadRibbon.axis`, Шаг 2.4)."""
    crossing_road_axes = crossing_road_axes or []
    crossing_rail_axes = crossing_rail_axes or []
    ribbons: list[BridgeRibbon] = []
    for feature in features:
        if feature.layer != "osm_roads" or not is_bridge(feature.raw_tags):
            continue

        axis = _as_line(feature.geometry)
        if axis is None:
            continue

        width_m, width_confidence = compute_width_m(feature)
        ribbon = axis.buffer(width_m / 2, cap_style="flat")
        if ribbon.is_empty:
            continue

        deck_fn = abutment_elevation_fn(axis, terrain_elevation_fn)
        road_clearances = _clearances_at_crossings(axis, deck_fn, terrain_elevation_fn, crossing_road_axes)
        rail_clearances = _clearances_at_crossings(axis, deck_fn, terrain_elevation_fn, crossing_rail_axes)
        all_clearances = road_clearances + rail_clearances
        clearance = min(all_clearances) if all_clearances else None

        road_deficit = min_clearance_road_m - min(road_clearances) if road_clearances else 0.0
        rail_deficit = min_clearance_rail_m - min(rail_clearances) if rail_clearances else 0.0
        lift = max(road_deficit, rail_deficit, 0.0)

        status = STATUS_OFFICIAL
        if lift > 0.0:
            deck_fn = _lifted(deck_fn, lift)
            clearance = clearance + lift
            status = STATUS_CALCULATED

        highway_class = feature.attributes.get("highway_class", "unclassified")
        ribbons.append(
            BridgeRibbon(
                osm_id=feature.osm_id,
                ribbon=ribbon,
                axis=axis,
                width_m=width_m,
                width_confidence=width_confidence,
                surface=feature.attributes.get("surface"),
                highway_class=highway_class,
                network=classify_road_network(highway_class),
                deck_elevation_fn=deck_fn,
                clearance_m=clearance,
                status=status,
            )
        )
    return ribbons
