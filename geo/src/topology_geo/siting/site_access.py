"""Очистка площадки и подъезды (Шаг 3.8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from topology_geo.geometry.road_network import NETWORK_BACKBONE


class _HasGeometry(Protocol):
    geometry: BaseGeometry


@dataclass(frozen=True)
class ClearingResult:
    removed: list[Any]
    kept: list[Any]


def clear_site(features: list[_HasGeometry], footprint: BaseGeometry) -> ClearingResult:
    """Действие: «удаление существующих объектов в границах участка со
    списком удалённого» — реальный список, не просто счётчик, чтобы
    отчёт (Шаг 3.9) мог перечислить, что именно удалено."""
    removed: list[Any] = []
    kept: list[Any] = []
    for feature in features:
        if feature.geometry.intersects(footprint):
            removed.append(feature)
        else:
            kept.append(feature)
    return ClearingResult(removed=removed, kept=kept)


class _RoadLike(Protocol):
    geometry: BaseGeometry
    network: str  # "каркасная" | "внутриквартальная" (geometry.road_network.NETWORK_*)


@dataclass(frozen=True)
class AccessRoad:
    path: BaseGeometry  # LineString от границы участка до точки примыкания
    connects_to: _RoadLike
    requires_user_confirmation: bool
    crosses_obstacle: bool


def build_access_road(
    site_boundary: BaseGeometry, roads: list[_RoadLike], obstacles: list[BaseGeometry],
) -> AccessRoad | None:
    """Действие: «подъезд к ближайшему внутриквартальному проезду по
    кратчайшему пути; примыкание к каркасной сети — только с подтверждением
    пользователя». Кратчайший путь — прямая между ближайшими точками
    границы участка и дороги (`shapely.ops.nearest_points`) — для «подъезда»
    (короткого локального примыкания, не маршрута через сеть) это и есть
    честный кратчайший путь, не эвристика."""
    if not roads:
        return None

    nearest_road = min(roads, key=lambda r: site_boundary.distance(r.geometry))
    site_point, road_point = nearest_points(site_boundary, nearest_road.geometry)
    from shapely.geometry import LineString

    path = LineString([site_point, road_point])

    crosses_obstacle = any(path.crosses(obstacle) or path.within(obstacle) for obstacle in obstacles)
    requires_confirmation = nearest_road.network == NETWORK_BACKBONE

    return AccessRoad(
        path=path, connects_to=nearest_road,
        requires_user_confirmation=requires_confirmation, crosses_obstacle=crosses_obstacle,
    )
