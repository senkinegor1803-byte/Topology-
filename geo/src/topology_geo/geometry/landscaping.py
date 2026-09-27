"""Ограждения, фонари и скамейки (Шаг 2.8, п. 2: «параметрические
генераторы: опоры, столбы, бордюры, ограждения, пролёты, фонари»; скамейки
— Шаг 2.9, действие «...скамейки»). Опоры/столбы — уже параметрические
(рельсы — Шаг 2.6 `geometry.rail`, ЛЭП — Шаг 2.7 `geometry.power`), бордюр
— Шаг 2.3 `geometry.streets`; здесь — оставшиеся типы, которых не было ни у
одного предыдущего шага.

`osm2pgsql` (Шаг 1.1) уже собирает `barrier=fence/wall` (линии),
`highway=street_lamp` и `amenity=bench` (точки) в таблицу `osm_landscaping`
— изменений импорта не потребовалось.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from topology_geo.selection.service import SiteFeature

FENCE = "fence"
WALL = "wall"

DEFAULT_FENCE_HEIGHT_M = 1.5
DEFAULT_WALL_HEIGHT_M = 2.0
WALL_THICKNESS_M = 0.2
FENCE_RAIL_WIDTH_M = 0.05
FENCE_POST_RADIUS_M = 0.05
# Типовой пролёт ограждения (расстояние между столбами) - действие Шага 2.8,
# п. 2 буквально называет «пролёты» отдельным пунктом; здесь это не
# отдельный генератор, а параметр расстановки столбов внутри генератора
# ограждения.
FENCE_POST_SPACING_M = 2.5

STREETLAMP_POLE_HEIGHT_M = 6.0
STREETLAMP_POLE_RADIUS_M = 0.08
STREETLAMP_HEAD_RADIUS_M = 0.15
STREETLAMP_HEAD_HEIGHT_M = 0.3

# Скамейка (Шаг 2.9, `amenity=bench`) - упрощённо один плоский короб
# (сиденье), без спинки/ножек по отдельности; типовые габариты, OSM почти
# никогда не даёт ориентацию скамейки (`direction`), поэтому короб всегда
# по осям координат - известное упрощение.
BENCH_LENGTH_M = 1.5
BENCH_DEPTH_M = 0.5
BENCH_SEAT_HEIGHT_M = 0.45


@dataclass(frozen=True)
class FenceSegment:
    osm_id: int
    kind: str  # FENCE | WALL
    ribbon: object  # тонкая полоса вдоль линии, локальные координаты
    height_m: float
    posts: tuple[tuple[float, float], ...]  # точки столбов вдоль линии (только для FENCE, для WALL - пусто)


@dataclass(frozen=True)
class StreetLamp:
    osm_id: int
    x: float
    y: float
    pole_height_m: float
    pole_radius_m: float
    head_radius_m: float
    head_height_m: float


@dataclass(frozen=True)
class Bench:
    osm_id: int
    x: float
    y: float
    length_m: float
    depth_m: float
    seat_height_m: float


def _as_lines(geom):
    if geom.geom_type.startswith("Multi"):
        return [g for g in geom.geoms if g.length > 0]
    return [geom] if geom.length > 0 else []


def _posts_along_line(line, spacing_m: float) -> tuple[tuple[float, float], ...]:
    length = line.length
    if length <= 0:
        return ()
    n_spans = max(1, math.ceil(length / spacing_m))
    return tuple(
        (point.x, point.y)
        for point in (line.interpolate(min(i * (length / n_spans), length)) for i in range(n_spans + 1))
    )


def build_fences(
    features: list[SiteFeature],
    *,
    fence_height_m: float = DEFAULT_FENCE_HEIGHT_M,
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M,
    post_spacing_m: float = FENCE_POST_SPACING_M,
) -> list[FenceSegment]:
    """Ограждения (`barrier=fence`) и стены (`barrier=wall`), Шаг 2.8, п. 2.

    Стена — сплошная тонкая лента на всю высоту (`extrude_polygon_mesh` в
    сборке IFC). Ограждение — упрощённо лёгкий верхний поручень (тонкая
    лента) ПЛЮС столбы через равный пролёт (`post_spacing_m`) по всей длине
    (без обрубка последнего пролёта — тот же приём, что и у расчётной
    расстановки опор ЛЭП, Шаг 2.7)."""
    result: list[FenceSegment] = []
    for feature in features:
        if feature.layer != "osm_landscaping":
            continue
        barrier = str(feature.raw_tags.get("barrier", "")).strip().lower()
        if barrier not in (FENCE, WALL):
            continue
        for line in _as_lines(feature.geometry):
            if barrier == WALL:
                ribbon = line.buffer(WALL_THICKNESS_M / 2, cap_style="flat")
                if ribbon.is_empty:
                    continue
                result.append(FenceSegment(osm_id=feature.osm_id, kind=WALL, ribbon=ribbon, height_m=wall_height_m, posts=()))
            else:
                ribbon = line.buffer(FENCE_RAIL_WIDTH_M / 2, cap_style="flat")
                if ribbon.is_empty:
                    continue
                posts = _posts_along_line(line, post_spacing_m)
                result.append(
                    FenceSegment(osm_id=feature.osm_id, kind=FENCE, ribbon=ribbon, height_m=fence_height_m, posts=posts)
                )
    return result


def build_streetlamps(features: list[SiteFeature]) -> list[StreetLamp]:
    """Фонари (`highway=street_lamp`, точки), Шаг 2.8, п. 2 - упрощённо
    столб постоянного сечения (`mesh_cylinder`) + небольшая головка
    светильника (тот же `mesh_cylinder` увеличенного радиуса сверху)."""
    result: list[StreetLamp] = []
    for feature in features:
        if feature.layer != "osm_landscaping" or feature.geometry.geom_type != "Point":
            continue
        if str(feature.raw_tags.get("highway", "")).strip().lower() != "street_lamp":
            continue
        result.append(
            StreetLamp(
                osm_id=feature.osm_id, x=feature.geometry.x, y=feature.geometry.y,
                pole_height_m=STREETLAMP_POLE_HEIGHT_M, pole_radius_m=STREETLAMP_POLE_RADIUS_M,
                head_radius_m=STREETLAMP_HEAD_RADIUS_M, head_height_m=STREETLAMP_HEAD_HEIGHT_M,
            )
        )
    return result


def build_benches(features: list[SiteFeature]) -> list[Bench]:
    """Скамейки (`amenity=bench`, точки), Шаг 2.9 - упрощённо один плоский
    короб типовых габаритов, без ориентации (см. докстринг модуля/
    `BENCH_*`)."""
    result: list[Bench] = []
    for feature in features:
        if feature.layer != "osm_landscaping" or feature.geometry.geom_type != "Point":
            continue
        if str(feature.raw_tags.get("amenity", "")).strip().lower() != "bench":
            continue
        result.append(
            Bench(
                osm_id=feature.osm_id, x=feature.geometry.x, y=feature.geometry.y,
                length_m=BENCH_LENGTH_M, depth_m=BENCH_DEPTH_M, seat_height_m=BENCH_SEAT_HEIGHT_M,
            )
        )
    return result
