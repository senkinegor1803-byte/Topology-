"""Железная дорога и трамвай (Шаг 1.7, п. 3; уточнения и платформы/переезды/
опоры контактной сети — Шаг 2.6).

Ширина насыпи (балластной призмы) больше не единая на все пути (плоская
`BALLAST_WIDTH_M` Шага 1.7) — считается по числу путей (`tracks=*`) и колее
(`gauge=*`, мм), см. `compute_ballast_width_m`. Платформы (`railway=platform`)
и переезды (`railway=level_crossing`/`crossing`) — из отдельных таблиц
`osm_railway_platforms`/`osm_railway_crossings` (`osm2pgsql/style.lua`).
Опоры контактной сети — упрощённая регулярная расстановка вдоль
электрифицированных путей (`electrified=*`), без деления на консоли/порталы.

Шпалы и рельсы «в ближнем кольце» (индивидуальная геометрия LOD, требующая
интеграции с кольцами Шага 2.1) в этом проходе НЕ реализованы — см.
`docs/rail.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from topology_geo.selection.service import SiteFeature

CONFIDENCE_FACT = "факт"
CONFIDENCE_DEFAULT = "умолчание"

DEFAULT_RAIL_TYPE = "rail"

# Обочина насыпи с каждой стороны колеи (упрощённое типовое сечение
# балластной призмы, без деления по классу пути/нагрузке).
BALLAST_SHOULDER_M = 1.24
# Междупутье (расстояние между осями соседних путей) при числе путей > 1 —
# типовое значение для прямых участков (СП 119.13330, диапазон ~4,1-4,5 м).
TRACK_SPACING_M = 4.5
# Колея по умолчанию (тег `gauge` отсутствует/некорректен) — российская
# широкая колея 1520 мм; единое значение для rail/tram/light_rail в этом
# проекте — упрощение, отдельного справочника колеи по типу пути не ведём.
DEFAULT_GAUGE_M = 1.520
# Прежнее плоское значение Шага 1.7 — при умолчаниях (1 путь, колея 1520 мм)
# новая формула даёт ровно это же число (1,520 + 2*1,24 = 4,0), совпадение не
# случайное: подобрано, чтобы не менять поведение по умолчанию.
BALLAST_WIDTH_M = DEFAULT_GAUGE_M + 2 * BALLAST_SHOULDER_M


@dataclass(frozen=True)
class RailRibbon:
    osm_id: int
    ballast: object  # shapely Polygon/MultiPolygon, локальные координаты
    rail_type: str  # rail | tram | light_rail и т.п. (тег railway=*)
    width_m: float = BALLAST_WIDTH_M  # ширина насыпи (Шаг 2.6): по числу путей и колее
    tracks: int = 1
    gauge_m: float = DEFAULT_GAUGE_M
    width_confidence: str = CONFIDENCE_DEFAULT
    # Ось пути (Шаг 2.5, п. 3: проверка габарита моста над нижележащими
    # путями нужна ось для точки пересечения) - только для ОДНОГО цельного
    # сегмента линии, тот же принцип, что и у `RoadRibbon.axis` (Шаг 2.4).
    axis: LineString | None = None


def _as_lines(geom):
    if geom.geom_type.startswith("Multi"):
        return [g for g in geom.geoms if g.length > 0]
    return [geom] if geom.length > 0 else []


def _parse_gauge_m(raw_tags: dict) -> tuple[float, str]:
    """Колея из тега `gauge` (в мм) -> метры; при отсутствии/некорректном
    значении - `DEFAULT_GAUGE_M`, `CONFIDENCE_DEFAULT`."""
    gauge_tag = raw_tags.get("gauge")
    if gauge_tag:
        try:
            gauge_mm = float(str(gauge_tag).strip())
        except ValueError:
            gauge_mm = None
        if gauge_mm is not None and gauge_mm > 0:
            return gauge_mm / 1000.0, CONFIDENCE_FACT
    return DEFAULT_GAUGE_M, CONFIDENCE_DEFAULT


def _parse_tracks(raw_tags: dict) -> tuple[int, str]:
    """Число путей из тега `tracks`; при отсутствии/некорректном значении -
    один путь, `CONFIDENCE_DEFAULT`."""
    tracks_tag = raw_tags.get("tracks")
    if tracks_tag:
        try:
            n = int(float(str(tracks_tag).strip()))
        except ValueError:
            n = None
        if n is not None and n > 0:
            return n, CONFIDENCE_FACT
    return 1, CONFIDENCE_DEFAULT


def compute_ballast_width_m(raw_tags: dict) -> tuple[float, int, float, str]:
    """Ширина насыпи по числу путей и колее (Шаг 2.6): один путь занимает
    `колея + 2*обочина`, каждый следующий добавляет типовое междупутье
    `TRACK_SPACING_M` (упрощённо - реальное междупутье растёт на кривых и
    может быть индивидуальным для каждой пары путей, здесь оно одно на всех).
    Возвращает (ширина_м, число_путей, колея_м, confidence — «факт», если
    хотя бы один из тегов `tracks`/`gauge` присутствовал и разобрался)."""
    gauge_m, gauge_confidence = _parse_gauge_m(raw_tags)
    tracks, tracks_confidence = _parse_tracks(raw_tags)
    per_track_width = gauge_m + 2 * BALLAST_SHOULDER_M
    total_width = per_track_width + (tracks - 1) * TRACK_SPACING_M
    confidence = (
        CONFIDENCE_FACT if CONFIDENCE_FACT in (gauge_confidence, tracks_confidence) else CONFIDENCE_DEFAULT
    )
    return total_width, tracks, gauge_m, confidence


def build_rail_ribbons(features: list[SiteFeature]) -> list[RailRibbon]:
    result: list[RailRibbon] = []
    for feature in features:
        if feature.layer != "osm_railways":
            continue
        lines = _as_lines(feature.geometry)
        if not lines:
            continue
        width_m, tracks, gauge_m, width_confidence = compute_ballast_width_m(feature.raw_tags)
        ballast = unary_union([line.buffer(width_m / 2, cap_style="flat") for line in lines])
        if ballast.is_empty:
            continue
        rail_type = feature.raw_tags.get("railway") or DEFAULT_RAIL_TYPE
        result.append(
            RailRibbon(
                osm_id=feature.osm_id, ballast=ballast, rail_type=rail_type,
                width_m=width_m, tracks=tracks, gauge_m=gauge_m, width_confidence=width_confidence,
                axis=lines[0] if len(lines) == 1 else None,
            )
        )
    return result


# --- платформы (railway=platform, Шаг 2.6) ---------------------------------

# Типовая пассажирская платформа для случая, когда контур в OSM не замкнут
# (платформа размечена одной линией вдоль пути, не полигоном) - ширина
# упрощённо-типовая, без деления на низкую/высокую или по классу станции.
PLATFORM_LINE_WIDTH_M = 3.0
# Высота платформы над уровнем земли/головки рельса - упрощённо, без деления
# на низкую (~200 мм) и высокую (~550 мм над УГР) платформу по типу состава.
PLATFORM_HEIGHT_M = 0.3


@dataclass(frozen=True)
class PlatformArea:
    osm_id: int
    footprint: object  # shapely Polygon/MultiPolygon, локальные координаты
    height_m: float


def build_platform_areas(features: list[SiteFeature]) -> list[PlatformArea]:
    """Пассажирские платформы (`osm_railway_platforms`, `railway=platform`,
    Шаг 2.6). В OSM платформа встречается и полигоном (замкнутый контур
    посадочной зоны), и одной линией вдоль пути - для линии ширина берётся
    упрощённо (`PLATFORM_LINE_WIDTH_M`), т.к. сама линия площади не имеет."""
    result: list[PlatformArea] = []
    for feature in features:
        if feature.layer != "osm_railway_platforms":
            continue
        geom = feature.geometry
        if geom.geom_type in ("Polygon", "MultiPolygon") and not geom.is_empty:
            footprint = geom
        else:
            lines = _as_lines(geom)
            if not lines:
                continue
            footprint = unary_union([line.buffer(PLATFORM_LINE_WIDTH_M / 2, cap_style="flat") for line in lines])
            if footprint.is_empty:
                continue
        result.append(PlatformArea(osm_id=feature.osm_id, footprint=footprint, height_m=PLATFORM_HEIGHT_M))
    return result


# --- переезды (railway=level_crossing/crossing, Шаг 2.6) -------------------


@dataclass(frozen=True)
class LevelCrossing:
    osm_id: int
    x: float
    y: float
    crossing_type: str  # level_crossing (со шлагбаумом/сигнализацией) | crossing (пешеходный)
    size_m: float  # сторона упрощённого квадратного пятна разметки переезда


def _nearest_rail_width(x: float, y: float, rail_ribbons: list[RailRibbon]) -> float:
    if not rail_ribbons:
        return BALLAST_WIDTH_M
    point = Point(x, y)
    nearest = min(rail_ribbons, key=lambda r: r.ballast.distance(point))
    return nearest.width_m


def build_level_crossings(features: list[SiteFeature], rail_ribbons: list[RailRibbon]) -> list[LevelCrossing]:
    """Переезды и пешеходные переходы через пути (`osm_railway_crossings`,
    `railway=level_crossing`/`crossing`, Шаг 2.6) - точка на пути, где её
    пересекает дорога или пешеходная дорожка. Представлены упрощённо
    квадратным пятном разметки со стороной, равной ширине насыпи БЛИЖАЙШЕГО
    пути в этой точке - точная геометрия пересекающей дороги не
    восстанавливается (это разметка факта переезда, а не проезжей части)."""
    result: list[LevelCrossing] = []
    for feature in features:
        if feature.layer != "osm_railway_crossings" or feature.geometry.geom_type != "Point":
            continue
        crossing_type = feature.raw_tags.get("railway") or "level_crossing"
        point = feature.geometry
        size_m = _nearest_rail_width(point.x, point.y, rail_ribbons)
        result.append(
            LevelCrossing(osm_id=feature.osm_id, x=point.x, y=point.y, crossing_type=crossing_type, size_m=size_m)
        )
    return result


# --- опоры контактной сети (electrified=*, Шаг 2.6) ------------------------

# Типовой шаг опор контактной сети на прямых участках - упрощённо, без учёта
# кривых (там шаг меньше) и ветровой нагрузки.
POLE_SPACING_M = 50.0
# Габарит приближения опор от оси пути - упрощённо один борт (СТН Ц-01-95,
# типовое значение для опор на насыпи вне кривых), без порталов/консолей на
# многопутных участках.
POLE_OFFSET_FROM_AXIS_M = 3.1
POLE_HEIGHT_M = 9.0
POLE_RADIUS_M = 0.15


@dataclass(frozen=True)
class CatenaryPole:
    osm_id: int  # исходный way пути, вдоль которого стоит опора
    x: float
    y: float
    index: int  # номер опоры вдоль пути - для уникального имени продукта IFC


def _is_electrified(raw_tags: dict) -> bool:
    value = str(raw_tags.get("electrified", "")).strip().lower()
    return value not in ("", "no")


def place_catenary_poles(
    features: list[SiteFeature],
    *,
    spacing_m: float = POLE_SPACING_M,
    offset_m: float = POLE_OFFSET_FROM_AXIS_M,
) -> list[CatenaryPole]:
    """Опоры контактной сети упрощённо (Шаг 2.6) - только для
    электрифицированных путей (`electrified` не пусто и не `no`), с
    регулярным шагом вдоль оси, с одной стороны от пути (без чередования
    сторон, порталов на многопутных участках и без учёта кривизны - честное
    упрощение, не полная модель контактной сети)."""
    result: list[CatenaryPole] = []
    for feature in features:
        if feature.layer != "osm_railways" or not _is_electrified(feature.raw_tags):
            continue
        for line in _as_lines(feature.geometry):
            length = line.length
            if length <= 0:
                continue
            n_poles = max(1, int(length // spacing_m) + 1)
            for i in range(n_poles):
                distance = min(i * spacing_m, length)
                point = line.interpolate(distance)
                ahead = line.interpolate(min(distance + 0.1, length))
                dx, dy = ahead.x - point.x, ahead.y - point.y
                norm = math.hypot(dx, dy)
                if norm == 0:
                    continue
                perp_x, perp_y = -dy / norm, dx / norm
                result.append(
                    CatenaryPole(
                        osm_id=feature.osm_id,
                        x=point.x + perp_x * offset_m,
                        y=point.y + perp_y * offset_m,
                        index=i,
                    )
                )
    return result
