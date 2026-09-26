"""Вода: полигоны на уровне воды, реки/ручьи — ленты (Шаг 1.7, п. 2).

Два уточнения к базовой версии Шага 1.7 (найдены при разборе того, что
реально нужно для полномасштабных водных объектов):

1. **Сглаживание контура.** OSM часто оцифровывает берег/русло редкой
   ломаной — заметные острые углы, не похожие на настоящий берег. Контур
   (кольцо полигона, ось линии) сглаживается методом Чайкина (срезание
   углов, `_chaikin_smooth_closed`/`_chaikin_smooth_open`) ПЕРЕД тем, как из
   него строится геометрия (буфер ленты, ось для отметки уровня) — сглажена
   именно форма, а не просто число вершин.
2. **Уровень воды — не единый минимум по всему контуру для вытянутых
   водоёмов.** Для маленького КОМПАКТНОГО водоёма единый уровень (`min` по
   контуру) физически верен — поверхность воды там действительно плоская, и
   так и остаётся (`_long_axis_line` намеренно возвращает `None` ниже
   порога `WATER_AREA_ELONGATION_RATIO` — единственная ось компактного
   контура выбирается почти произвольно и с тем же успехом легла бы ПОПЕРЁК
   уклона, что хуже, а не лучше единого `min`, см. докстринг функции). Но
   для ВЫТЯНУТОГО водоёма/реки на склоне единый `min` по всему контуру
   «роет траншею» на одном конце — уровень должен меняться гладко ВДОЛЬ
   течения и быть одинаковым в сечении, перпендикулярном ему
   (`relief.tin.build_profile_elevation_fn`, та же техника, что и у
   продольного профиля дороги, Шаг 1.5 п. 2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from topology_geo.relief.tin import build_profile_elevation_fn
from topology_geo.selection.service import SiteFeature

DEFAULT_WATERWAY_WIDTH_M = 3.0
SMOOTHING_ITERATIONS = 2
WATER_LEVEL_PROFILE_STEP_M = 5.0
WATER_LEVEL_SMOOTHING_WINDOW_M = 50.0
WATER_AREA_ELONGATION_RATIO = 3.0  # длинная сторона / короткая — порог "вытянутый, не компактный"

ElevationFn = Callable[[float, float], float | None]
LevelFn = Callable[[float, float], float]


@dataclass(frozen=True)
class WaterArea:
    osm_id: int
    polygon: object  # shapely Polygon, локальные координаты (уже сглажен)
    level_z: float  # представительная отметка (min сглаженного профиля) - для Pset
    level_fn: LevelFn | None = None  # реальная (по точке) отметка поверхности воды


@dataclass(frozen=True)
class WaterwayRibbon:
    osm_id: int
    ribbon: object
    width_m: float
    level_fn: LevelFn | None = None


def _as_polygons(geom):
    if geom.geom_type.startswith("Multi"):
        return [g for g in geom.geoms if not g.is_empty]
    return [] if geom.is_empty else [geom]


def _as_lines(geom):
    if geom.geom_type.startswith("Multi"):
        return [g for g in geom.geoms if g.length > 0]
    return [geom] if geom.length > 0 else []


def _chaikin_cut(p0: tuple[float, float], p1: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
    r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
    return q, r


def _chaikin_smooth_closed(coords: list[tuple[float, float]], iterations: int) -> list[tuple[float, float]]:
    """Сглаживание ЗАМКНУТОГО кольца методом Чайкина (срезание углов)."""
    pts = coords[:-1] if len(coords) > 1 and coords[0] == coords[-1] else list(coords)
    if len(pts) < 3:
        return coords
    for _ in range(iterations):
        new_pts: list[tuple[float, float]] = []
        n = len(pts)
        for i in range(n):
            new_pts.extend(_chaikin_cut(pts[i], pts[(i + 1) % n]))
        pts = new_pts
    pts.append(pts[0])
    return pts


def _chaikin_smooth_open(coords: list[tuple[float, float]], iterations: int) -> list[tuple[float, float]]:
    """То же для ОТКРЫТОЙ линии — концы фиксированы, не срезаются: иначе
    место схождения нескольких рек/ручьёв в один узел (общая вершина в OSM)
    у каждого сегмента сместилось бы по-своему, и топология сети в месте
    слияния разошлась бы (видимый разрыв)."""
    pts = list(coords)
    if len(pts) < 3:
        return coords
    for _ in range(iterations):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            new_pts.extend(_chaikin_cut(pts[i], pts[i + 1]))
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


def _smooth_polygon(polygon: Polygon, iterations: int = SMOOTHING_ITERATIONS) -> Polygon:
    exterior = _chaikin_smooth_closed(list(polygon.exterior.coords), iterations)
    interiors = [_chaikin_smooth_closed(list(ring.coords), iterations) for ring in polygon.interiors]
    try:
        smoothed = Polygon(exterior, interiors)
    except ValueError:
        return polygon
    if not smoothed.is_valid:
        smoothed = smoothed.buffer(0)  # тот же приём, что `repair_footprint` у зданий (Шаг 1.6)
    return smoothed if smoothed.geom_type == "Polygon" and not smoothed.is_empty else polygon


def _smooth_line(line: LineString, iterations: int = SMOOTHING_ITERATIONS) -> LineString:
    coords = _chaikin_smooth_open(list(line.coords), iterations)
    smoothed = LineString(coords)
    return smoothed if smoothed.is_valid and not smoothed.is_empty else line


def _long_axis_line(polygon: Polygon) -> LineString | None:
    """Линия вдоль большего измерения ВЫТЯНУТОГО полигона (грубое приближение
    оси течения) — через середины двух КОРОТКИХ сторон минимального
    охватывающего прямоугольника (`minimum_rotated_rectangle`). `None` для
    компактного водоёма (соотношение сторон меньше
    `WATER_AREA_ELONGATION_RATIO`) — намеренно, не как приближение: единственная
    ось компактного/квадратного контура выбирается почти произвольно (порядок
    вершин `minimum_rotated_rectangle`) и с тем же успехом может лечь
    ПЕРЕК уклона, а не вдоль него, тогда профиль вдоль неё был бы ПОЧТИ
    ПОСТОЯННЫМ там, где сам уклон, наоборот, значим — то есть хуже, не лучше,
    единого `min` по всему контуру. Единый уровень для компактного водоёма к
    тому же физически верен: поверхность воды там действительно плоская."""
    mrr = polygon.minimum_rotated_rectangle
    if mrr.geom_type != "Polygon":
        return None
    coords = list(mrr.exterior.coords)[:-1]
    if len(coords) != 4:
        return None
    edges = [LineString([coords[i], coords[(i + 1) % 4]]) for i in range(4)]
    lengths = [edge.length for edge in edges]
    if max(lengths) <= 0:
        return None
    short_idx = min(range(4), key=lambda i: lengths[i])
    opposite_idx = (short_idx + 2) % 4
    long_idx = (short_idx + 1) % 4
    short_len, long_len = lengths[short_idx], lengths[long_idx]
    if short_len <= 0 or long_len / short_len < WATER_AREA_ELONGATION_RATIO:
        return None
    p1 = edges[short_idx].interpolate(0.5, normalized=True)
    p2 = edges[opposite_idx].interpolate(0.5, normalized=True)
    axis = LineString([p1, p2])
    return axis if axis.length > 0 else None


def _water_area_level(polygon: Polygon, terrain_elevation_fn: ElevationFn) -> tuple[LevelFn, float]:
    axis = _long_axis_line(polygon)
    if axis is None:
        elevations = [terrain_elevation_fn(x, y) for x, y in polygon.exterior.coords]
        elevations = [z for z in elevations if z is not None]
        level_z = min(elevations) if elevations else 0.0
        return (lambda x, y, z=level_z: z), level_z

    level_fn = build_profile_elevation_fn(
        axis, terrain_elevation_fn, step=WATER_LEVEL_PROFILE_STEP_M, window_m=WATER_LEVEL_SMOOTHING_WINDOW_M
    )
    boundary_levels = [level_fn(x, y) for x, y in polygon.exterior.coords]
    level_z = min(boundary_levels) if boundary_levels else 0.0
    return level_fn, level_z


def build_water_areas(
    features: list[SiteFeature], terrain_elevation_fn: ElevationFn
) -> list[WaterArea]:
    """Водоёмы (`osm_water_areas`): сглаженный контур, уровень воды —
    сглаженный продольный профиль вдоль длинной оси (см. докстринг модуля),
    не единый `min` по всему контуру."""
    result: list[WaterArea] = []
    for feature in features:
        if feature.layer != "osm_water_areas":
            continue
        for polygon in _as_polygons(feature.geometry):
            polygon = _smooth_polygon(polygon)
            level_fn, level_z = _water_area_level(polygon, terrain_elevation_fn)
            result.append(WaterArea(osm_id=feature.osm_id, polygon=polygon, level_z=level_z, level_fn=level_fn))
    return result


def _nearest_line_level_fn(lines: list[LineString], level_fns: list[LevelFn]) -> LevelFn:
    def _fn(x: float, y: float) -> float:
        point = Point(x, y)
        nearest_idx = min(range(len(lines)), key=lambda i: lines[i].distance(point))
        return level_fns[nearest_idx](x, y)

    return _fn


def build_waterway_ribbons(
    features: list[SiteFeature],
    terrain_elevation_fn: ElevationFn,
    *,
    width_m: float = DEFAULT_WATERWAY_WIDTH_M,
) -> list[WaterwayRibbon]:
    """Реки/ручьи (`osm_waterways`) -> сглаженная ось -> лента постоянной
    ширины (Этап 2 уточнит по расходу/классу), отметка — сглаженный
    продольный профиль вдоль оси (см. докстринг модуля)."""
    result: list[WaterwayRibbon] = []
    for feature in features:
        if feature.layer != "osm_waterways":
            continue
        lines = [_smooth_line(line) for line in _as_lines(feature.geometry)]
        lines = [line for line in lines if line.length > 0]
        if not lines:
            continue
        ribbon = unary_union([line.buffer(width_m / 2, cap_style="flat") for line in lines])
        if ribbon.is_empty:
            continue
        level_fns = [
            build_profile_elevation_fn(
                line, terrain_elevation_fn, step=WATER_LEVEL_PROFILE_STEP_M, window_m=WATER_LEVEL_SMOOTHING_WINDOW_M
            )
            for line in lines
        ]
        level_fn = level_fns[0] if len(lines) == 1 else _nearest_line_level_fn(lines, level_fns)
        result.append(WaterwayRibbon(osm_id=feature.osm_id, ribbon=ribbon, width_m=width_m, level_fn=level_fn))
    return result
