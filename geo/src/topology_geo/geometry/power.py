"""Электросети с опорами (Шаг 2.7): опоры/башни, провода цепной линией,
подстанции упрощёнными объёмами, охранная зона по классу напряжения.

`osm2pgsql` (Шаг 1.1) уже собирает `power=tower/pole/substation` (точки),
`power=line/minor_line` (линии) и `power=substation/plant` (полигоны, если
контур замкнут) в одну таблицу `osm_power` (`geometry`, смешанные типы —
тот же приём, что и `osm_vegetation`: точка-дерево + полигон-лес в одном
слое, различаются по `feature.geometry.geom_type`/`raw_tags["power"]`, а не
отдельными таблицами) — изменений импорта для этого шага не потребовалось.
`voltage=*` уже нормализуется в кВ на выборке (Шаг 1.4, `selection.normalize.
normalize_power`), здесь переиспользуется через `feature.attributes[
"voltage_kv"]`, повторный разбор тега не нужен.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import Point

from topology_geo.selection.service import SiteFeature

CONFIDENCE_FACT = "факт"
CONFIDENCE_DEFAULT = "умолчание"

# --- опоры/башни (п. 1-2) ---------------------------------------------------

DEFAULT_POLE_HEIGHT_M = 9.0
DEFAULT_TOWER_HEIGHT_M = 20.0
DEFAULT_POLE_MATERIAL = "wood"
# Радиус упрощённого цилиндра ствола опоры по материалу (`material=*`) —
# постоянного сечения (столб), в отличие от башни ниже.
POLE_RADIUS_M_BY_MATERIAL = {"wood": 0.12, "concrete": 0.15, "steel": 0.10}
# Решётчатая опора ЛЭП (`power=tower`) — упрощённо сужающийся к вершине
# силуэт (конус), БЕЗ реальных ферм/раскосов — то же упрощение уровня
# детализации, что и у опор контактной сети (Шаг 2.6, `geometry.rail`).
TOWER_RADIUS_BASE_M = 1.5
TOWER_RADIUS_TOP_M = 0.3

# Типовой пролёт ВЛ для расчётной расстановки (нет реальных точек опор,
# Шаг 2.7, п. 2) — условное значение, реальный пролёт зависит от класса
# напряжения и рельефа местности.
CALCULATED_POLE_SPACING_M = 60.0
# Опора считается «принадлежащей» линии, если её точка не дальше этого
# расстояния от оси (снос при цифровке OSM, а не отдельная опора).
POLE_SNAP_DISTANCE_M = 3.0


@dataclass(frozen=True)
class PoleTower:
    osm_id: int | None  # None — расчётная опора, не по факту точки OSM
    x: float
    y: float
    height_m: float
    height_confidence: str
    radius_base_m: float
    radius_top_m: float
    material: str
    series: str | None  # `design`/`structure` тег как есть, если был; иначе не подставляется
    voltage_kv: float | None
    source: str  # CONFIDENCE_FACT — реальная точка OSM; CONFIDENCE_DEFAULT — расчётная расстановка


def _non_empty_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_positive_float(value) -> float | None:
    if value is None:
        return None
    try:
        n = float(str(value).strip())
    except ValueError:
        return None
    return n if n > 0 else None


def _parse_positive_int(value) -> int | None:
    if value is None:
        return None
    try:
        n = int(float(str(value).strip()))
    except ValueError:
        return None
    return n if n > 0 else None


def _as_lines(geom):
    if geom.geom_type.startswith("Multi"):
        return [g for g in geom.geoms if g.length > 0]
    return [geom] if geom.length > 0 else []


def _pole_height_m(raw_tags: dict, power_value: str) -> tuple[float, str]:
    height = _parse_positive_float(raw_tags.get("height"))
    if height is not None:
        return height, CONFIDENCE_FACT
    return (DEFAULT_TOWER_HEIGHT_M if power_value == "tower" else DEFAULT_POLE_HEIGHT_M), CONFIDENCE_DEFAULT


def _pole_shape(power_value: str, raw_tags: dict) -> tuple[float, float, str]:
    """(radius_base_m, radius_top_m, материал) по типу опоры (п. 1)."""
    if power_value == "tower":
        material = _non_empty_str(raw_tags.get("material")) or "steel"
        return TOWER_RADIUS_BASE_M, TOWER_RADIUS_TOP_M, material
    material = (_non_empty_str(raw_tags.get("material")) or DEFAULT_POLE_MATERIAL).lower()
    radius = POLE_RADIUS_M_BY_MATERIAL.get(material, POLE_RADIUS_M_BY_MATERIAL[DEFAULT_POLE_MATERIAL])
    return radius, radius, material


def build_poles(features: list[SiteFeature]) -> list[PoleTower]:
    """Реальные опоры/башни по точкам OSM (`power=tower/pole`, п. 1-2)."""
    result: list[PoleTower] = []
    for feature in features:
        if feature.layer != "osm_power" or feature.geometry.geom_type != "Point":
            continue
        power_value = str(feature.raw_tags.get("power", "")).strip().lower()
        if power_value not in ("tower", "pole"):
            continue
        height_m, height_confidence = _pole_height_m(feature.raw_tags, power_value)
        radius_base, radius_top, material = _pole_shape(power_value, feature.raw_tags)
        series = _non_empty_str(feature.raw_tags.get("design")) or _non_empty_str(feature.raw_tags.get("structure"))
        result.append(
            PoleTower(
                osm_id=feature.osm_id, x=feature.geometry.x, y=feature.geometry.y,
                height_m=height_m, height_confidence=height_confidence,
                radius_base_m=radius_base, radius_top_m=radius_top, material=material, series=series,
                voltage_kv=feature.attributes.get("voltage_kv"), source=CONFIDENCE_FACT,
            )
        )
    return result


def place_calculated_poles(
    features: list[SiteFeature],
    real_poles: list[PoleTower],
    *,
    spacing_m: float = CALCULATED_POLE_SPACING_M,
    snap_distance_m: float = POLE_SNAP_DISTANCE_M,
) -> list[PoleTower]:
    """Расчётная расстановка опор по шагу (п. 2) — ТОЛЬКО для линий
    (`power=line/minor_line`), у которых нет НИ ОДНОЙ реальной опоры
    поблизости; если хотя бы одна найдена, линия считается обеспеченной
    реальными точками — расчётная расстановка это замена отсутствующим
    данным, а не дополнение к ним. Опоры распределены РАВНОМЕРНО по всей
    длине линии (с шагом, близким к `spacing_m`, но без обрубка последнего
    пролёта), а не с фиксированным шагом от начала."""
    result: list[PoleTower] = []
    default_radius = POLE_RADIUS_M_BY_MATERIAL[DEFAULT_POLE_MATERIAL]
    for feature in features:
        if feature.layer != "osm_power":
            continue
        power_value = str(feature.raw_tags.get("power", "")).strip().lower()
        if power_value not in ("line", "minor_line"):
            continue
        voltage_kv = feature.attributes.get("voltage_kv")
        for sub_line in _as_lines(feature.geometry):
            length = sub_line.length
            if length <= 0:
                continue
            has_real_pole = any(sub_line.distance(Point(p.x, p.y)) <= snap_distance_m for p in real_poles)
            if has_real_pole:
                continue
            n_spans = max(1, math.ceil(length / spacing_m))
            for i in range(n_spans + 1):
                distance = min(i * (length / n_spans), length)
                point = sub_line.interpolate(distance)
                result.append(
                    PoleTower(
                        osm_id=None, x=point.x, y=point.y,
                        height_m=DEFAULT_POLE_HEIGHT_M, height_confidence=CONFIDENCE_DEFAULT,
                        radius_base_m=default_radius, radius_top_m=default_radius,
                        material=DEFAULT_POLE_MATERIAL, series=None,
                        voltage_kv=voltage_kv, source=CONFIDENCE_DEFAULT,
                    )
                )
    return result


# --- провода: цепная линия между опорами (п. 3) -----------------------------

DEFAULT_CABLES = 3  # одна цепь, 3 фазы — нет тега `cables`, типовое ВЛ переменного тока
# Горизонтальный разнос соседних проводов в пролёте — упрощённо один ряд
# (не треугольная/вертикальная раскладка траверс реальной опоры).
PHASE_SPACING_M = 1.5
# Визуальная ширина ленты провода в IFC (`ifc.assemble.mesh_ribbon_along_path`)
# — тонкая полоса, не реальный диаметр провода (обычно 1-3 см).
WIRE_RIBBON_WIDTH_M = 0.05
# Провода подвешены не на самой вершине опоры, а на траверсах чуть ниже —
# упрощённо как доля высоты опоры.
WIRE_ATTACHMENT_RATIO = 0.9
# Прогиб провода — параболическое приближение цепной линии (стандартный
# инженерный приём при малом прогибе относительно пролёта, используется в
# реальном проектировании ВЛ), не точное гиперболическое решение и не
# расчёт по натяжению/температуре/гололёду; здесь единый типовой процент от
# длины пролёта для всех классов напряжения.
SAG_RATIO = 0.03
WIRE_SAMPLE_COUNT = 8


@dataclass(frozen=True)
class WireSpan:
    line_osm_id: int
    strand_index: int
    cables: int
    voltage_kv: float | None
    path: list[tuple[float, float, float]]  # сэмплированная цепная линия, локальные координаты


def _catenary_path(
    x0: float, y0: float, z0: float, x1: float, y1: float, z1: float,
    span_length: float, sag_ratio: float, n: int,
) -> list[tuple[float, float, float]]:
    sag = span_length * sag_ratio
    points: list[tuple[float, float, float]] = []
    for i in range(n + 1):
        t = i / n
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        z_linear = z0 + t * (z1 - z0)
        # максимум прогиба геометрически на середине пролёта (упрощённо —
        # у реальной цепной линии при разных отметках опор минимум смещён
        # к более низкой из них).
        z = z_linear - sag * 4 * t * (1 - t)
        points.append((x, y, z))
    return points


def build_wire_spans(
    features: list[SiteFeature],
    poles: list[PoleTower],
    terrain_elevation_fn,
    *,
    snap_distance_m: float = POLE_SNAP_DISTANCE_M,
    sag_ratio: float = SAG_RATIO,
    sample_count: int = WIRE_SAMPLE_COUNT,
) -> list[WireSpan]:
    """Провода между последовательными опорами вдоль линии (п. 3). Число
    проводов в пролёте — тег `cables` (все провода как есть, включая
    возможный трос грозозащиты), при отсутствии — `DEFAULT_CABLES`. Опоры
    (реальные и расчётные вперемешку, см. `place_calculated_poles`) для
    КОНКРЕТНОЙ линии находятся заново по расстоянию до её оси — то же
    геометрическое условие, что и при расчётной расстановке, поэтому
    расчётные опоры (стоящие точно на оси) и реальные (в пределах допуска)
    подбираются одинаково без отдельной привязки «опора -> линия»."""
    result: list[WireSpan] = []
    for feature in features:
        if feature.layer != "osm_power":
            continue
        power_value = str(feature.raw_tags.get("power", "")).strip().lower()
        if power_value not in ("line", "minor_line"):
            continue
        cables = _parse_positive_int(feature.raw_tags.get("cables")) or DEFAULT_CABLES
        for sub_line in _as_lines(feature.geometry):
            length = sub_line.length
            if length <= 0:
                continue
            on_line = [p for p in poles if sub_line.distance(Point(p.x, p.y)) <= snap_distance_m]
            if len(on_line) < 2:
                continue
            ordered = sorted(on_line, key=lambda p: sub_line.project(Point(p.x, p.y)))
            for a, b in zip(ordered, ordered[1:]):
                span_length = math.hypot(b.x - a.x, b.y - a.y)
                if span_length <= 0:
                    continue
                z_a = (terrain_elevation_fn(a.x, a.y) or 0.0) + a.height_m * WIRE_ATTACHMENT_RATIO
                z_b = (terrain_elevation_fn(b.x, b.y) or 0.0) + b.height_m * WIRE_ATTACHMENT_RATIO
                dx, dy = b.x - a.x, b.y - a.y
                norm = math.hypot(dx, dy)
                perp_x, perp_y = (-dy / norm, dx / norm) if norm > 0 else (0.0, 0.0)
                for strand in range(cables):
                    offset = (strand - (cables - 1) / 2) * PHASE_SPACING_M
                    path = _catenary_path(
                        a.x + perp_x * offset, a.y + perp_y * offset, z_a,
                        b.x + perp_x * offset, b.y + perp_y * offset, z_b,
                        span_length, sag_ratio, sample_count,
                    )
                    result.append(
                        WireSpan(
                            line_osm_id=feature.osm_id, strand_index=strand, cables=cables,
                            voltage_kv=feature.attributes.get("voltage_kv"), path=path,
                        )
                    )
    return result


# --- подстанции/ТП упрощённым объёмом (п. 4) --------------------------------

SUBSTATION_HEIGHT_M = 4.0


@dataclass(frozen=True)
class Substation:
    osm_id: int
    footprint: object  # shapely Polygon/MultiPolygon, локальные координаты
    base_z: float
    height_m: float
    kind: str  # substation | plant


def min_elevation_over_polygon(polygon, terrain_elevation_fn) -> float:
    polys = polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]
    zs = [terrain_elevation_fn(x, y) for poly in polys for x, y in poly.exterior.coords]
    zs = [z for z in zs if z is not None]
    return min(zs) if zs else 0.0


def build_substations(
    features: list[SiteFeature], terrain_elevation_fn, *, height_m: float = SUBSTATION_HEIGHT_M
) -> list[Substation]:
    """Подстанции/ТП упрощённым объёмом (п. 4) — только оконтуренные
    (полигон); точечные `power=substation` (только позиция, без площадки в
    OSM) объёма не имеют и честно пропускаются, не изобретается размер."""
    result: list[Substation] = []
    for feature in features:
        if feature.layer != "osm_power":
            continue
        power_value = str(feature.raw_tags.get("power", "")).strip().lower()
        if power_value not in ("substation", "plant"):
            continue
        geom = feature.geometry
        if geom.geom_type not in ("Polygon", "MultiPolygon") or geom.is_empty:
            continue
        base_z = min_elevation_over_polygon(geom, terrain_elevation_fn)
        result.append(Substation(osm_id=feature.osm_id, footprint=geom, base_z=base_z, height_m=height_m, kind=power_value))
    return result


# --- охранная зона по классу напряжения (п. 5) ------------------------------

# Полуширина охранной зоны ВЛ от крайнего провода (здесь — от оси линии,
# упрощённо) по классу напряжения — реальные нормативные значения (ПП РФ
# №160 от 24.02.2009 «О порядке установления охранных зон объектов
# электросетевого хозяйства...», приложение); верхняя граница диапазона
# включительно.
# Представительная высота объёма охранной зоны (Шаг 2.7, п. 5: «объём вокруг
# линии») — не выведена из фактических опор конкретной линии (это отдельный
# builder, без доступа к ним), типовое значение с запасом выше большинства
# опор/башен распределительных и низких магистральных классов напряжения.
SAFETY_ZONE_HEIGHT_M = 25.0

SAFETY_ZONE_HALF_WIDTH_M_BY_VOLTAGE_KV: list[tuple[float, float]] = [
    (1.0, 2.0),
    (20.0, 10.0),
    (35.0, 15.0),
    (110.0, 20.0),
    (220.0, 25.0),
    (500.0, 30.0),
    (750.0, 40.0),
    (1150.0, 55.0),
]


@dataclass(frozen=True)
class PowerSafetyZone:
    line_osm_id: int
    corridor: object  # shapely Polygon/MultiPolygon, локальные координаты
    voltage_kv: float
    half_width_m: float


def _safety_zone_half_width_m(voltage_kv: float | None) -> float | None:
    if voltage_kv is None:
        return None
    for upper_kv, half_width in SAFETY_ZONE_HALF_WIDTH_M_BY_VOLTAGE_KV:
        if voltage_kv <= upper_kv:
            return half_width
    return SAFETY_ZONE_HALF_WIDTH_M_BY_VOLTAGE_KV[-1][1]


def build_power_safety_zones(features: list[SiteFeature]) -> list[PowerSafetyZone]:
    """Охранная зона вокруг линии по классу напряжения (п. 5) — БЕЗ тега
    `voltage` класс не определить, зона честно не строится (тот же принцип,
    что и у проверки габарита моста без осей пересекаемых объектов, Шаг
    2.5, п. 3: не гадаем норматив вместо отсутствующих данных)."""
    result: list[PowerSafetyZone] = []
    for feature in features:
        if feature.layer != "osm_power":
            continue
        power_value = str(feature.raw_tags.get("power", "")).strip().lower()
        if power_value not in ("line", "minor_line"):
            continue
        voltage_kv = feature.attributes.get("voltage_kv")
        half_width = _safety_zone_half_width_m(voltage_kv)
        if half_width is None:
            continue
        for sub_line in _as_lines(feature.geometry):
            corridor = sub_line.buffer(half_width, cap_style="flat")
            if corridor.is_empty:
                continue
            result.append(
                PowerSafetyZone(
                    line_osm_id=feature.osm_id, corridor=corridor, voltage_kv=voltage_kv, half_width_m=half_width
                )
            )
    return result
