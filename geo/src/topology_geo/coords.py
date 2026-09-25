"""Пересчёт координат и высот проекта (Шаг 0.4 плана: docs/plan.md).

Системы, между которыми конвейер преобразует данные:

- источник плана OSM: географические координаты WGS-84 (EPSG:4326);
- проектная плоская система: МСК-59 (Пермский край), 3 трёхградусные зоны,
  эллипсоид Красовского;
- высоты топосъёмки / изысканий: Балтийская система (нормальные высоты);
- высоты TessaDEM: над геоидом EGM96 (приближённо эллипсоидальные для наших целей).

ВАЖНО про параметры МСК-59 ниже: это расчётные параметры по ГОСТ Р 51794-2008,
как их публикуют геодезические справочники и производители ГИС (mapbasic.ru/msk59,
terraingis.ru/msk-59.html) — рабочее приближение для разработки, а не официально
утверждённые Росреестром параметры зоны. Шаг 0.4 плана прямо требует получить
официальные параметры и проверить их на 5–10 контрольных точках (пункты ГГС, углы
зданий из топосъёмки) с критерием расхождения ≤ 0.1 м в плане и ≤ 0.05 м по высоте.
Это организационная задача (нужны официальный документ и полевые данные изыскателя,
роль ИЗ в плане) — AI-сессия не может её закрыть, но модуль ниже даёт код и функции
проверки, которые останется прогнать на реальных контрольных точках.

Источники параметров МСК-59 (см. также docs/coordinate-systems.md):
- https://mapbasic.ru/msk59
- https://terraingis.ru/msk-59.html
"""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import CRS, Transformer

WGS84 = CRS.from_epsg(4326)


@dataclass(frozen=True)
class MskZoneParams:
    """Параметры одной 3-градусной зоны МСК-59 (эллипсоид Красовского, Gauss-Kruger)."""

    zone: int
    lon0: float  # осевой меридиан зоны, град.
    false_easting: float  # X0, м
    false_northing: float  # Y0, м
    # 7-параметрический сдвиг Красовский -> WGS-84 (towgs84, формат PROJ: dx,dy,dz,rx,ry,rz,ds[ppm])
    towgs84: tuple[float, float, float, float, float, float, float]

    def to_proj4(self) -> str:
        dx, dy, dz, rx, ry, rz, ds = self.towgs84
        return (
            "+proj=tmerc +lat_0=0"
            f" +lon_0={self.lon0}"
            " +k=1"
            f" +x_0={self.false_easting}"
            f" +y_0={self.false_northing}"
            " +ellps=krass"
            f" +towgs84={dx},{dy},{dz},{rx},{ry},{rz},{ds}"
            " +units=m +no_defs"
        )


# Общий 7-параметрический сдвиг для СК-63 (Красовский) -> WGS-84, используемый во всех
# зонах МСК-59 в найденных справочниках.
_TOWGS84_SK63 = (23.57, -140.95, -79.8, 0.0, 0.35, 0.79, -0.22)

MSK59_ZONES: dict[int, MskZoneParams] = {
    1: MskZoneParams(zone=1, lon0=53.55, false_easting=1_250_000, false_northing=-5_914_743.504, towgs84=_TOWGS84_SK63),
    2: MskZoneParams(zone=2, lon0=56.55, false_easting=2_250_000, false_northing=-5_914_743.504, towgs84=_TOWGS84_SK63),
    3: MskZoneParams(zone=3, lon0=59.55, false_easting=3_250_000, false_northing=-5_914_743.504, towgs84=_TOWGS84_SK63),
}

# Границы зон по долготе: каждая зона покрывает lon0 +/- 1.5 градуса.
_ZONE_HALF_WIDTH_DEG = 1.5


def pick_msk59_zone(lon: float) -> int:
    """Определить номер зоны МСК-59 по долготе (град., WGS-84)."""
    best_zone = min(
        MSK59_ZONES.values(),
        key=lambda z: abs(lon - z.lon0),
    )
    if abs(lon - best_zone.lon0) > _ZONE_HALF_WIDTH_DEG + 0.5:
        raise ValueError(
            f"Долгота {lon} вне зон МСК-59 (Пермский край); проверьте координаты участка"
        )
    return best_zone.zone


def _transformer_for_zone(zone: int, *, inverse: bool) -> Transformer:
    params = MSK59_ZONES[zone]
    msk_crs = CRS.from_proj4(params.to_proj4())
    src, dst = (msk_crs, WGS84) if inverse else (WGS84, msk_crs)
    return Transformer.from_crs(src, dst, always_xy=True)


def wgs84_to_msk59(lon: float, lat: float, zone: int | None = None) -> tuple[float, float, int]:
    """Перевести (lon, lat) WGS-84 в плоские координаты (x, y) МСК-59.

    Возвращает (x, y, zone). Если zone не задан, выбирается по долготе.
    """
    if zone is None:
        zone = pick_msk59_zone(lon)
    x, y = _transformer_for_zone(zone, inverse=False).transform(lon, lat)
    return x, y, zone


def msk59_to_wgs84(x: float, y: float, zone: int) -> tuple[float, float]:
    """Перевести плоские координаты (x, y) МСК-59 зоны `zone` в (lon, lat) WGS-84."""
    lon, lat = _transformer_for_zone(zone, inverse=True).transform(x, y)
    return lon, lat


@dataclass(frozen=True)
class ControlPoint:
    """Контрольная точка для проверки/калибровки пересчёта (Шаг 0.4, п. 3)."""

    name: str
    lon: float
    lat: float
    msk_x: float | None = None
    msk_y: float | None = None
    h_source: float | None = None  # высота в исходной системе (например EGM96)
    h_target: float | None = None  # эталонная высота (например Балтийская)


@dataclass(frozen=True)
class PlanimetricCheckResult:
    name: str
    dx: float
    dy: float
    error_m: float


def verify_planimetric(points: list[ControlPoint], zone: int) -> list[PlanimetricCheckResult]:
    """Сверить пересчёт WGS-84 -> МСК-59 с эталонными координатами контрольных точек.

    Критерий приёмки Шага 0.4: error_m <= 0.1 м для каждой точки.
    """
    results = []
    for p in points:
        if p.msk_x is None or p.msk_y is None:
            raise ValueError(f"У контрольной точки {p.name!r} не заданы эталонные MSK-координаты")
        x, y, _ = wgs84_to_msk59(p.lon, p.lat, zone=zone)
        dx, dy = x - p.msk_x, y - p.msk_y
        results.append(PlanimetricCheckResult(p.name, dx, dy, (dx**2 + dy**2) ** 0.5))
    return results


@dataclass(frozen=True)
class HeightOffsetModel:
    """Локальная модель перехода h_source (например EGM96) -> h_target (Балтийская).

    Простейшая калибровка по контрольным точкам: константа (для <3 точек) либо
    наклонная плоскость offset + a*x + b*y (метод наименьших квадратов, для >=3 точек),
    построенная в локальных координатах относительно первой точки. Это ровно тот
    инструмент, который нужен для проверки на 5-10 контрольных точках по Шагу 0.4;
    без реальных данных изысканий (h_target) он не может быть откалиброван осмысленно.
    """

    lon0: float
    lat0: float
    offset: float
    grad_x: float = 0.0
    grad_y: float = 0.0

    def apply(self, h_source: float, lon: float, lat: float, zone: int) -> float:
        x, y, _ = wgs84_to_msk59(lon, lat, zone=zone)
        x0, y0, _ = wgs84_to_msk59(self.lon0, self.lat0, zone=zone)
        return h_source + self.offset + self.grad_x * (x - x0) + self.grad_y * (y - y0)


def fit_height_offset(points: list[ControlPoint], zone: int) -> HeightOffsetModel:
    """Откалибровать HeightOffsetModel по контрольным точкам с известными h_source и h_target."""
    calibrated = [p for p in points if p.h_source is not None and p.h_target is not None]
    if len(calibrated) < 1:
        raise ValueError("Нужна хотя бы одна контрольная точка с h_source и h_target")

    lon0, lat0 = calibrated[0].lon, calibrated[0].lat
    diffs = [(p, p.h_target - p.h_source) for p in calibrated]  # type: ignore[operator]

    if len(calibrated) < 3:
        mean_offset = sum(d for _, d in diffs) / len(diffs)
        return HeightOffsetModel(lon0=lon0, lat0=lat0, offset=mean_offset)

    # МНК-плоскость offset + grad_x*x + grad_y*y по локальным координатам.
    xs, ys, zs = [], [], []
    x0, y0, _ = wgs84_to_msk59(lon0, lat0, zone=zone)
    for p, d in diffs:
        x, y, _ = wgs84_to_msk59(p.lon, p.lat, zone=zone)
        xs.append(x - x0)
        ys.append(y - y0)
        zs.append(d)

    n = len(zs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(v * v for v in xs)
    syy = sum(v * v for v in ys)
    sxy = sum(a * b for a, b in zip(xs, ys))
    sxz = sum(a * b for a, b in zip(xs, zs))
    syz = sum(a * b for a, b in zip(ys, zs))
    sz = sum(zs)

    # Нормальные уравнения для [grad_x, grad_y, offset]^T.
    a = [
        [sxx, sxy, sx],
        [sxy, syy, sy],
        [sx, sy, float(n)],
    ]
    b = [sxz, syz, sz]
    grad_x, grad_y, offset = _solve_3x3(a, b)
    return HeightOffsetModel(lon0=lon0, lat0=lat0, offset=offset, grad_x=grad_x, grad_y=grad_y)


def _solve_3x3(a: list[list[float]], b: list[float]) -> tuple[float, float, float]:
    """Решение системы 3x3 методом Крамера (без numpy — модуль сознательно лёгкий)."""

    def det3(m: list[list[float]]) -> float:
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    d = det3(a)
    if abs(d) < 1e-9:
        raise ValueError("Контрольные точки коллинеарны/вырождены — нельзя откалибровать плоскость")

    result = []
    for col in range(3):
        m = [row[:] for row in a]
        for row in range(3):
            m[row][col] = b[row]
        result.append(det3(m) / d)
    return result[0], result[1], result[2]


@dataclass(frozen=True)
class HeightCheckResult:
    name: str
    error_m: float


def verify_heights(
    points: list[ControlPoint], model: HeightOffsetModel, zone: int
) -> list[HeightCheckResult]:
    """Критерий приёмки Шага 0.4: error_m <= 0.05 м для каждой контрольной точки."""
    results = []
    for p in points:
        if p.h_source is None or p.h_target is None:
            continue
        predicted = model.apply(p.h_source, p.lon, p.lat, zone=zone)
        results.append(HeightCheckResult(p.name, abs(predicted - p.h_target)))
    return results
