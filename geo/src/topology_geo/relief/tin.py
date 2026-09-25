"""TIN участка: адаптивная триангуляция рельефа, врезка дорог/воды/зданий
(Шаг 1.5).

Подход к «врезке» (п. 2 плана) — не полноценная constrained-триангуляция
(отдельной библиотеки для неё в проекте нет), а конструктивный приём с тем же
результатом: вдоль оси дороги, по границе полигона воды и по контуру здания
явно добавляются точки с УЖЕ ПРАВИЛЬНОЙ отметкой (сглаженный профиль дороги,
уровень уреза, отметка площадки), а фоновые точки рельефа внутри этих зон не
берутся вовсе. Результат триангулируется как один набор точек
(`scipy.spatial.Delaunay`), поэтому дорога/вода/здание становятся частью той
же сетки, а не накладываются поверх — что и требует критерий приёмки «дорога
не висит и не уходит глубже 0.1 м».

Самопересечений треугольников триангуляция Делоне не даёт по построению;
вырожденные (нулевой площади) грани возможны только от дублирующихся или
коллинеарных точек — от них избавляется дедупликация координат на входе,
контроль после триангуляции — `find_degenerate_triangles`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from topology_geo.relief.service import Grid

DEGENERATE_AREA_EPS_M2 = 1e-6


def sample_bilinear(values: np.ndarray, grid: Grid, world_x: float, world_y: float) -> float | None:
    """Билинейно прочитать высоту растра в мировой точке `(world_x, world_y)`.

    Возвращает None, если точка вне растра. `grid.transform` — угловая
    (rasterio-)конвенция: `transform * (col, row)` даёт верхний левый угол
    пикселя, а не его центр, поэтому здесь вычитается 0.5 — иначе выборка
    систематически смещена на полпикселя относительно значений `values`
    (значение `values[row, col]` относится к ЦЕНТРУ пикселя).
    """
    col = (world_x - grid.transform.c) / grid.transform.a - 0.5
    row = (world_y - grid.transform.f) / grid.transform.e - 0.5
    if col < 0 or row < 0 or col > grid.width - 1 or row > grid.height - 1:
        return None

    c0, r0 = int(np.floor(col)), int(np.floor(row))
    c1, r1 = min(c0 + 1, grid.width - 1), min(r0 + 1, grid.height - 1)
    fc, fr = col - c0, row - r0

    v00, v01 = values[r0, c0], values[r0, c1]
    v10, v11 = values[r1, c0], values[r1, c1]
    top = v00 * (1 - fc) + v01 * fc
    bottom = v10 * (1 - fc) + v11 * fc
    return float(top * (1 - fr) + bottom * fr)


@dataclass(frozen=True)
class TinPoint:
    x: float
    y: float
    z: float
    source: str  # "terrain" | "road" | "water" | "building"


@dataclass
class SiteTin:
    vertices: np.ndarray  # (N, 3): x, y, z в локальных координатах участка
    triangles: np.ndarray  # (M, 3) индексы вершин
    _delaunay: Delaunay

    def interpolate_z(self, x: float, y: float) -> float | None:
        """Барицентрическая интерполяция высоты в точке (x, y). None, если
        точка вне выпуклой оболочки TIN."""
        simplex = self._delaunay.find_simplex(np.array([[x, y]]))[0]
        if simplex < 0:
            return None
        tri = self.triangles[simplex]
        transform = self._delaunay.transform[simplex]
        delta = np.array([x, y]) - transform[2]
        bary = transform[:2].dot(delta)
        weights = np.array([bary[0], bary[1], 1 - bary.sum()])
        return float(np.dot(weights, self.vertices[tri, 2]))


def _dedupe_points(points: list[TinPoint], tol: float = 0.01) -> list[TinPoint]:
    """Убрать точки, совпадающие (в пределах `tol` м) с уже добавленной —
    источник таких дублей: точки перегородок разных объектов встретились в
    одном месте. Приоритет источника: building > water > road > terrain
    (важнее не потерять явную отметку здания/воды/дороги)."""
    priority = {"building": 3, "water": 2, "road": 1, "terrain": 0}
    kept: list[TinPoint] = []
    grid_index: dict[tuple[int, int], list[int]] = {}

    for p in points:
        key = (round(p.x / tol), round(p.y / tol))
        collided_idx = None
        for idx in grid_index.get(key, []):
            other = kept[idx]
            if abs(other.x - p.x) <= tol and abs(other.y - p.y) <= tol:
                collided_idx = idx
                break
        if collided_idx is None:
            grid_index.setdefault(key, []).append(len(kept))
            kept.append(p)
        elif priority[p.source] > priority[kept[collided_idx].source]:
            kept[collided_idx] = p

    return kept


def _background_grid(radius_m: float, step: float, exclude: list[BaseGeometry]) -> list[tuple[float, float]]:
    n = int(np.ceil(radius_m / step))
    pts = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x, y = i * step, j * step
            if x * x + y * y > radius_m * radius_m:
                continue
            point = Point(x, y)
            if any(zone.contains(point) for zone in exclude):
                continue
            pts.append((x, y))
    return pts


def _smoothed_profile(line: LineString, elevation_fn, *, step: float, window_m: float) -> list[tuple[float, float, float]]:
    """Точки вдоль `line` с продольно сглаженной отметкой (скользящее среднее
    исходного рельефа вдоль оси — п. 2 плана, "продольный - сглаживание по оси")."""
    length = line.length
    n = max(int(length / step), 1)
    distances = np.linspace(0, length, n + 1)
    raw_z = []
    for d in distances:
        p = line.interpolate(d)
        z = elevation_fn(p.x, p.y)
        raw_z.append(z if z is not None else np.nan)
    raw_z = np.array(raw_z, dtype=float)
    if np.isnan(raw_z).all():
        raw_z[:] = 0.0
    else:
        nan_mask = np.isnan(raw_z)
        raw_z[nan_mask] = np.nanmean(raw_z)

    half_window = max(int(window_m / step / 2), 1)
    # Продлить массив по краям линейной экстраполяцией перед усреднением -
    # иначе несимметричное окно на границах линии даёт систематическое
    # смещение сглаженного значения (на линейном уклоне: тем больше, чем
    # ближе к краю), а не просто более широкий разброс.
    if len(raw_z) >= 2:
        left_slope = raw_z[1] - raw_z[0]
        right_slope = raw_z[-1] - raw_z[-2]
    else:
        left_slope = right_slope = 0.0
    left_pad = raw_z[0] - left_slope * np.arange(half_window, 0, -1)
    right_pad = raw_z[-1] + right_slope * np.arange(1, half_window + 1)
    padded = np.concatenate([left_pad, raw_z, right_pad])

    smoothed = np.array(
        [padded[i: i + 2 * half_window + 1].mean() for i in range(len(raw_z))]
    )

    result = []
    for d, z in zip(distances, smoothed):
        p = line.interpolate(d)
        result.append((p.x, p.y, float(z)))
    return result


def build_site_tin(
    relief_values: np.ndarray,
    relief_grid: Grid,
    center_x: float,
    center_y: float,
    radius_m: float,
    features: list,
    *,
    background_step_m: float = 25.0,
    road_step_m: float = 5.0,
    road_smoothing_window_m: float = 30.0,
    road_default_width_m: float = 6.0,
) -> SiteTin:
    """Построить TIN участка радиуса `radius_m` (локальные координаты, центр
    (0,0)) из растра рельефа `relief_values`/`relief_grid` (мировые МСК-59
    координаты), врезав дороги/воду/здания из `features` (Шаг 1.4 — объекты
    уже в локальных координатах участка).
    """

    def elevation_fn(local_x: float, local_y: float) -> float | None:
        return sample_bilinear(relief_values, relief_grid, center_x + local_x, center_y + local_y)

    roads = [f for f in features if f.layer == "osm_roads"]
    water = [f for f in features if f.layer == "osm_water_areas"]
    buildings = [f for f in features if f.layer == "osm_buildings"]

    exclude_zones: list[BaseGeometry] = []
    points: list[TinPoint] = []

    for feature in roads:
        geom = feature.geometry
        lines = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        width = feature.attributes.get("lanes") and feature.attributes["lanes"] * 3.0 or road_default_width_m
        for line in lines:
            if line.length == 0:
                continue
            profile = _smoothed_profile(
                line, elevation_fn, step=road_step_m, window_m=road_smoothing_window_m
            )
            corridor = line.buffer(width / 2, cap_style="flat")
            exclude_zones.append(corridor)
            for x, y, z in profile:
                points.append(TinPoint(x, y, z, "road"))
                offset_dir_x, offset_dir_y = _perp_offset(line, x, y)
                points.append(TinPoint(x + offset_dir_x * width / 2, y + offset_dir_y * width / 2, z, "road"))
                points.append(TinPoint(x - offset_dir_x * width / 2, y - offset_dir_y * width / 2, z, "road"))

    for feature in water:
        geom = feature.geometry
        polys = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for poly in polys:
            if poly.is_empty:
                continue
            exclude_zones.append(poly)
            levels = [elevation_fn(x, y) for x, y in poly.exterior.coords]
            levels = [v for v in levels if v is not None]
            water_level = min(levels) if levels else 0.0
            for x, y in poly.exterior.coords:
                points.append(TinPoint(x, y, water_level, "water"))

    for feature in buildings:
        geom = feature.geometry
        polys = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for poly in polys:
            if poly.is_empty:
                continue
            exclude_zones.append(poly)
            levels = [elevation_fn(x, y) for x, y in poly.exterior.coords]
            levels = [v for v in levels if v is not None]
            pad_level = min(levels) if levels else 0.0
            for x, y in poly.exterior.coords:
                points.append(TinPoint(x, y, pad_level, "building"))

    for x, y in _background_grid(radius_m, background_step_m, exclude_zones):
        z = elevation_fn(x, y)
        if z is not None:
            points.append(TinPoint(x, y, z, "terrain"))

    points = _dedupe_points(points)
    if len(points) < 3:
        raise ValueError("недостаточно точек для построения TIN участка")

    xy = np.array([(p.x, p.y) for p in points])
    z = np.array([p.z for p in points])
    delaunay = Delaunay(xy)
    vertices = np.column_stack([xy, z])

    return SiteTin(vertices=vertices, triangles=delaunay.simplices, _delaunay=delaunay)


def _perp_offset(line: LineString, x: float, y: float) -> tuple[float, float]:
    """Единичный вектор, перпендикулярный касательной линии `line` в точке
    (используется, чтобы поставить точки по краям проезжей части)."""
    d = 0.5
    p_before = line.interpolate(max(line.project(Point(x, y)) - d, 0))
    p_after = line.interpolate(min(line.project(Point(x, y)) + d, line.length))
    dx, dy = p_after.x - p_before.x, p_after.y - p_before.y
    length = (dx**2 + dy**2) ** 0.5
    if length == 0:
        return 0.0, 0.0
    return -dy / length, dx / length


def find_degenerate_triangles(tin: SiteTin, *, eps_m2: float = DEGENERATE_AREA_EPS_M2) -> list[int]:
    """Индексы треугольников с площадью <= `eps_m2` (Шаг 1.5, п. 3)."""
    degenerate = []
    for i, tri in enumerate(tin.triangles):
        p0, p1, p2 = tin.vertices[tri, :2]
        area = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])) / 2
        if area <= eps_m2:
            degenerate.append(i)
    return degenerate


def max_deviation_along_line(tin: SiteTin, line: LineString, target_z_fn, *, step: float = 2.0) -> float:
    """Наибольшее |TIN(x,y) - target_z(x,y)| вдоль `line` — числовая проверка
    критерия Шага 1.5: дорога не должна отклоняться от TIN больше чем на 0.1 м.
    """
    length = line.length
    n = max(int(length / step), 1)
    max_dev = 0.0
    for d in np.linspace(0, length, n + 1):
        p = line.interpolate(d)
        tin_z = tin.interpolate_z(p.x, p.y)
        if tin_z is None:
            continue
        target_z = target_z_fn(p.x, p.y)
        if target_z is None:
            continue
        max_dev = max(max_dev, abs(tin_z - target_z))
    return max_dev
