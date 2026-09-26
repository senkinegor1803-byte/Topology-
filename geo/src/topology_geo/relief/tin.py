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

Фоновый рельеф — регулярная квадратная сетка («метод квадратных призм»
инженерной геодезии: каждая ячейка сетки — квадрат в плане, вершины несут
отметку, два треугольника Делоне на ячейку и есть развёртка призмы по
диагонали). Шаг сетки по умолчанию — 1 м (`background_step_m`): крупнее
нельзя — при бо́льшем шаге треугольники фоновой сетки становятся заметны
как грани («артефакты») при плоском затенении в вебвьюере (Шаг 1.9), и
крупная ячейка срезает точность точечных источников (топосъёмка) ещё до
TIN (см. `docs/relief.md`). После сэмплирования сетки — «многоструктурное»
сглаживание (`smooth_grid_elevations`): фон сглаживается 2D-скользящим
средним отдельно от дороги (та сглаживается своим 1D-скользящим средним
вдоль оси, `_smoothed_profile`) и отдельно от воды/зданий (их отметка —
константа по контуру, точная по построению, сглаживанию не подлежит:
это единственное, что гарантирует критерий «≤0.1 м» для дороги и ровную
площадку у здания/воды). Сглаживание фона не смещает наклонную плоскость
(среднее линейной функции по симметричному окну равно её значению в
центре) — оно убирает высокочастотный шум/ступеньки источника (например,
интерполяцию грубого TessaDEM на мелкую сетку), не искажая форму рельефа.

ВАЖНОЕ ОГРАНИЧЕНИЕ МАСШТАБА (измерено реальным прогоном, не оценка):
`build_site_tin` строит ОДИН несвязанный (не тайловый) `scipy.spatial.Delaunay`
на все точки участка сразу — при шаге фона 1 м это нормально на радиусе
Шага 1.5/MVP (500 м, `docs/plan.md`: ~784 тыс. точек, ~14 с, ~1.5 ГБ пиковой
памяти), но НЕ масштабируется на весь диапазон, который разрешает валидация
API (`radius_m` 500-3000 м, `api/schemas.py`): на 1,5 км (~7 млн точек)
процесс уже потреблял >12 ГБ и продолжал расти к моменту, когда пришлось
прервать прогон, чтобы не уронить среду; на 3 км (~28 млн точек) это
заведомо хуже. Дело не в качестве реализации (фон уже отдаётся сырыми numpy-
массивами, не `TinPoint`, дедуп не гоняется по фону вовсе) — сам `Delaunay`
на триангуляции такого размера требует память, которую разумно закладывать
только под настоящий кластер. Для полного радиуса 3 км в проекте уже есть
масштабируемое решение — Шаг 2.1 (`tiling/terrain.py`): тот же принцип
призм+сглаживания, но по тайлам 250×250 м (по 62,5 тыс. точек на тайл
независимо от общего радиуса модели, реально параллелится Celery). Поэтому
`build_site_tin` отказывает явной ошибкой при риске такого объёма
(`max_background_points`), а не падает по OOM где-то в середине —
падение по памяти в процессе Celery-воркера может увести с собой ДРУГИЕ,
не связанные с этой задачей, если воркер общий.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.ndimage import uniform_filter
from scipy.spatial import Delaunay, KDTree
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from topology_geo.relief.service import Grid

DEGENERATE_AREA_EPS_M2 = 1e-6

# Безопасный потолок точек фоновой сетки для НЕтайлового `Delaunay` одним
# куском (см. докстринг модуля, раздел «важное ограничение масштаба») — при
# шаге 1 м это соответствует радиусу участка ~800 м, с запасом выше
# протестированных 500 м MVP (Шаг 1.5/Этап 1) и с большим запасом ниже
# радиуса (~1,5 км), на котором реальный прогон уже уходил за 12 ГБ.
MAX_BACKGROUND_POINTS_UNTILED = 2_000_000


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


def sample_bilinear_grid(values: np.ndarray, grid: Grid, world_x: np.ndarray, world_y: np.ndarray) -> np.ndarray:
    """Векторизованный аналог `sample_bilinear` (те же условности, включая
    полупиксельную поправку) для массива точек сразу — точка вне растра даёт
    `nan` вместо `None`. Фоновая сетка на шаге 1 м даёт от сотен тысяч до
    десятков миллионов точек (Шаг 1.5 на радиусе 3 км, Шаг 2.1 по всем тайлам)
    — поточечный Python-цикл там на порядки медленнее векторного numpy."""
    col = (world_x - grid.transform.c) / grid.transform.a - 0.5
    row = (world_y - grid.transform.f) / grid.transform.e - 0.5
    inside = (col >= 0) & (row >= 0) & (col <= grid.width - 1) & (row <= grid.height - 1)

    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    c0c = np.clip(c0, 0, grid.width - 1)
    r0c = np.clip(r0, 0, grid.height - 1)
    c1 = np.clip(c0 + 1, 0, grid.width - 1)
    r1 = np.clip(r0 + 1, 0, grid.height - 1)
    fc = col - c0
    fr = row - r0

    v00, v01 = values[r0c, c0c], values[r0c, c1]
    v10, v11 = values[r1, c0c], values[r1, c1]
    top = v00 * (1 - fc) + v01 * fc
    bottom = v10 * (1 - fc) + v11 * fc
    result = top * (1 - fr) + bottom * fr

    out = np.full(world_x.shape, np.nan)
    out[inside] = result[inside]
    return out


def smooth_grid_elevations(elevations: np.ndarray, window_cells: int) -> np.ndarray:
    """Скользящее среднее по регулярной 2D-сетке отметок фона («многоструктурное
    сглаживание», п. 5 плана, для фонового рельефа — дорога/вода/здания
    сглаживаются отдельно от фона своими правилами, см. докстринг модуля).

    `nan` (нет покрытия DEM) исключается из среднего по соседям явно —
    наивный `uniform_filter` по массиву с `nan`, замененным на 0, размыл бы
    её как «низкую точку» в соседние валидные ячейки. Чётный `window_cells`
    смещает центр окна на полъячейки (несимметрично) — округляется вверх до
    нечётного, иначе сглаживание плоскости давало бы систематическую ошибку
    вместо точного среднего в центре окна.
    """
    if window_cells <= 1:
        return elevations
    if window_cells % 2 == 0:
        window_cells += 1

    valid = ~np.isnan(elevations)
    if not valid.any():
        return elevations

    filled = np.where(valid, elevations, 0.0)
    mean_filled = uniform_filter(filled, size=window_cells, mode="nearest")
    mean_valid = uniform_filter(valid.astype(np.float64), size=window_cells, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(mean_valid > 0, mean_filled / mean_valid, np.nan)
    return smoothed


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
    _kdtree: KDTree | None = field(default=None, repr=False, compare=False)

    def interpolate_z(self, x: float, y: float) -> float:
        """Барицентрическая интерполяция высоты в точке (x, y). Вне выпуклой
        оболочки TIN (например, угол ленты полосы у самой границы участка,
        буфер бордюра чуть шире врезанного в TIN коридора дороги) —
        экстраполяция отметкой БЛИЖАЙШЕЙ вершины TIN, а не абсолютный 0:
        последнее давало реальный, воспроизведённый баг — угол дорожного
        покрытия на настоящей отметке рельефа (например, 150 м) рисовался на
        Z=0, то есть «падал под землю» на всю высоту рельефа, а не просто
        на сантиметры (см. `docs/pavement.md`, найдено при добавлении
        предохранителя «дорога не уходит под рельеф»)."""
        simplex = self._delaunay.find_simplex(np.array([[x, y]]))[0]
        if simplex >= 0:
            tri = self.triangles[simplex]
            transform = self._delaunay.transform[simplex]
            delta = np.array([x, y]) - transform[2]
            bary = transform[:2].dot(delta)
            weights = np.array([bary[0], bary[1], 1 - bary.sum()])
            return float(np.dot(weights, self.vertices[tri, 2]))

        if self._kdtree is None:
            self._kdtree = KDTree(self.vertices[:, :2])
        _, nearest_idx = self._kdtree.query([x, y])
        return float(self.vertices[nearest_idx, 2])


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


def _fill_polygon_flat(poly: BaseGeometry, z: float, step: float, source: str) -> list[TinPoint]:
    """Точки на постоянной отметке `z` внутри `poly` (не только на контуре) —
    без них плоская площадка здания/уреза воды не гарантированно остаётся
    плоской: несвязанная (не constrained) триангуляция Делоне может
    «перепрыгнуть» пустую внутренность контура треугольником из соседних
    фоновых точек СНАРУЖИ, если они расположены достаточно плотно (вскрылось
    при переходе фона на шаг 1 м, Шаг 1.5 — раньше фон был реже контуров
    типичного здания и такой треугольник Делоне не строил)."""
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx + step / 2, maxx, step)
    ys = np.arange(miny + step / 2, maxy, step)
    if len(xs) == 0 or len(ys) == 0:
        return []
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    inside = shapely.contains_xy(poly, gx, gy)
    return [TinPoint(float(x), float(y), z, source) for x, y in zip(gx[inside], gy[inside])]


def _smoothed_background_arrays(
    radius_m: float,
    step: float,
    exclude: list[BaseGeometry],
    relief_values: np.ndarray,
    relief_grid: Grid,
    center_x: float,
    center_y: float,
    *,
    smoothing_window_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Точки фонового рельефа — квадратная сетка с шагом `step` (призмы,
    см. докстринг модуля), отметки сглажены `smooth_grid_elevations` ДО
    вырезания зон дорог/воды/зданий (сглаживание должно видеть реальный
    рельеф под ними, а не дыру — иначе съедет ближайший к границе фон).

    Возвращает СЫРЫЕ numpy-массивы (x, y, z), не `list[TinPoint]`: при шаге
    1 м у радиуса 3 км это ~28 млн точек — миллионы Python-объектов
    (`TinPoint`) и последующий поэлементный `_dedupe_points` (словарь на
    Python) на таком объёме исчерпывают память (проверено реальным прогоном
    — процесс убит OOM на радиусе 1,5 км). Фон не нуждается в дедупликации
    вовсе: это точная сетка без внутренних дублей, а с точками дорог/воды/
    зданий он не пересекается по построению (`excluded_mask` ниже вырезает
    там фон уже на этом шаге, геометрически) — `build_site_tin` склеивает
    результат с (небольшим) списком приоритетных точек напрямую массивами.

    Сетка для сглаживания строится ШИРЕ радиуса участка на половину окна
    (`pad`) и обрезается обратно после — иначе у точек ближе к радиусу, чем
    половина окна, `smooth_grid_elevations` (`mode="nearest"`) повторяла бы
    крайний семпл вместо реального соседнего пикселя растра, а на наклонной
    плоскости это даёт систематическое смещение (тот же эффект, что и
    несимметричное окно в `_smoothed_profile` — тут вместо линейной
    экстраполяции просто берётся реальный запас растра, `RELIEF_MARGIN_M`
    в `jobs/steps.py`, он для этого и больше окна сглаживания)."""
    n = int(np.ceil(radius_m / step))
    window_cells = max(round(smoothing_window_m / step), 1)
    if window_cells % 2 == 0:
        window_cells += 1
    pad = window_cells // 2

    idx = np.arange(-n - pad, n + pad + 1) * step
    padded_x, padded_y = np.meshgrid(idx, idx, indexing="ij")

    raw = sample_bilinear_grid(relief_values, relief_grid, center_x + padded_x, center_y + padded_y)
    smoothed_padded = smooth_grid_elevations(raw, window_cells)
    del raw

    hi = padded_x.shape[0] - pad
    local_x = padded_x[pad:hi, pad:hi]
    local_y = padded_y[pad:hi, pad:hi]
    smoothed = smoothed_padded[pad:hi, pad:hi]
    del padded_x, padded_y, smoothed_padded

    inside_circle = local_x * local_x + local_y * local_y <= radius_m * radius_m
    if exclude:
        # `intersects_xy`, не `contains_xy`: на шаге 1 м фоновая точка регулярно
        # попадает ТОЧНО на границу здания/воды (не только на угол) — строгий
        # `contains` (только внутренность) её не исключил бы, и точка с
        # реальной (не плоской) отметкой рельефа встала бы в триангуляцию
        # рядом с плоской площадкой здания/уреза воды, испортив её плоскостность.
        excluded_mask = shapely.intersects_xy(unary_union(exclude), local_x, local_y)
    else:
        excluded_mask = np.zeros_like(inside_circle, dtype=bool)
    keep = inside_circle & ~excluded_mask & ~np.isnan(smoothed)

    return local_x[keep], local_y[keep], smoothed[keep]


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


def build_profile_elevation_fn(
    line: LineString, elevation_fn, *, step: float, window_m: float
) -> Callable[[float, float], float]:
    """Функция отметки по точке `(x, y)`, основанная на сглаженном продольном
    профиле вдоль `line` (`_smoothed_profile`) — та же техника, что и у
    отметки дороги (Шаг 1.5, п. 2), применённая к реке/водоёму
    (`geometry/water.py`): в сечении, перпендикулярном течению, поверхность
    воды физически плоская (одна и та же отметка на обоих берегах), а вдоль
    течения меняется гладко — не единым минимумом по всему контуру сразу
    (последнее «роет траншею» для вытянутого водоёма/реки на склоне, см.
    `docs/water.md`). Точка проецируется на `line`, отметка — линейная
    интерполяция сглаженного профиля в этой проекции."""
    profile = _smoothed_profile(line, elevation_fn, step=step, window_m=window_m)
    distances = np.array([line.project(Point(x, y)) for x, y, _ in profile])
    zs = np.array([z for _, _, z in profile])
    order = np.argsort(distances)
    distances, zs = distances[order], zs[order]

    def level_fn(x: float, y: float) -> float:
        d = line.project(Point(x, y))
        return float(np.interp(d, distances, zs))

    return level_fn


def long_axis_line(polygon) -> tuple[LineString | None, float, float]:
    """Линия вдоль большего измерения вытянутого полигона — грубое
    приближение оси (реки/полосы дороги) через середины двух КОРОТКИХ сторон
    минимального охватывающего прямоугольника (`minimum_rotated_rectangle`);
    используется вместе с `build_profile_elevation_fn` там, где нужна
    отметка, гладко меняющаяся ВДОЛЬ вытянутого объекта, а не единая на весь
    контур (вытянутый водоём/русло — `geometry/water.py`; полоса дороги —
    `ifc/assemble.py`, Шаг 2.3, п. 5, «продольное сглаживание и поперечный
    уклон»). Возвращает `(ось, длина короткой стороны, длина длинной
    стороны)` — вызывающий код сам решает, достаточно ли вытянут контур,
    чтобы доверять оси (для почти квадратного контура она выбирается
    порядком вершин `minimum_rotated_rectangle` почти произвольно)."""
    mrr = polygon.minimum_rotated_rectangle
    if mrr.geom_type != "Polygon":
        return None, 0.0, 0.0
    coords = list(mrr.exterior.coords)[:-1]
    if len(coords) != 4:
        return None, 0.0, 0.0
    edges = [LineString([coords[i], coords[(i + 1) % 4]]) for i in range(4)]
    lengths = [edge.length for edge in edges]
    if max(lengths) <= 0:
        return None, 0.0, 0.0
    short_idx = min(range(4), key=lambda i: lengths[i])
    opposite_idx = (short_idx + 2) % 4
    long_idx = (short_idx + 1) % 4
    short_len, long_len = lengths[short_idx], lengths[long_idx]
    p1 = edges[short_idx].interpolate(0.5, normalized=True)
    p2 = edges[opposite_idx].interpolate(0.5, normalized=True)
    axis = LineString([p1, p2])
    if axis.length <= 0:
        return None, short_len, long_len
    return axis, short_len, long_len


def build_site_tin(
    relief_values: np.ndarray,
    relief_grid: Grid,
    center_x: float,
    center_y: float,
    radius_m: float,
    features: list,
    *,
    background_step_m: float = 1.0,
    background_smoothing_window_m: float = 5.0,
    road_step_m: float = 1.0,
    road_smoothing_window_m: float = 30.0,
    road_default_width_m: float = 6.0,
    max_background_points: int = MAX_BACKGROUND_POINTS_UNTILED,
) -> SiteTin:
    """Построить TIN участка радиуса `radius_m` (локальные координаты, центр
    (0,0)) из растра рельефа `relief_values`/`relief_grid` (мировые МСК-59
    координаты), врезав дороги/воду/здания из `features` (Шаг 1.4 — объекты
    уже в локальных координатах участка).

    `max_background_points` — предохранитель от OOM на большом радиусе (см.
    докстринг модуля): при `background_step_m=1.0` это ограничивает
    практический радиус этой (нетайловой) функции; для радиуса, на который
    рассчитан весь диапазон API (до 3 км), нужен тайловый путь Шага 2.1
    (`tiling.terrain.build_tile_terrain`), не эта функция напрямую.
    """
    n = int(np.ceil(radius_m / background_step_m))
    projected_background_points = (2 * n + 1) ** 2
    if projected_background_points > max_background_points:
        raise ValueError(
            f"радиус {radius_m:.0f} м с шагом фона {background_step_m:.2f} м даёт "
            f"~{projected_background_points:,} точек фона — выше безопасного предела "
            f"{max_background_points:,} для нетайлового TIN одним куском (реальный прогон "
            "на таком объёме уходит в OOM, см. докстринг модуля); для этого радиуса нужен "
            "тайловый путь Шага 2.1 (tiling.terrain.build_tile_terrain), а не build_site_tin "
            "напрямую"
        )

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
            points.extend(_fill_polygon_flat(poly, water_level, background_step_m, "water"))

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
            points.extend(_fill_polygon_flat(poly, pad_level, background_step_m, "building"))

    points = _dedupe_points(points)  # только приоритетные (дорога/вода/здание) — их немного, дедуп дешёвый

    bg_x, bg_y, bg_z = _smoothed_background_arrays(
        radius_m, background_step_m, exclude_zones, relief_values, relief_grid, center_x, center_y,
        smoothing_window_m=background_smoothing_window_m,
    )
    # Фон не дедуплицируется с приоритетными точками поэлементно (см.
    # докстринг `_smoothed_background_arrays` — на 28 млн точек это то, что
    # исчерпывает память): `excluded_mask` там уже вырезал фон из зон дорог/
    # воды/зданий геометрически, точного совпадения координат быть не должно.
    if points:
        priority_xy = np.array([(p.x, p.y) for p in points], dtype=np.float64)
        priority_z = np.array([p.z for p in points], dtype=np.float64)
        xy = np.vstack([priority_xy, np.column_stack([bg_x, bg_y])])
        z = np.concatenate([priority_z, bg_z])
    else:
        xy = np.column_stack([bg_x, bg_y])
        z = bg_z

    if len(xy) < 3:
        raise ValueError("недостаточно точек для построения TIN участка")

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
        target_z = target_z_fn(p.x, p.y)
        if target_z is None:
            continue
        max_dev = max(max_dev, abs(tin_z - target_z))
    return max_dev
