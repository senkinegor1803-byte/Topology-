"""Формы крыш зданий (Шаг 2.2, п. 1-2).

Геометрические соглашения — не придуманы, а взяты из практики реальных
рендереров OSM-3D (OSM2World и совместимых, спецификация
`Simple3DBuildingsV1` на вики OpenStreetMap):

- `height` — ПОЛНАЯ высота здания (до конька/шатра); высота стен (карниз) =
  `height - roof:height` (вики: "The base height of a roof is height minus
  roof:height").
- Для не прямоугольных контуров крыша строится по ориентированному
  минимальному прямоугольнику контура (OBBox) — тот же приём, что и у
  реальных рендереров ("if the area is not rectangular... the OBBox is
  computed and used for the roof construction"). Когда используется форма
  ската (не `flat`), геометрия ВСЕГО здания (стены и крыша) строится по
  OBBox, а не по исходному контуру — иначе крыша не стыковалась бы со
  стенами по исходному контуру (щель на карнизе); для прямоугольных зданий
  разница с реальным контуром незаметна, для сильно непрямоугольных —
  осознанное упрощение силуэта той же природы, что и у реальных рендереров.
- `roof:direction` — направление ската ВНИЗ (от конька к карнизу), не
  направление самого конька (вики: "viewing direction from high (ridge) to
  low (eaves)"); конёк перпендикулярен этому направлению.
- `roof:orientation=along` (по умолчанию) — конёк вдоль длинной стороны
  OBBox; `=across` — вдоль короткой. Для вальмовой/шатровой крыши
  направление конька не варьируется (всегда вдоль длинной стороны) — иначе
  «съедаемое» вальмами измерение могло бы оказаться длиннее сохраняемого
  (геометрически некорректно): `roof:orientation` для них не учитывается,
  это осознанное упрощение.

Все объёмы конструкций проверены точными замкнутыми формулами (тесты) —
не «на глаз»:
- двускатная/односкатная: `V = 0.5 * поперечник * высота * конёк_или_длина`
  (одинаковая формула — сечение перпендикулярно скату одна и та же
  треугольная площадь, что для симметричной, что для односторонней крыши);
- вальмовая: `V = h * (L*W/2 - W²/6)` (L ≥ W, точная интегральная формула,
  при L=W вырождается в объём пирамиды L²h/3 — согласованность проверена);
- шатровая: `V = L * W * h / 3` (объём пирамиды над четырёхугольным
  основанием, для любых L, W, не только квадрата).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

from topology_geo.selection.service import SiteFeature

CONFIDENCE_FACT = "факт"
CONFIDENCE_DEFAULT = "умолчание"

ROOF_FLAT = "flat"
ROOF_GABLED = "gabled"
ROOF_HIPPED = "hipped"
ROOF_PYRAMIDAL = "pyramidal"
ROOF_SKILLION = "skillion"

SUPPORTED_ROOF_SHAPES = (ROOF_GABLED, ROOF_HIPPED, ROOF_PYRAMIDAL, ROOF_SKILLION)

# Типичный уклон скатной кровли жилых зданий (град.) - используется только
# когда нет ни roof:height, ни roof:angle (confidence="умолчание").
DEFAULT_ROOF_ANGLE_DEG = 30.0

Vertex = tuple[float, float, float]
Face = tuple[int, ...]
Vec2 = tuple[float, float]


@dataclass(frozen=True)
class OrientedBox:
    """Ориентированный минимальный прямоугольник контура (OBBox)."""

    center: Vec2
    length: float  # длинная сторона (L)
    width: float  # короткая сторона (W), width <= length
    long_axis: Vec2  # единичный вектор вдоль L
    short_axis: Vec2  # единичный вектор вдоль W, поворот long_axis на +90° (CCW)


def oriented_bounding_box(polygon: Polygon) -> OrientedBox:
    rect = polygon.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)[:4]
    p0, p1, p2 = np.array(coords[0]), np.array(coords[1]), np.array(coords[2])
    edge1, edge2 = p1 - p0, p2 - p1
    len1, len2 = float(np.linalg.norm(edge1)), float(np.linalg.norm(edge2))
    if len1 >= len2:
        length, width = len1, len2
        long_axis = (edge1 / len1) if len1 > 0 else np.array([1.0, 0.0])
    else:
        length, width = len2, len1
        long_axis = (edge2 / len2) if len2 > 0 else np.array([1.0, 0.0])
    short_axis = np.array([-long_axis[1], long_axis[0]])
    center = np.mean(coords, axis=0)
    return OrientedBox(
        center=(float(center[0]), float(center[1])), length=length, width=width,
        long_axis=(float(long_axis[0]), float(long_axis[1])), short_axis=(float(short_axis[0]), float(short_axis[1])),
    )


@dataclass(frozen=True)
class RoofParams:
    shape: str  # "flat" или один из SUPPORTED_ROOF_SHAPES
    shape_confidence: str
    height_m: float  # 0.0 для flat
    height_confidence: str
    ridge_along_long_axis: bool  # для gabled: конёк вдоль длинной (True) или короткой (False) стороны
    direction: Vec2 | None  # направление ската вниз (только skillion)


def _bearing_to_vector(bearing_deg: float) -> Vec2:
    """Компас (0=север=+y, по часовой стрелке) -> единичный вектор (x, y)
    в проекционных координатах (+x восток, +y север)."""
    rad = math.radians(bearing_deg)
    return math.sin(rad), math.cos(rad)


def _parse_float(raw: object) -> float | None:
    if not raw:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def compute_roof_params(feature: SiteFeature, obb: OrientedBox) -> RoofParams:
    """Параметры крыши по тегам `roof:*` (Шаг 2.2, п. 1). `flat`/нераспознанная
    форма (dome, mansard, gambrel...) -> RoofParams с shape=flat, height=0 —
    вызывающий код просто использует прежнюю плоскую призму (Шаг 1.6)."""
    raw_shape = str(feature.raw_tags.get("roof:shape") or "").strip().lower()
    if raw_shape not in SUPPORTED_ROOF_SHAPES:
        confidence = CONFIDENCE_FACT if raw_shape in ("flat", "") else CONFIDENCE_DEFAULT
        return RoofParams(ROOF_FLAT, confidence, 0.0, CONFIDENCE_FACT, True, None)

    shape = raw_shape
    roof_height = _parse_float(feature.raw_tags.get("roof:height"))
    roof_angle = _parse_float(feature.raw_tags.get("roof:angle"))

    if roof_height is not None and roof_height > 0:
        height_m, height_confidence = roof_height, CONFIDENCE_FACT
    else:
        angle = roof_angle if (roof_angle is not None and 0 < roof_angle < 90) else DEFAULT_ROOF_ANGLE_DEG
        # односкатная поднимается на всю ширину, остальные - от карниза до конька (половина)
        run = obb.width if shape == ROOF_SKILLION else obb.width / 2.0
        height_m = run * math.tan(math.radians(angle))
        height_confidence = CONFIDENCE_FACT if roof_angle is not None else CONFIDENCE_DEFAULT

    direction_deg = _parse_float(feature.raw_tags.get("roof:direction"))

    if shape == ROOF_SKILLION:
        direction = _bearing_to_vector(direction_deg) if direction_deg is not None else obb.short_axis
        return RoofParams(shape, CONFIDENCE_FACT, height_m, height_confidence, True, direction)

    if shape in (ROOF_HIPPED, ROOF_PYRAMIDAL):
        # конёк/вершина всегда вдоль длинной стороны - см. докстринг модуля
        return RoofParams(shape, CONFIDENCE_FACT, height_m, height_confidence, True, None)

    # gabled: направление конька - из roof:direction (перпендикулярно скату),
    # иначе из roof:orientation (along/across), иначе по умолчанию "along"
    if direction_deg is not None:
        down_x, down_y = _bearing_to_vector(direction_deg)
        ridge_guess = (down_y, -down_x)  # поворот на 90°
        dot_long = abs(ridge_guess[0] * obb.long_axis[0] + ridge_guess[1] * obb.long_axis[1])
        dot_short = abs(ridge_guess[0] * obb.short_axis[0] + ridge_guess[1] * obb.short_axis[1])
        ridge_along_long = dot_long >= dot_short
    else:
        orientation = str(feature.raw_tags.get("roof:orientation") or "along").strip().lower()
        ridge_along_long = orientation != "across"

    return RoofParams(shape, CONFIDENCE_FACT, height_m, height_confidence, ridge_along_long, None)


def _obb_corners_ccw(obb: OrientedBox) -> list[Vec2]:
    """4 угла OBBox в порядке против часовой стрелки (при взгляде сверху) -
    long_axis x short_axis образуют правую тройку (short_axis - поворот
    long_axis на +90°), поэтому обход в порядке (-,-) (+,-) (+,+) (-,+) в
    локальных координатах (u вдоль long_axis, v вдоль short_axis) идёт CCW."""
    cx, cy = obb.center
    lx, ly = obb.long_axis
    sx, sy = obb.short_axis
    hl, hw = obb.length / 2.0, obb.width / 2.0
    return [
        (cx - hl * lx - hw * sx, cy - hl * ly - hw * sy),
        (cx + hl * lx - hw * sx, cy + hl * ly - hw * sy),
        (cx + hl * lx + hw * sx, cy + hl * ly + hw * sy),
        (cx - hl * lx + hw * sx, cy - hl * ly + hw * sy),
    ]


def _wall_mesh(corners: list[Vec2], base_z: float, top_z: float) -> tuple[list[Vertex], list[Face]]:
    """Стены OBBox от base_z до top_z + нижняя крышка (без верхней - её
    достраивает крыша)."""
    n = len(corners)
    bottom = [(x, y, base_z) for x, y in corners]
    top = [(x, y, top_z) for x, y in corners]
    vertices = bottom + top
    faces: list[Face] = [(0, 2, 1), (0, 3, 2)]  # нижняя крышка (прямоугольник), смотрит вниз
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j))
        faces.append((i, n + j, n + i))
    return vertices, faces


def build_pitched_building_mesh(
    obb: OrientedBox, params: RoofParams, base_z: float, eave_z: float
) -> tuple[list[Vertex], list[Face]]:
    """Стены (OBBox, base_z..eave_z) + скатная крыша (eave_z..ridge/шатёр) -
    единый замкнутый меш. `params.shape` должен быть одним из
    `SUPPORTED_ROOF_SHAPES` (не `flat` - для плоской крыши используется
    прежняя `assemble.extrude_polygon_mesh` по исходному контуру)."""
    corners = _obb_corners_ccw(obb)
    vertices, faces = _wall_mesh(corners, base_z, eave_z)
    n = len(vertices)  # = 8 (4 низ + 4 верх стен = карниз)
    b0, b1, b2, b3 = range(4, 8)  # верх стен = карниз, тот же порядок, что corners
    ridge_z = eave_z + params.height_m

    if params.shape == ROOF_PYRAMIDAL:
        apex = (obb.center[0], obb.center[1], ridge_z)
        vertices.append(apex)
        apex_i = n
        faces += [(b0, b1, apex_i), (b1, b2, apex_i), (b2, b3, apex_i), (b3, b0, apex_i)]
        return vertices, faces

    if params.shape == ROOF_SKILLION:
        # скат вниз по params.direction: угол с "верхней" стороны (против
        # direction) остаётся на ridge_z, противоположный опускается до eave_z.
        dx, dy = params.direction
        proj = [dx * (x - obb.center[0]) + dy * (y - obb.center[1]) for x, y in corners]
        max_proj = max(abs(p) for p in proj) or 1.0
        roof_top = [
            (corners[i][0], corners[i][1], eave_z + params.height_m * (max_proj - proj[i]) / (2 * max_proj))
            for i in range(4)
        ]
        vertices.extend(roof_top)
        r0, r1, r2, r3 = n, n + 1, n + 2, n + 3
        faces += [(b0, b1, r1), (b0, r1, r0), (b1, b2, r2), (b1, r2, r1)]
        faces += [(b2, b3, r3), (b2, r3, r2), (b3, b0, r0), (b3, r0, r3)]
        faces += [(r0, r1, r2), (r0, r2, r3)]  # плоскость ската (верх)
        return vertices, faces

    # gabled / hipped: конёк вдоль long_axis (по умолчанию, вальмовая всегда)
    # или short_axis (двускатная, roof:orientation=across) - циклический сдвиг
    # угла b0..b3 на 1, чтобы сохранить (ridge_dir x cross_dir = +z) для тех
    # же формул нормалей, что выведены для ridge_along_long=True (см. тест
    # на объём и на замкнутость меша - оба варианта конька проверены).
    ridge_along_long = params.ridge_along_long_axis
    if ridge_along_long:
        rb0, rb1, rb2, rb3 = b0, b1, b2, b3
        ridge_dir = obb.long_axis
        ridge_half_len, cross_half_len = obb.length / 2.0, obb.width / 2.0
    else:
        rb0, rb1, rb2, rb3 = b1, b2, b3, b0
        ridge_dir = obb.short_axis
        ridge_half_len, cross_half_len = obb.width / 2.0, obb.length / 2.0

    if params.shape == ROOF_HIPPED:
        ridge_half_len = max(ridge_half_len - cross_half_len, 0.0)

    cx, cy = obb.center
    rx, ry = ridge_dir
    ra = (cx - ridge_half_len * rx, cy - ridge_half_len * ry, ridge_z)
    rb = (cx + ridge_half_len * rx, cy + ridge_half_len * ry, ridge_z)
    vertices += [ra, rb]
    ra_i, rb_i = n, n + 1

    faces += [(rb0, rb1, rb_i), (rb0, rb_i, ra_i)]  # скат 1
    faces += [(rb2, rb3, ra_i), (rb2, ra_i, rb_i)]  # скат 2 (противоположный)
    faces.append((rb0, ra_i, rb3))  # торец у ra (щипец/вальма)
    faces.append((rb1, rb2, rb_i))  # торец у rb

    return vertices, faces
