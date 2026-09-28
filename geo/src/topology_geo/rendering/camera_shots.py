"""Автоматические ракурсы (Шаг 4.4) — чистая геометрия камер, без
рендера: положение и направление камеры считаются алгебраически (для
«птичьего полёта» — по угловому размеру кадра, для уровня пешехода — по
трассировке луча в плане), сам кадр не строится (это отдельный шаг,
Шаг 4.5 — рендер-воркер).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

EYE_HEIGHT_M = 1.6  # уровень пешехода (п. 2, буквально по плану)
BIRD_EYE_ANGLE_DEG = 45.0  # угол камеры "птичьего полёта" (п. 1, буквально по плану)
BIRD_EYE_TARGET_FRAME_FRACTION = 0.4  # ЖК занимает ~40% кадра (п. 1)
DEFAULT_FOV_DEG = 50.0  # типовой полный угол обзора камеры


@dataclass(frozen=True)
class Camera:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    fov_deg: float = DEFAULT_FOV_DEG
    label: str = ""


def _footprint_diagonal_m(footprint: BaseGeometry) -> float:
    minx, miny, maxx, maxy = footprint.bounds
    return math.hypot(maxx - minx, maxy - miny)


def bird_eye_cameras(
    footprint: BaseGeometry, building_height_m: float, base_z: float = 0.0, *, fov_deg: float = DEFAULT_FOV_DEG,
) -> list[Camera]:
    """Действие п. 1: «4 камеры под 45°, ЖК занимает ~40% кадра». Дистанция
    — из углового размера кадра: чтобы объект диагонали `size` занял долю
    `BIRD_EYE_TARGET_FRAME_FRACTION` угла обзора `fov_deg`, требуемый
    угловой размер `θ = fov_deg · fraction`, расстояние `d = (size/2) /
    tan(θ/2)` — стандартная формула камеры, не подобрана эмпирически."""
    centroid = footprint.centroid
    size = max(_footprint_diagonal_m(footprint), building_height_m)
    target_angle_rad = math.radians(fov_deg * BIRD_EYE_TARGET_FRAME_FRACTION)
    distance = (size / 2.0) / math.tan(target_angle_rad / 2.0)

    target = (centroid.x, centroid.y, base_z + building_height_m / 2.0)
    elevation_rad = math.radians(BIRD_EYE_ANGLE_DEG)
    horizontal_distance = distance * math.cos(elevation_rad)
    vertical_distance = distance * math.sin(elevation_rad)

    cameras = []
    for i, compass in enumerate(("С", "В", "Ю", "З")):
        bearing_rad = math.radians(i * 90.0)
        cam_x = centroid.x + horizontal_distance * math.sin(bearing_rad)
        cam_y = centroid.y + horizontal_distance * math.cos(bearing_rad)
        cam_z = base_z + vertical_distance
        cameras.append(Camera(
            position=(cam_x, cam_y, cam_z), target=target, fov_deg=fov_deg, label=f"птичий полёт — {compass}",
        ))
    return cameras


def pedestrian_view_points(
    footprint: BaseGeometry, sidewalks: list[LineString], obstacles: list[BaseGeometry],
    *, eye_height_m: float = EYE_HEIGHT_M, sample_step_m: float = 10.0, max_distance_m: float = 150.0,
) -> list[Camera]:
    """Действие п. 2: «точки на ближайших тротуарах, откуда ЖК виден без
    перекрытия (проверка лучом), высота 1,6 м». Видимость — прямая линия
    от точки до центроида ЖК не пересекает ни одно из `obstacles» (другие
    здания/препятствия) — честная проверка в плане (2D), не полная 3D-
    трассировка (которая потребовала бы модели высот всех препятствий и
    самого рендера, недоступных на этом шаге)."""
    target_point = footprint.centroid
    cameras = []
    for sidewalk in sidewalks:
        length = sidewalk.length
        if length <= 0:
            continue
        n_samples = max(1, int(length / sample_step_m))
        for i in range(n_samples + 1):
            point = sidewalk.interpolate(i / n_samples, normalized=True)
            if point.distance(target_point) > max_distance_m:
                continue
            sight_line = LineString([point, target_point])
            blocked = any(sight_line.crosses(obstacle) for obstacle in obstacles)
            if blocked:
                continue
            cameras.append(Camera(
                position=(point.x, point.y, eye_height_m),
                target=(target_point.x, target_point.y, eye_height_m),
                label="уровень пешехода",
            ))
    return cameras


def flythrough_path(
    footprint: BaseGeometry, building_height_m: float, base_z: float = 0.0,
    *, duration_s: float = 25.0, fps: int = 30, radius_multiplier: float = 1.5,
) -> list[Camera]:
    """Действие п. 3: «круговая траектория 20-30 с» — набор ключевых кадров
    камеры по окружности вокруг ЖК на высоте птичьего полёта (та же
    дистанция/высота, что `bird_eye_cameras`, но НЕПРЕРЫВНО по углу, не 4
    фиксированные точки)."""
    if not (20.0 <= duration_s <= 30.0):
        raise ValueError("облёт должен занимать 20-30 с по плану")

    centroid = footprint.centroid
    size = max(_footprint_diagonal_m(footprint), building_height_m)
    radius = size * radius_multiplier
    elevation_rad = math.radians(BIRD_EYE_ANGLE_DEG)
    horizontal_distance = radius * math.cos(elevation_rad)
    cam_z = base_z + radius * math.sin(elevation_rad)
    target = (centroid.x, centroid.y, base_z + building_height_m / 2.0)

    frame_count = int(duration_s * fps)
    cameras = []
    for frame in range(frame_count):
        angle_rad = 2 * math.pi * frame / frame_count
        cam_x = centroid.x + horizontal_distance * math.sin(angle_rad)
        cam_y = centroid.y + horizontal_distance * math.cos(angle_rad)
        cameras.append(Camera(position=(cam_x, cam_y, cam_z), target=target, label=f"облёт — кадр {frame}"))
    return cameras
