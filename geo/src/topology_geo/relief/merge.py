"""Слияние источников рельефа с приоритетом и плавным переходом (Шаг 1.2, п. 3).

Формула — `docs/math-model.md` §2.4: если на участке есть более точный
источник (топосъёмка/облако точек), он приоритетнее общего DEM (TessaDEM), а
граница между ними сшивается плавным переходом шириной 30-50 м, чтобы не
было видимой «ступеньки».

`overlay` и `base` должны быть на ОДНОЙ сетке (одинаковые shape/transform) —
выравнивание разных источников на общую сетку делает `service.align_to_grid`.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def transition_alpha(overlay_valid_mask: np.ndarray, pixel_size_m: float, transition_width_m: float) -> np.ndarray:
    """Вес overlay (0..1): 1 в глубине overlay, 0 за пределами полосы перехода,
    плавно (по знаковому расстоянию до границы) между ними — см.
    `docs/math-model.md` §2.4. Полоса перехода имеет полную ширину
    `transition_width_m`, симметрично по `w/2` внутрь и наружу от границы, так
    что ровно на границе alpha = 0.5 (иначе на стыке был бы скачок веса, а не
    сама «ступенька» высоты — но её мы как раз и убираем).
    """
    if transition_width_m <= 0:
        return overlay_valid_mask.astype("float64")

    dist_inside = distance_transform_edt(overlay_valid_mask) * pixel_size_m
    dist_outside = distance_transform_edt(~overlay_valid_mask) * pixel_size_m
    signed_dist = np.where(overlay_valid_mask, dist_inside, -dist_outside)

    half_width = transition_width_m / 2.0
    alpha = np.clip((signed_dist + half_width) / transition_width_m, 0.0, 1.0)
    return alpha


def merge_with_transition(
    base: np.ndarray,
    overlay: np.ndarray,
    overlay_valid_mask: np.ndarray,
    *,
    pixel_size_m: float,
    transition_width_m: float = 40.0,
) -> np.ndarray:
    """Слить `base` (менее точный, но полный) и `overlay` (точнее, но частичный)
    с плавным переходом в полосе `transition_width_m` вокруг границы overlay.
    """
    if base.shape != overlay.shape or base.shape != overlay_valid_mask.shape:
        raise ValueError("base, overlay и overlay_valid_mask должны быть на одной сетке (одинаковый shape)")

    alpha = transition_alpha(overlay_valid_mask, pixel_size_m, transition_width_m)
    return alpha * overlay + (1.0 - alpha) * base


def max_adjacent_step(array: np.ndarray) -> float:
    """Наибольший перепад высоты между соседними пикселями (прокси для критерия
    Шага 1.2: «без ступеньки больше 0.2 м» — используется в тестах слияния и
    может переиспользоваться на реальном пилоте для количественной проверки).
    """
    dx = np.abs(np.diff(array, axis=1))
    dy = np.abs(np.diff(array, axis=0))
    return float(max(dx.max(initial=0.0), dy.max(initial=0.0)))
