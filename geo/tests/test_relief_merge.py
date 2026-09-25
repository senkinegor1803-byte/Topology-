"""Тесты слияния источников рельефа с плавным переходом (Шаг 1.2, п. 3;
формула — docs/math-model.md §2.4).

`test_transition_is_continuous_across_boundary` — регрессия на реальный баг,
найденный при ручной проверке: изначальная реализация считала alpha
независимо для "внутри" и "снаружи" маски и давала разрыв (скачок 0.9 -> 0.1)
прямо на границе вместо плавного перехода через 0.5.
"""

from __future__ import annotations

import numpy as np
import pytest

from topology_geo.relief.merge import max_adjacent_step, merge_with_transition, transition_alpha


def _square_mask(n: int, lo: int, hi: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    mask[lo:hi, lo:hi] = True
    return mask


def test_deep_inside_overlay_gets_full_weight():
    mask = _square_mask(80, 25, 55)
    alpha = transition_alpha(mask, pixel_size_m=1.0, transition_width_m=10.0)
    assert alpha[40, 40] == pytest.approx(1.0)


def test_far_outside_overlay_gets_zero_weight():
    mask = _square_mask(80, 25, 55)
    alpha = transition_alpha(mask, pixel_size_m=1.0, transition_width_m=10.0)
    assert alpha[2, 2] == pytest.approx(0.0)


def test_boundary_weight_is_close_to_half():
    mask = _square_mask(80, 25, 55)
    alpha = transition_alpha(mask, pixel_size_m=1.0, transition_width_m=10.0)
    # ближайшие пиксели по обе стороны границы (строка 40, col 24 снаружи / col 25 внутри)
    assert 0.3 <= alpha[40, 24] <= 0.5
    assert 0.5 <= alpha[40, 25] <= 0.7


def test_transition_is_continuous_across_boundary():
    """Регрессия: alpha не должна "прыгать" при переходе через границу маски —
    только монотонно расти изнутри наружу внутрь (или наоборот, в зависимости
    от направления обхода), без скачков больше одного шага дискретизации."""
    mask = _square_mask(80, 25, 55)
    alpha = transition_alpha(mask, pixel_size_m=1.0, transition_width_m=10.0)

    row = 40
    profile = alpha[row, 15:35]
    diffs = np.diff(profile)
    assert np.all(diffs >= -1e-9), f"alpha должна не убывать при движении к центру overlay: {profile}"
    assert diffs.max() <= 0.25, f"скачок между соседними пикселями слишком велик: {profile}"


def test_smooth_merge_reduces_step_vs_hard_merge():
    n = 80
    mask = _square_mask(n, 25, 55)
    base = np.full((n, n), 100.0)
    overlay = np.full((n, n), 105.0)

    smooth = merge_with_transition(base, overlay, mask, pixel_size_m=1.0, transition_width_m=10.0)
    hard = np.where(mask, overlay, base)

    assert max_adjacent_step(smooth) < max_adjacent_step(hard)
    # критерий Шага 1.2: без "ступеньки" -- для разницы источников 5 м и полосы
    # 10 м шаг между соседними пикселями (1 м) должен быть << 0.2 м * (5/0.2)
    # т.е. плавным относительно перепада между источниками
    assert max_adjacent_step(smooth) <= 1.0


def test_merge_matches_sources_far_from_boundary():
    n = 80
    mask = _square_mask(n, 25, 55)
    base = np.full((n, n), 100.0)
    overlay = np.full((n, n), 105.0)

    merged = merge_with_transition(base, overlay, mask, pixel_size_m=1.0, transition_width_m=10.0)
    assert merged[40, 40] == pytest.approx(105.0)
    assert merged[2, 2] == pytest.approx(100.0)


def test_zero_transition_width_is_hard_mask():
    n = 40
    mask = _square_mask(n, 10, 30)
    alpha = transition_alpha(mask, pixel_size_m=1.0, transition_width_m=0.0)
    np.testing.assert_array_equal(alpha, mask.astype("float64"))


def test_merge_rejects_mismatched_shapes():
    base = np.zeros((10, 10))
    overlay = np.zeros((10, 11))
    mask = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError):
        merge_with_transition(base, overlay, mask, pixel_size_m=1.0, transition_width_m=10.0)


def test_max_adjacent_step_on_flat_array_is_zero():
    flat = np.full((10, 10), 42.0)
    assert max_adjacent_step(flat) == 0.0
