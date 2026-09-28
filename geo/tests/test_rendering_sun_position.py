"""Тесты Шага 4.3, п. 1: положение солнца — проверено известными
астрономическими фактами (высота на полдень солнцестояния = 90° - |широта
- склонение|), не подогнанными числами."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from topology_geo.rendering.sun_position import compute_sun_position, sun_direction_vector

PERM_LAT, PERM_LON = 58.0, 56.24


def test_june_solstice_noon_elevation_matches_known_formula():
    """21 июня, полдень по Гринвичу (lon=0, солнечный полдень ~ 12:00 UTC,
    с точностью до уравнения времени, ≤ ~15 мин) - склонение +23.44°,
    известная формула высоты на полдень: 90° - |широта - склонение|."""
    dt = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
    position = compute_sun_position(dt, lat_deg=PERM_LAT, lon_deg=0.0)

    expected_elevation = 90.0 - abs(PERM_LAT - 23.44)
    assert position.elevation_deg == pytest.approx(expected_elevation, abs=1.0)
    assert position.declination_deg == pytest.approx(23.44, abs=0.5)


def test_december_solstice_noon_elevation_matches_known_formula():
    dt = datetime(2026, 12, 21, 12, 0, tzinfo=timezone.utc)
    position = compute_sun_position(dt, lat_deg=PERM_LAT, lon_deg=0.0)

    expected_elevation = 90.0 - abs(PERM_LAT + 23.44)
    assert position.elevation_deg == pytest.approx(expected_elevation, abs=1.0)
    assert position.declination_deg == pytest.approx(-23.44, abs=0.5)


def test_sun_is_higher_at_summer_noon_than_winter_noon_at_perm():
    summer = compute_sun_position(datetime(2026, 6, 21, 8, 0, tzinfo=timezone.utc), PERM_LAT, PERM_LON)
    winter = compute_sun_position(datetime(2026, 12, 21, 8, 0, tzinfo=timezone.utc), PERM_LAT, PERM_LON)

    assert summer.elevation_deg > winter.elevation_deg


def test_solar_noon_azimuth_is_south_in_northern_hemisphere():
    """В северном полушарии (широта > склонения) солнце в истинный
    полдень - строго на юге (азимут 180°, от севера по часовой стрелке)."""
    dt = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)  # ~равноденствие, lon=0 -> полдень ~ 12:00 UTC
    position = compute_sun_position(dt, lat_deg=PERM_LAT, lon_deg=0.0)

    assert position.azimuth_deg == pytest.approx(180.0, abs=3.0)


def test_sun_below_horizon_at_local_midnight_in_winter():
    dt = datetime(2026, 12, 21, 0, 0, tzinfo=timezone.utc)  # полночь по Гринвичу ~ полночь на lon=0
    position = compute_sun_position(dt, lat_deg=PERM_LAT, lon_deg=0.0)

    assert position.elevation_deg < 0


def test_elevation_is_periodic_over_24_hours():
    """Высота солнца в один и тот же момент разными сутками спустя (365
    солнечных суток ~ 366 звёздных, но для целых СУТОК разница пренебрежимо
    мала на уровне точности алгоритма) должна быть близка - грубая, но
    честная проверка периодичности, без внешнего эталона."""
    dt1 = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)
    dt2 = datetime(2026, 5, 2, 10, 0, tzinfo=timezone.utc)
    p1 = compute_sun_position(dt1, PERM_LAT, PERM_LON)
    p2 = compute_sun_position(dt2, PERM_LAT, PERM_LON)

    assert abs(p1.elevation_deg - p2.elevation_deg) < 1.0


def test_sun_direction_vector_is_unit_length():
    position = compute_sun_position(datetime(2026, 6, 21, 8, 0, tzinfo=timezone.utc), PERM_LAT, PERM_LON)
    vx, vy, vz = sun_direction_vector(position)

    norm = math.sqrt(vx**2 + vy**2 + vz**2)
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_sun_direction_vector_z_sign_matches_elevation_sign():
    above_horizon = compute_sun_position(datetime(2026, 6, 21, 8, 0, tzinfo=timezone.utc), PERM_LAT, PERM_LON)
    below_horizon = compute_sun_position(datetime(2026, 12, 21, 0, 0, tzinfo=timezone.utc), PERM_LAT, 0.0)

    assert above_horizon.elevation_deg > 0
    assert sun_direction_vector(above_horizon)[2] > 0
    assert below_horizon.elevation_deg < 0
    assert sun_direction_vector(below_horizon)[2] < 0


def test_compute_sun_position_accepts_naive_utc_datetime():
    aware = compute_sun_position(datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc), PERM_LAT, 0.0)
    naive = compute_sun_position(datetime(2026, 6, 21, 12, 0), PERM_LAT, 0.0)

    assert aware.elevation_deg == pytest.approx(naive.elevation_deg, abs=1e-9)
