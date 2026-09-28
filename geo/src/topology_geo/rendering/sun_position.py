"""Положение солнца по координатам, дате и времени (Шаг 4.3, п. 1).

Общепринятый упрощённый алгоритм (Michalsky 1988 / NOAA Solar Position
Calculations — стандартная астрономическая формула, не придумана для
проекта), точность порядка 0,01° — достаточно для расчёта теней на кадре
(критерий шага), избыточная точность (учёт нутации, атмосферной рефракции)
не нужна и не реализована.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class SunPosition:
    elevation_deg: float  # высота над горизонтом, град (отрицательная - солнце под горизонтом)
    azimuth_deg: float  # азимут от севера по часовой стрелке, град [0, 360)
    declination_deg: float  # склонение солнца на эту дату, град


def _julian_day(dt_utc: datetime) -> float:
    """Юлианский день (полдень UTC = .5) — стандартная формула."""
    a = (14 - dt_utc.month) // 12
    y = dt_utc.year + 4800 - a
    m = dt_utc.month + 12 * a - 3
    jdn = dt_utc.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    day_fraction = (dt_utc.hour - 12) / 24 + dt_utc.minute / 1440 + dt_utc.second / 86400
    return jdn + day_fraction


def compute_sun_position(dt_utc: datetime, lat_deg: float, lon_deg: float) -> SunPosition:
    """Положение солнца в момент `dt_utc` (должен быть timezone-aware UTC
    или наивным UTC) для точки (`lat_deg`, `lon_deg`, WGS-84)."""
    if dt_utc.tzinfo is not None:
        dt_utc = dt_utc.astimezone(timezone.utc).replace(tzinfo=None)

    jd = _julian_day(dt_utc)
    n = jd - 2451545.0  # дней от эпохи J2000.0

    mean_longitude = math.radians((280.460 + 0.9856474 * n) % 360)
    mean_anomaly = math.radians((357.528 + 0.9856003 * n) % 360)
    ecliptic_longitude = mean_longitude + math.radians(1.915) * math.sin(mean_anomaly) \
        + math.radians(0.020) * math.sin(2 * mean_anomaly)
    obliquity = math.radians(23.439 - 0.0000004 * n)

    right_ascension = math.atan2(math.cos(obliquity) * math.sin(ecliptic_longitude), math.cos(ecliptic_longitude))
    declination = math.asin(math.sin(obliquity) * math.sin(ecliptic_longitude))

    # Гринвичское звёздное время (упрощённо, точность достаточна для теней)
    gmst_hours = (6.697374558 + 0.06570982441908 * n + 1.00273790935 * (dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600)) % 24
    local_sidereal_hours = (gmst_hours + lon_deg / 15) % 24
    hour_angle = math.radians(local_sidereal_hours * 15 - math.degrees(right_ascension))

    lat_rad = math.radians(lat_deg)
    elevation = math.asin(
        math.sin(lat_rad) * math.sin(declination) + math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle)
    )
    azimuth = math.atan2(
        math.sin(hour_angle),
        math.cos(hour_angle) * math.sin(lat_rad) - math.tan(declination) * math.cos(lat_rad),
    )
    azimuth_deg = (math.degrees(azimuth) + 180) % 360  # atan2 даёт от юга - сдвиг к азимуту от севера

    return SunPosition(
        elevation_deg=math.degrees(elevation), azimuth_deg=azimuth_deg, declination_deg=math.degrees(declination),
    )


def sun_direction_vector(position: SunPosition) -> tuple[float, float, float]:
    """Единичный вектор НА солнце в локальной системе (X-восток, Y-север,
    Z-вверх) — для передачи направленного света в сцену рендера (Шаг 4.1/
    4.5: `bpy` ожидает направление, не углы)."""
    elevation = math.radians(position.elevation_deg)
    azimuth = math.radians(position.azimuth_deg)
    return (
        math.cos(elevation) * math.sin(azimuth),
        math.cos(elevation) * math.cos(azimuth),
        math.sin(elevation),
    )
