"""Огибающая допустимой застройки (Шаг 3.3, п. 2-4).

Алгоритм п. 2 буквально по плану: участок → минус отступы → минус
охранные/иные запретные зоны → ограничение по высоте (минимум из ПЗЗ,
приаэродромной территории, зон охраны ОКН). Геометрия — `shapely`, та же
библиотека, что весь остальной проект (`geometry/*.py`), локальные
координаты участка (тот же принцип, что и вся модель с Шага 1.4 —
пересчёт в МСК-59/локальные координаты делается ДО вызова, не здесь).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shapely.geometry.base import BaseGeometry

from topology_geo.constraints.pzz import PzzZone


@dataclass(frozen=True)
class HeightLimitedZone:
    """Запретная/ограничивающая зона (обычно `constraints.store.
    ConstraintZone.geom`, уже в локальных координатах) — вычитается из
    пятна застройки; если несёт предельную высоту (приаэродромная
    территория, зона охраны ОКН), учитывается и в ограничении по высоте."""

    geometry: BaseGeometry
    label: str
    max_height_m: float | None = None


@dataclass(frozen=True)
class BuildingEnvelope:
    footprint: BaseGeometry  # участок минус отступы минус запретные зоны
    max_height_m: float | None
    height_limit_source: str | None
    regulation_card: dict[str, Any] = field(default_factory=dict)


def compute_building_envelope(
    site_polygon: BaseGeometry, pzz_zone: PzzZone, obstacle_zones: list[HeightLimitedZone],
) -> BuildingEnvelope:
    """Действие п. 2. `obstacle_zones` — охранные/иные запретные зоны,
    пересекающие участок (обычно результат `constraints.store.find_zones`,
    переведённый в локальные координаты вызывающей стороной)."""
    footprint = site_polygon.buffer(-pzz_zone.setback_m)
    for zone in obstacle_zones:
        if not footprint.is_empty:
            footprint = footprint.difference(zone.geometry)

    height_candidates: list[tuple[float, str]] = []
    if pzz_zone.max_height_m is not None:
        height_candidates.append((pzz_zone.max_height_m, f"ПЗЗ {pzz_zone.zone_code}"))
    for zone in obstacle_zones:
        if zone.max_height_m is not None:
            height_candidates.append((zone.max_height_m, zone.label))

    if height_candidates:
        max_height_m, height_limit_source = min(height_candidates, key=lambda t: t[0])
    else:
        max_height_m, height_limit_source = None, None

    regulation_card = {
        "zone_code": pzz_zone.zone_code,
        "vri": pzz_zone.vri,
        "max_height_m": max_height_m,
        "height_limit_source": height_limit_source,
        "max_building_percent": pzz_zone.max_building_percent,
        "setback_m": pzz_zone.setback_m,
        "document_basis": pzz_zone.document_basis,
        "footprint_area_m2": footprint.area if not footprint.is_empty else 0.0,
    }

    return BuildingEnvelope(
        footprint=footprint, max_height_m=max_height_m,
        height_limit_source=height_limit_source, regulation_card=regulation_card,
    )


# --- проверки нормативных расстояний (п. 4) ------------------------------
# СП 4.13130.2013, табл. 1: противопожарное расстояние между зданиями по
# степени огнестойкости (I-II/III/IV-V), упрощённая репрезентативная
# таблица (реальный норматив учитывает и класс конструктивной пожарной
# опасности - здесь берётся минимальный набор степеней, без него).
FIRE_RESISTANCE_DEGREES = ("I-II", "III", "IV-V")

_FIRE_BREAK_M: dict[tuple[str, str], float] = {
    ("I-II", "I-II"): 6.0,
    ("I-II", "III"): 8.0,
    ("I-II", "IV-V"): 10.0,
    ("III", "III"): 8.0,
    ("III", "IV-V"): 10.0,
    ("IV-V", "IV-V"): 15.0,
}


def check_fire_break_m(degree_a: str, degree_b: str) -> float:
    """Минимальное противопожарное расстояние между зданиями (СП
    4.13130.2013, табл. 1, упрощённо) — порядок аргументов не важен."""
    key = (degree_a, degree_b) if (degree_a, degree_b) in _FIRE_BREAK_M else (degree_b, degree_a)
    if key not in _FIRE_BREAK_M:
        raise ValueError(
            f"неизвестная степень огнестойкости: {degree_a!r}/{degree_b!r} "
            f"(ожидается одна из {FIRE_RESISTANCE_DEGREES})"
        )
    return _FIRE_BREAK_M[key]


# Инсоляция соседних зданий (СанПиН 1.2.3685-21, требует расчёта теней по
# положению солнца) - ЧЕСТНО не реализована в этом проходе: нужна модель
# солнца/теней, которой в проекте ещё нет (план вводит её отдельно, Шаг
# 4.3 «Солнце, небо, сезоны»); реализовать инсоляционную проверку раньше
# своей же геометрической основы означало бы либо дублировать её здесь
# заранее, либо посчитать формально без реальной модели солнца - оставлено
# как честная зависимость от Шага 4.3.
