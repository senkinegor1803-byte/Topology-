"""Проверки посадки (Шаг 3.9, п. 1): огибающая застройки, красные линии,
ЗОУИТ, сети, ЛЭП. Упрощённая инсоляция ЧЕСТНО не реализована — требует
модели солнца/теней (план вводит её отдельным Шагом 4.3 «Солнце, небо,
сезоны»), которой в проекте ещё нет (тот же честный отказ, что уже
зафиксирован для этой же проверки в Шаге 3.3, `constraints/envelope.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shapely.geometry.base import BaseGeometry

CHECK_ENVELOPE = "огибающая застройки"
CHECK_ZONE = "зона ограничения"  # конкретный вид - в Collision.zone_type
NETWORK_TYPE_LABELS_RU = {"К": "канализация", "В": "водопровод", "Т": "теплосеть", "Г": "газопровод", "Кл": "кабельная линия"}


@dataclass(frozen=True)
class Collision:
    check_type: str
    description: str
    object_refs: list[str]
    zone_type: str | None = None


def check_within_envelope(building_footprint: BaseGeometry, envelope_footprint: BaseGeometry) -> Collision | None:
    """п. 1: «огибающая застройки» — здание должно ЦЕЛИКОМ помещаться в
    допустимую огибающую (Шаг 3.3)."""
    if envelope_footprint.contains(building_footprint):
        return None
    outside_area = building_footprint.difference(envelope_footprint).area
    return Collision(
        check_type=CHECK_ENVELOPE,
        description=f"здание выходит за огибающую допустимой застройки на {outside_area:.1f} м²",
        object_refs=["building", "envelope"],
    )


def check_zone_overlaps(building_footprint: BaseGeometry, zones: list[Any]) -> list[Collision]:
    """п. 1: «красные линии, ЗОУИТ, ЛЭП» — все три буквально пересечение с
    зонами `constraints.store.ConstraintZone` (уже в локальных координатах
    участка — перевод из WGS-84 делает вызывающая сторона), различаются
    только `zone_type`, не отдельной логикой: универсальная проверка на
    универсальной таблице зон (Шаг 3.1), не три копии одной функции."""
    collisions = []
    for zone in zones:
        geom = zone.geometry if hasattr(zone, "geometry") else zone.geom
        if building_footprint.intersects(geom):
            overlap_area = building_footprint.intersection(geom).area
            registry_number = getattr(zone, "registry_number", None)
            collisions.append(Collision(
                check_type=CHECK_ZONE,
                description=f"пересечение с зоной «{zone.zone_type}» ({overlap_area:.1f} м²)",
                object_refs=["building", registry_number or zone.zone_type],
                zone_type=zone.zone_type,
            ))
    return collisions


def check_network_clearances(
    building_footprint: BaseGeometry, network_segments: list[Any], *, min_clearance_m: float,
) -> list[Collision]:
    """п. 1: «сети (горизонтальные... расстояния)» — расстояние от здания
    до подземной сети (Шаг 3.4) меньше норматива. ВЕРТИКАЛЬНОЕ расстояние
    ЧЕСТНО не проверяется — у здания в этой модели нет глубины заложения
    фундамента (её не из чего взять: посадка ЖК не строит фундамент,
    только пятно в плане), проверять вертикальный зазор без неё означало
    бы либо выдумать глубину, либо формально сравнить с нормативной
    глубиной сети саму по себе, что не то же самое."""
    collisions = []
    for segment in network_segments:
        distance = building_footprint.distance(segment.geometry)
        if distance < min_clearance_m:
            label = NETWORK_TYPE_LABELS_RU.get(segment.network_type, segment.network_type or "неизвестная сеть")
            collisions.append(Collision(
                check_type="сети (горизонтальное расстояние)",
                description=f"{label}: {distance:.2f} м < норматив {min_clearance_m} м",
                object_refs=["building", segment.layer],
            ))
    return collisions


def run_all_checks(
    building_footprint: BaseGeometry, envelope_footprint: BaseGeometry, zones: list[Any],
    network_segments: list[Any], *, network_min_clearance_m: float = 2.0,
) -> list[Collision]:
    collisions: list[Collision] = []
    envelope_collision = check_within_envelope(building_footprint, envelope_footprint)
    if envelope_collision is not None:
        collisions.append(envelope_collision)
    collisions.extend(check_zone_overlaps(building_footprint, zones))
    collisions.extend(check_network_clearances(building_footprint, network_segments, min_clearance_m=network_min_clearance_m))
    return collisions
