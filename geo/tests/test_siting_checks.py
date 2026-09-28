"""Тесты Шага 3.9, п. 1: проверки посадки (огибающая, зоны, сети)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from shapely.geometry import LineString, box

from topology_geo.siting.checks import (
    check_network_clearances,
    check_within_envelope,
    check_zone_overlaps,
    run_all_checks,
)


@dataclass
class _Zone:
    zone_type: str
    geometry: object
    registry_number: str | None = None


@dataclass
class _NetworkSegment:
    network_type: str
    layer: str
    geometry: object


def test_check_within_envelope_no_collision_when_fully_inside():
    building = box(10, 10, 20, 20)
    envelope = box(0, 0, 100, 100)
    assert check_within_envelope(building, envelope) is None


def test_check_within_envelope_collision_when_outside():
    building = box(90, 10, 110, 20)
    envelope = box(0, 0, 100, 100)

    collision = check_within_envelope(building, envelope)

    assert collision is not None
    assert collision.check_type == "огибающая застройки"
    assert "10.0" in collision.description or "м²" in collision.description


def test_check_zone_overlaps_finds_intersecting_zone():
    building = box(0, 0, 10, 10)
    zone = _Zone(zone_type="красная линия", geometry=box(5, 5, 15, 15), registry_number="RL-1")

    collisions = check_zone_overlaps(building, [zone])

    assert len(collisions) == 1
    assert collisions[0].zone_type == "красная линия"
    assert "RL-1" in collisions[0].object_refs


def test_check_zone_overlaps_no_collision_when_far():
    building = box(0, 0, 10, 10)
    zone = _Zone(zone_type="ООПТ", geometry=box(1000, 1000, 1010, 1010))

    assert check_zone_overlaps(building, [zone]) == []


def test_check_network_clearances_detects_close_network():
    building = box(0, 0, 10, 10)
    close_network = _NetworkSegment(network_type="К", layer="К1", geometry=LineString([(11, 0), (11, 10)]))

    collisions = check_network_clearances(building, [close_network], min_clearance_m=2.0)

    assert len(collisions) == 1
    assert "канализация" in collisions[0].description


def test_check_network_clearances_no_collision_when_far_enough():
    building = box(0, 0, 10, 10)
    far_network = _NetworkSegment(network_type="В", layer="В1", geometry=LineString([(20, 0), (20, 10)]))

    assert check_network_clearances(building, [far_network], min_clearance_m=2.0) == []


def test_run_all_checks_combines_all_check_types():
    building = box(0, 0, 10, 10)
    envelope = box(-5, -5, 5, 15)  # здание частично выходит за огибающую
    zone = _Zone(zone_type="ЗОУИТ", geometry=box(8, 8, 20, 20))
    network = _NetworkSegment(network_type="Г", layer="Г1", geometry=LineString([(10.5, 0), (10.5, 10)]))

    collisions = run_all_checks(building, envelope, [zone], [network], network_min_clearance_m=2.0)

    check_types = {c.check_type for c in collisions}
    assert "огибающая застройки" in check_types
    assert "зона ограничения" in check_types
    assert "сети (горизонтальное расстояние)" in check_types
