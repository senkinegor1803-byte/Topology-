"""Тесты Шага 3.3, п. 2-4: огибающая допустимой застройки + проверка
противопожарных расстояний."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from shapely.geometry import box

from topology_geo.constraints.envelope import (
    FIRE_RESISTANCE_DEGREES,
    HeightLimitedZone,
    check_fire_break_m,
    compute_building_envelope,
)
from topology_geo.constraints.pzz import PzzZone

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _pzz(setback_m=5.0, max_height_m=40.0, max_building_percent=40.0):
    return PzzZone(
        zone_code="Ж-1", vri=["жилая застройка"], setback_m=setback_m,
        max_height_m=max_height_m, max_building_percent=max_building_percent,
        document_basis="ПЗЗ, решение №1", source_name="тест", data_timestamp=TS,
        geom_geojson={"type": "Polygon", "coordinates": []},
    )


def test_envelope_shrinks_by_setback_when_no_obstacles():
    site = box(0, 0, 100, 100)
    envelope = compute_building_envelope(site, _pzz(setback_m=5.0), obstacle_zones=[])

    expected = box(5, 5, 95, 95)
    assert envelope.footprint.equals(expected)
    assert envelope.max_height_m == 40.0
    assert envelope.height_limit_source == "ПЗЗ Ж-1"
    assert envelope.regulation_card["footprint_area_m2"] == pytest.approx(90.0 * 90.0)


def test_envelope_subtracts_obstacle_zone():
    site = box(0, 0, 100, 100)
    obstacle = HeightLimitedZone(geometry=box(40, 40, 60, 60), label="охранная зона ЛЭП")

    envelope = compute_building_envelope(site, _pzz(setback_m=0.0), obstacle_zones=[obstacle])

    assert envelope.footprint.area == pytest.approx(100 * 100 - 20 * 20)
    assert not envelope.footprint.intersects(box(41, 41, 59, 59))


def test_envelope_height_limit_is_minimum_of_all_sources():
    site = box(0, 0, 100, 100)
    okn_zone = HeightLimitedZone(geometry=box(200, 200, 210, 210), label="зона охраны ОКН №5", max_height_m=15.0)
    priaerodrome = HeightLimitedZone(geometry=box(300, 300, 310, 310), label="приаэродромная территория", max_height_m=50.0)

    envelope = compute_building_envelope(site, _pzz(max_height_m=40.0), obstacle_zones=[okn_zone, priaerodrome])

    assert envelope.max_height_m == 15.0  # минимум из 40 (ПЗЗ) / 15 (ОКН) / 50 (приаэродромная)
    assert envelope.height_limit_source == "зона охраны ОКН №5"


def test_envelope_regulation_card_carries_vri_and_document():
    site = box(0, 0, 100, 100)
    envelope = compute_building_envelope(site, _pzz(), obstacle_zones=[])

    card = envelope.regulation_card
    assert card["vri"] == ["жилая застройка"]
    assert card["document_basis"] == "ПЗЗ, решение №1"
    assert card["max_building_percent"] == 40.0
    assert card["setback_m"] == 5.0


def test_envelope_empty_when_setback_consumes_whole_site():
    site = box(0, 0, 10, 10)
    envelope = compute_building_envelope(site, _pzz(setback_m=20.0), obstacle_zones=[])

    assert envelope.footprint.is_empty
    assert envelope.regulation_card["footprint_area_m2"] == 0.0


@pytest.mark.parametrize("degree_a,degree_b,expected_m", [
    ("I-II", "I-II", 6.0), ("I-II", "III", 8.0), ("III", "I-II", 8.0),
    ("I-II", "IV-V", 10.0), ("IV-V", "IV-V", 15.0),
])
def test_check_fire_break_m(degree_a, degree_b, expected_m):
    assert check_fire_break_m(degree_a, degree_b) == expected_m


def test_check_fire_break_m_unknown_degree_raises():
    with pytest.raises(ValueError, match="огнестойкости"):
        check_fire_break_m("I-II", "VI")


def test_fire_resistance_degrees_all_pairs_covered():
    for a in FIRE_RESISTANCE_DEGREES:
        for b in FIRE_RESISTANCE_DEGREES:
            assert check_fire_break_m(a, b) > 0
