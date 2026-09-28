"""Тесты Шага 3.2: расчётные зоны — правила по норме + «считать только там,
где официальной зоны нет»."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from topology_geo.constraints.calculated import (
    calculate_cemetery_szz_m,
    calculate_coastal_protective_strip_m,
    calculate_network_protection_zone_m,
    calculate_power_line_zone_half_width_m,
    calculate_water_protection_zone_m,
    zones_needing_calculation,
)
from topology_geo.constraints.store import ConstraintZone, STATUS_CALCULATED, STATUS_OFFICIAL, ensure_schema, load_zone

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
SQUARE = {"type": "Polygon", "coordinates": [[
    [56.20, 58.00], [56.21, 58.00], [56.21, 58.01], [56.20, 58.01], [56.20, 58.00],
]]}
BBOX = (56.19, 57.99, 56.22, 58.02)


@pytest.mark.parametrize("area_ha,expected_m", [(5.0, 50.0), (10.0, 50.0), (15.0, 100.0), (20.0, 100.0), (35.0, 300.0), (100.0, 300.0)])
def test_calculate_cemetery_szz_m(area_ha, expected_m):
    assert calculate_cemetery_szz_m(area_ha) == expected_m


@pytest.mark.parametrize("length_km,expected_m", [(3.0, 50.0), (10.0, 50.0), (25.0, 100.0), (50.0, 100.0), (80.0, 200.0)])
def test_calculate_water_protection_zone_m(length_km, expected_m):
    assert calculate_water_protection_zone_m(length_km) == expected_m


@pytest.mark.parametrize("slope_deg,expected_m", [(None, 30.0), (0.0, 30.0), (1.5, 40.0), (2.9, 40.0), (3.0, 50.0), (10.0, 50.0)])
def test_calculate_coastal_protective_strip_m(slope_deg, expected_m):
    assert calculate_coastal_protective_strip_m(slope_deg) == expected_m


def test_calculate_power_line_zone_half_width_m_matches_geometry_power_table():
    from topology_geo.geometry.power import _safety_zone_half_width_m

    assert calculate_power_line_zone_half_width_m(10.0) == _safety_zone_half_width_m(10.0)
    assert calculate_power_line_zone_half_width_m(110.0) == 20.0
    assert calculate_power_line_zone_half_width_m(None) is None


@pytest.mark.parametrize("network_type,diameter_mm,expected_m", [
    ("В", 200.0, 5.0), ("В", 500.0, 10.0), ("К", 250.0, 5.0),
    ("Т", 800.0, 3.0), ("Г", 150.0, 2.0), ("Г", 400.0, 7.0), ("Кл", 0.0, 1.0),
])
def test_calculate_network_protection_zone_m(network_type, diameter_mm, expected_m):
    assert calculate_network_protection_zone_m(network_type, diameter_mm) == expected_m


def test_calculate_network_protection_zone_m_unknown_type_raises():
    with pytest.raises(ValueError, match="неизвестный тип сети"):
        calculate_network_protection_zone_m("Ж", 100.0)


@pytest.fixture()
def db(pg_test_db):
    ensure_schema(pg_test_db)
    return pg_test_db


def test_zones_needing_calculation_true_when_no_official_zone(db):
    assert zones_needing_calculation(db, BBOX, "охранная зона ЛЭП") is True


def test_zones_needing_calculation_false_when_official_zone_exists(db):
    load_zone(db, ConstraintZone(
        zone_type="охранная зона ЛЭП", status=STATUS_OFFICIAL, source_name="НСПД (тест)",
        data_timestamp=TS, geom_geojson=SQUARE, registry_number="PZ-1",
    ))
    assert zones_needing_calculation(db, BBOX, "охранная зона ЛЭП") is False
    # другой вид зоны в том же месте - расчёт всё равно нужен
    assert zones_needing_calculation(db, BBOX, "водоохранная зона") is True


def test_zones_needing_calculation_ignores_existing_calculated_zone(db):
    """Уже посчитанная РАНЕЕ расчётная зона не должна блокировать пересчёт -
    только ОФИЦИАЛЬНАЯ зона отменяет расчёт (действие п. 2 плана)."""
    load_zone(db, ConstraintZone(
        zone_type="охранная зона ЛЭП", status=STATUS_CALCULATED, source_name="расчёт (тест)",
        data_timestamp=TS, geom_geojson=SQUARE,
    ))
    assert zones_needing_calculation(db, BBOX, "охранная зона ЛЭП") is True
