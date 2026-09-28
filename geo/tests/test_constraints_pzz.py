"""Тесты Шага 3.3, п. 1: территориальные зоны ПЗЗ (реальный PostGIS)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from topology_geo.constraints.pzz import PzzZone, ensure_schema, find_pzz_zone_for_point, load_pzz_zone

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
SQUARE = {"type": "Polygon", "coordinates": [[
    [56.20, 58.00], [56.21, 58.00], [56.21, 58.01], [56.20, 58.01], [56.20, 58.00],
]]}


@pytest.fixture()
def db(pg_test_db):
    ensure_schema(pg_test_db)
    return pg_test_db


def test_load_and_find_pzz_zone_for_point_inside(db):
    load_pzz_zone(db, PzzZone(
        zone_code="Ж-1", vri=["жилая застройка", "объекты социальной инфраструктуры"],
        setback_m=3.0, max_height_m=40.0, max_building_percent=40.0,
        document_basis="ПЗЗ г. Перми, решение №1", source_name="ИСОГД (тест)",
        data_timestamp=TS, geom_geojson=SQUARE,
    ))

    zone = find_pzz_zone_for_point(db, lon=56.205, lat=58.005)
    assert zone is not None
    assert zone.zone_code == "Ж-1"
    assert "жилая застройка" in zone.vri
    assert zone.max_height_m == 40.0
    assert zone.setback_m == 3.0


def test_find_pzz_zone_for_point_outside_returns_none(db):
    load_pzz_zone(db, PzzZone(
        zone_code="Ж-1", vri=["жилая застройка"], setback_m=3.0,
        source_name="ИСОГД (тест)", data_timestamp=TS, geom_geojson=SQUARE,
    ))

    assert find_pzz_zone_for_point(db, lon=10.0, lat=10.0) is None
