"""Тесты Шага 3.1 (загрузчик официальных ограничений) + Шага 3.2 (расчётные
зоны, та же таблица/загрузчик, `status="расчётно"`): версионирование,
поиск по bbox, история версий — на реальном PostGIS (`pg_test_db`)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from topology_geo.constraints.store import (
    STATUS_CALCULATED,
    STATUS_OFFICIAL,
    ConstraintZone,
    ensure_schema,
    find_zones,
    load_zone,
    load_zones_from_geojson,
    zone_history,
)

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)

SQUARE_NEAR_PERM = {"type": "Polygon", "coordinates": [[
    [56.20, 58.00], [56.21, 58.00], [56.21, 58.01], [56.20, 58.01], [56.20, 58.00],
]]}
SQUARE_FAR_AWAY = {"type": "Polygon", "coordinates": [[
    [10.0, 10.0], [10.1, 10.0], [10.1, 10.1], [10.0, 10.1], [10.0, 10.0],
]]}


@pytest.fixture()
def db(pg_test_db):
    ensure_schema(pg_test_db)
    return pg_test_db


def test_ensure_schema_is_idempotent(db):
    ensure_schema(db)  # повторный вызов не должен падать


def test_load_zone_official_requires_registry_number_via_geojson_loader(db):
    fc = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "geometry": SQUARE_NEAR_PERM,
        "properties": {"zone_type": "ЗОУИТ водоканала"},
    }]}
    with pytest.raises(ValueError, match="реестрового номера"):
        load_zones_from_geojson(db, fc, source_name="НСПД (тест)", data_timestamp=TS)


def test_load_zones_from_geojson_real_round_trip(db):
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature", "geometry": SQUARE_NEAR_PERM,
                "properties": {
                    "zone_type": "Приаэродромная территория", "registry_number": "59:00-1.1",
                    "document_basis": "Приказ Росавиации №123", "regime": "ограничение по высоте 50 м",
                },
            },
            {
                "type": "Feature", "geometry": SQUARE_FAR_AWAY,
                "properties": {"zone_type": "ООПТ", "registry_number": "59:00-2.1"},
            },
        ],
    }

    ids = load_zones_from_geojson(db, fc, source_name="НСПД (тест)", data_timestamp=TS)
    assert len(ids) == 2

    near = find_zones(db, bbox=(56.19, 57.99, 56.22, 58.02))
    assert len(near) == 1
    assert near[0].zone_type == "Приаэродромная территория"
    assert near[0].registry_number == "59:00-1.1"
    assert near[0].status == STATUS_OFFICIAL
    assert near[0].document_basis == "Приказ Росавиации №123"
    assert near[0].regime == "ограничение по высоте 50 м"
    assert near[0].version == 1
    assert near[0].superseded_at is None


def test_load_zone_with_existing_registry_number_creates_new_version_not_overwrite(db):
    zone_v1 = ConstraintZone(
        zone_type="ОКН, зона охраны", status=STATUS_OFFICIAL, source_name="ИСОГД (тест)",
        data_timestamp=TS, geom_geojson=SQUARE_NEAR_PERM, registry_number="59:00-9.1",
        document_basis="Приказ №1",
    )
    load_zone(db, zone_v1)

    zone_v2 = ConstraintZone(
        zone_type="ОКН, зона охраны", status=STATUS_OFFICIAL, source_name="ИСОГД (тест, обновление)",
        data_timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc), geom_geojson=SQUARE_NEAR_PERM,
        registry_number="59:00-9.1", document_basis="Приказ №2 (актуализация)",
    )
    load_zone(db, zone_v2)

    current = find_zones(db, bbox=(56.19, 57.99, 56.22, 58.02))
    assert len(current) == 1  # старая версия не видна в текущих
    assert current[0].version == 2
    assert current[0].document_basis == "Приказ №2 (актуализация)"

    history = zone_history(db, "59:00-9.1")
    assert [h.version for h in history] == [2, 1]
    assert history[1].superseded_at is not None  # старая версия сохранена, помечена устаревшей
    assert history[0].superseded_at is None


def test_calculated_zone_without_registry_number_each_load_is_independent(db):
    zone = ConstraintZone(
        zone_type="охранная зона ЛЭП", status=STATUS_CALCULATED, source_name="расчёт (тест)",
        data_timestamp=TS, geom_geojson=SQUARE_NEAR_PERM, regime="не ближе 10 м, ВЛ 10 кВ",
    )
    load_zone(db, zone)
    load_zone(db, zone)

    current = find_zones(db, bbox=(56.19, 57.99, 56.22, 58.02))
    assert len(current) == 2  # без реестрового номера версионировать нечего - две независимые строки
    assert all(z.status == STATUS_CALCULATED for z in current)
    assert all(z.registry_number is None for z in current)


def test_find_zones_include_superseded(db):
    load_zone(db, ConstraintZone(
        zone_type="ПЗЗ", status=STATUS_OFFICIAL, source_name="s", data_timestamp=TS,
        geom_geojson=SQUARE_NEAR_PERM, registry_number="Z1",
    ))
    load_zone(db, ConstraintZone(
        zone_type="ПЗЗ", status=STATUS_OFFICIAL, source_name="s", data_timestamp=TS,
        geom_geojson=SQUARE_NEAR_PERM, registry_number="Z1",
    ))

    current_only = find_zones(db, bbox=(56.19, 57.99, 56.22, 58.02))
    all_versions = find_zones(db, bbox=(56.19, 57.99, 56.22, 58.02), include_superseded=True)
    assert len(current_only) == 1
    assert len(all_versions) == 2
