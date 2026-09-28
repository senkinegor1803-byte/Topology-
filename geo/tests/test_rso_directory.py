"""Тесты Шага 3.10: справочник РСО и каскад определения владельца сети
(реальный PostGIS)."""

from __future__ import annotations

from datetime import date

import pytest

from topology_geo.rso.directory import (
    CHANNEL_GOSUSLUGI,
    STATUS_NEEDS_CLARIFICATION,
    RsoEntry,
    ensure_schema,
    find_rso_for_point,
    register_rso,
    resolve_network_owner,
)

SQUARE = {"type": "Polygon", "coordinates": [[
    [56.0, 58.0], [57.0, 58.0], [57.0, 59.0], [56.0, 59.0], [56.0, 58.0],
]]}


@pytest.fixture()
def db(pg_test_db):
    ensure_schema(pg_test_db)
    return pg_test_db


def test_register_and_find_rso_for_point(db):
    register_rso(db, RsoEntry(
        network_type="В", organization_name="Пермводоканал", contacts="tel: 000",
        application_channel=CHANNEL_GOSUSLUGI, verified_at=date(2026, 1, 1), geom_geojson=SQUARE,
    ))

    entry = find_rso_for_point(db, "В", lon=56.5, lat=58.5)

    assert entry is not None
    assert entry.organization_name == "Пермводоканал"
    assert entry.application_channel == CHANNEL_GOSUSLUGI


def test_find_rso_for_point_outside_service_area_returns_none(db):
    register_rso(db, RsoEntry(
        network_type="В", organization_name="Пермводоканал", application_channel=CHANNEL_GOSUSLUGI,
        verified_at=date(2026, 1, 1), geom_geojson=SQUARE,
    ))

    assert find_rso_for_point(db, "В", lon=10.0, lat=10.0) is None


def test_find_rso_for_point_wrong_network_type_returns_none(db):
    register_rso(db, RsoEntry(
        network_type="В", organization_name="Пермводоканал", application_channel=CHANNEL_GOSUSLUGI,
        verified_at=date(2026, 1, 1), geom_geojson=SQUARE,
    ))

    assert find_rso_for_point(db, "Г", lon=56.5, lat=58.5) is None


def test_resolve_network_owner_prefers_osm_operator(db):
    register_rso(db, RsoEntry(
        network_type="В", organization_name="Пермводоканал", application_channel=CHANNEL_GOSUSLUGI,
        verified_at=date(2026, 1, 1), geom_geojson=SQUARE,
    ))

    resolution = resolve_network_owner(db, "В", lon=56.5, lat=58.5, osm_operator="ООО Водосервис")

    assert resolution.owner_name == "ООО Водосервис"
    assert resolution.source == "OSM operator"


def test_resolve_network_owner_falls_back_to_directory(db):
    register_rso(db, RsoEntry(
        network_type="В", organization_name="Пермводоканал", application_channel=CHANNEL_GOSUSLUGI,
        verified_at=date(2026, 1, 1), geom_geojson=SQUARE,
    ))

    resolution = resolve_network_owner(db, "В", lon=56.5, lat=58.5, osm_operator=None)

    assert resolution.owner_name == "Пермводоканал"
    assert resolution.source == "справочник РСО"
    assert resolution.application_channel == CHANNEL_GOSUSLUGI


def test_resolve_network_owner_needs_clarification_when_nothing_found(db):
    resolution = resolve_network_owner(db, "Т", lon=56.5, lat=58.5, osm_operator=None)

    assert resolution.owner_name is None
    assert resolution.source == STATUS_NEEDS_CLARIFICATION
