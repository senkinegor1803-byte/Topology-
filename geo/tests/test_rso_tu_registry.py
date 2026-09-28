"""Тесты Шага 3.12: реестр запросов ТУ, напоминания о сроках, привязка
точки подключения (реальный PostGIS)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from topology_geo.rso.tu_registry import (
    DEFAULT_RESPONSE_DEADLINE_DAYS,
    STATUS_OVERDUE,
    STATUS_RECEIVED,
    STATUS_SENT,
    ensure_schema,
    get_tu_request,
    mark_overdue,
    overdue_requests,
    record_response,
    register_tu_request,
)

POINT = {"type": "Point", "coordinates": [56.24, 58.01]}


@pytest.fixture()
def db(pg_test_db):
    ensure_schema(pg_test_db)
    return pg_test_db


def test_register_tu_request_computes_deadline_from_sent_at(db):
    sent_at = date(2026, 1, 1)
    request_id = register_tu_request(
        db, organization_name="Пермводоканал", network_type="В", channel="Госуслуги", sent_at=sent_at,
    )

    request = get_tu_request(db, request_id)

    assert request.status == STATUS_SENT
    assert request.deadline == sent_at + timedelta(days=DEFAULT_RESPONSE_DEADLINE_DAYS)


def test_register_tu_request_custom_deadline_days(db):
    sent_at = date(2026, 1, 1)
    request_id = register_tu_request(
        db, organization_name="МРСК", network_type="Кл", channel="письмо", sent_at=sent_at,
        response_deadline_days=14,
    )

    request = get_tu_request(db, request_id)
    assert request.deadline == date(2026, 1, 15)


def test_record_response_updates_status_and_connection_point(db):
    request_id = register_tu_request(
        db, organization_name="Газпром межрегионгаз", network_type="Г", channel="личный кабинет",
        sent_at=date(2026, 1, 1),
    )

    record_response(
        db, request_id, incoming_number="ТУ-123/2026", received_at=date(2026, 1, 20),
        connection_point_geojson=POINT, connection_params="давление 0.3 МПа, диаметр 100 мм",
    )

    request = get_tu_request(db, request_id)
    assert request.status == STATUS_RECEIVED
    assert request.incoming_number == "ТУ-123/2026"
    assert request.connection_point_geojson == POINT
    assert request.connection_params == "давление 0.3 МПа, диаметр 100 мм"


def test_overdue_requests_finds_only_expired_unanswered(db):
    overdue_id = register_tu_request(
        db, organization_name="A", network_type="В", channel="письмо", sent_at=date(2025, 1, 1),
        response_deadline_days=30,
    )
    fresh_id = register_tu_request(
        db, organization_name="B", network_type="Т", channel="МФЦ", sent_at=date.today(),
        response_deadline_days=30,
    )

    overdue = overdue_requests(db, as_of=date.today())

    assert {r.id for r in overdue} == {overdue_id}
    assert fresh_id not in {r.id for r in overdue}


def test_overdue_requests_excludes_already_answered(db):
    request_id = register_tu_request(
        db, organization_name="A", network_type="В", channel="письмо", sent_at=date(2025, 1, 1),
        response_deadline_days=30,
    )
    record_response(db, request_id, incoming_number="TU-1", received_at=date(2025, 2, 1))

    assert overdue_requests(db, as_of=date.today()) == []


def test_mark_overdue_sets_status_and_returns_count(db):
    register_tu_request(
        db, organization_name="A", network_type="В", channel="письмо", sent_at=date(2025, 1, 1),
        response_deadline_days=30,
    )

    updated = mark_overdue(db, as_of=date.today())

    assert updated == 1
    requests = overdue_requests(db, as_of=date(2020, 1, 1))  # уже не "отправлен", а "просрочен" - не найдётся тем же запросом
    assert requests == []
    all_requests = [get_tu_request(db, 1)]
    assert all_requests[0].status == STATUS_OVERDUE
