"""Тесты Шага 4.10: авторизация, роли, владение задачами, уведомления,
публичные ссылки — на реальном PostGIS (`pg_test_db`)."""

from __future__ import annotations

import uuid

import pytest

from topology_geo.auth.store import (
    ROLE_ADMIN,
    ROLE_DESIGNER,
    ROLE_VIEWER,
    EmailAlreadyRegisteredError,
    authenticate,
    create_notification,
    create_public_link,
    create_session,
    create_user,
    ensure_schema,
    get_job_owner,
    get_user_by_id,
    get_user_by_token,
    hash_password,
    list_job_ids_for_user,
    list_notifications,
    mark_notification_read,
    record_job_ownership,
    resolve_public_link,
    revoke_public_link,
    revoke_session,
    set_user_role,
    verify_password,
)
from topology_geo.jobs.steps import DEFAULT_STEP_NAMES
from topology_geo.jobs.store import create_job


@pytest.fixture()
def db(pg_test_db):
    ensure_schema(pg_test_db)
    return pg_test_db


def test_hash_password_roundtrip():
    password_hash = hash_password("secret123")
    assert verify_password("secret123", password_hash)
    assert not verify_password("wrong", password_hash)


def test_hash_password_uses_random_salt():
    assert hash_password("secret123") != hash_password("secret123")


def test_create_user_and_authenticate(db):
    user = create_user(db, email="a@example.com", password="secret123", role=ROLE_DESIGNER)
    assert user.role == ROLE_DESIGNER
    assert user.closed_contour_access is False

    authenticated = authenticate(db, "a@example.com", "secret123")
    assert authenticated is not None
    assert authenticated.id == user.id

    assert authenticate(db, "a@example.com", "wrong-password") is None
    assert authenticate(db, "unknown@example.com", "secret123") is None


def test_create_user_defaults_to_viewer_role(db):
    user = create_user(db, email="viewer@example.com", password="secret123")
    assert user.role == ROLE_VIEWER


def test_create_user_rejects_unknown_role(db):
    with pytest.raises(ValueError, match="неизвестная роль"):
        create_user(db, email="x@example.com", password="secret123", role="суперпользователь")


def test_create_user_rejects_duplicate_email(db):
    create_user(db, email="dup@example.com", password="secret123")
    with pytest.raises(EmailAlreadyRegisteredError):
        create_user(db, email="dup@example.com", password="another")


def test_get_user_by_id_returns_none_for_unknown(db):
    assert get_user_by_id(db, uuid.uuid4()) is None


def test_session_lifecycle(db):
    user = create_user(db, email="s@example.com", password="secret123")
    token = create_session(db, user.id)

    fetched = get_user_by_token(db, token)
    assert fetched is not None
    assert fetched.id == user.id

    revoke_session(db, token)
    assert get_user_by_token(db, token) is None


def test_get_user_by_token_rejects_unknown_token(db):
    assert get_user_by_token(db, "not-a-real-token") is None


def test_set_user_role_updates_role_and_closed_contour_access(db):
    user = create_user(db, email="role@example.com", password="secret123")
    set_user_role(db, user.id, role=ROLE_ADMIN, closed_contour_access=True)

    updated = get_user_by_id(db, user.id)
    assert updated.role == ROLE_ADMIN
    assert updated.closed_contour_access is True


def test_job_ownership_roundtrip(db):
    user = create_user(db, email="owner@example.com", password="secret123")
    job = create_job(
        db, center_lon=56.24, center_lat=58.01, radius_m=500, layers=[], detail="LOD1",
        step_names=DEFAULT_STEP_NAMES,
    )

    assert get_job_owner(db, job.id) is None
    record_job_ownership(db, job.id, user.id)
    assert get_job_owner(db, job.id) == user.id
    assert list_job_ids_for_user(db, user.id) == [job.id]


def test_job_ownership_ignores_duplicate_recording(db):
    user = create_user(db, email="owner2@example.com", password="secret123")
    job = create_job(
        db, center_lon=56.24, center_lat=58.01, radius_m=500, layers=[], detail="LOD1",
        step_names=DEFAULT_STEP_NAMES,
    )
    record_job_ownership(db, job.id, user.id)
    record_job_ownership(db, job.id, user.id)  # не должно упасть/задвоить
    assert list_job_ids_for_user(db, user.id) == [job.id]


def test_notifications_roundtrip(db):
    user = create_user(db, email="notif@example.com", password="secret123")
    job = create_job(
        db, center_lon=56.24, center_lat=58.01, radius_m=500, layers=[], detail="LOD1",
        step_names=DEFAULT_STEP_NAMES,
    )

    notification_id = create_notification(db, user_id=user.id, message="Задача выполнена", job_id=job.id)

    all_notifications = list_notifications(db, user.id)
    assert len(all_notifications) == 1
    assert all_notifications[0].message == "Задача выполнена"
    assert all_notifications[0].read_at is None

    unread = list_notifications(db, user.id, unread_only=True)
    assert len(unread) == 1

    mark_notification_read(db, notification_id)
    assert list_notifications(db, user.id, unread_only=True) == []
    assert list_notifications(db, user.id)[0].read_at is not None


def test_public_link_roundtrip(db):
    job = create_job(
        db, center_lon=56.24, center_lat=58.01, radius_m=500, layers=[], detail="LOD1",
        step_names=DEFAULT_STEP_NAMES,
    )
    token = create_public_link(db, job.id)

    assert resolve_public_link(db, token) == job.id

    revoke_public_link(db, token)
    assert resolve_public_link(db, token) is None


def test_resolve_public_link_rejects_unknown_token(db):
    assert resolve_public_link(db, "not-a-real-token") is None
