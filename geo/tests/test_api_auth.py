"""Тесты Шага 4.10: личный кабинет — авторизация, роли, проекты,
уведомления, публичные ссылки, доступ к закрытому контуру. Тот же приём
подключения приложения, что в `test_api.py` (см. его докстринг)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from topology_geo.constraints.store import STATUS_OFFICIAL, ConstraintZone, load_zone


def _make_client(dbname: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(sys.modules):
        if name.startswith("topology_geo.tasks") or name.startswith("topology_geo.api"):
            del sys.modules[name]

    monkeypatch.setenv("POSTGRES_DB", dbname)
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    monkeypatch.setenv("TOPOLOGY_STORAGE_ROOT", str(tmp_path / "storage"))

    from fastapi.testclient import TestClient

    from topology_geo.api.app import app

    return TestClient(app)


@pytest.fixture()
def client(pg_test_db, tmp_path, monkeypatch):
    with _make_client(pg_test_db.info.dbname, tmp_path, monkeypatch) as c:
        yield c


def _register(client, email="user@example.com", password="secret123"):
    resp = client.post("/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 201, resp.text
    return resp.json()


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_register_creates_user_with_viewer_role(client):
    body = _register(client)
    assert body["user"]["role"] == "просмотр"
    assert body["user"]["closed_contour_access"] is False
    assert "token" in body


def test_register_rejects_duplicate_email(client):
    _register(client, email="dup@example.com")
    resp = client.post("/auth/register", json={"email": "dup@example.com", "password": "another123"})
    assert resp.status_code == 409


def test_register_rejects_short_password(client):
    resp = client.post("/auth/register", json={"email": "x@example.com", "password": "short"})
    assert resp.status_code == 422


def test_login_with_correct_and_incorrect_password(client):
    _register(client, email="login@example.com", password="secret123")

    ok = client.post("/auth/login", json={"email": "login@example.com", "password": "secret123"})
    assert ok.status_code == 200
    assert "token" in ok.json()

    bad = client.post("/auth/login", json={"email": "login@example.com", "password": "wrong"})
    assert bad.status_code == 401


def test_me_requires_authorization(client):
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_me_returns_current_user(client):
    body = _register(client, email="me@example.com")
    resp = client.get("/auth/me", headers=_auth_headers(body["token"]))
    assert resp.status_code == 200
    assert resp.json()["email"] == "me@example.com"


def test_logout_revokes_token(client):
    body = _register(client, email="out@example.com")
    token = body["token"]

    logout_resp = client.post("/auth/logout", headers=_auth_headers(token))
    assert logout_resp.status_code == 204

    resp = client.get("/auth/me", headers=_auth_headers(token))
    assert resp.status_code == 401


def test_role_update_requires_admin(client):
    viewer = _register(client, email="viewer@example.com")
    other = _register(client, email="other@example.com")

    resp = client.patch(
        f"/auth/users/{other['user']['id']}/role",
        json={"role": "администратор", "closed_contour_access": True},
        headers=_auth_headers(viewer["token"]),
    )
    assert resp.status_code == 403


def test_admin_can_promote_another_user(client, pg_test_db):
    from topology_geo.auth.store import ROLE_ADMIN, ensure_schema, set_user_role

    ensure_schema(pg_test_db)
    admin = _register(client, email="admin@example.com")
    set_user_role(pg_test_db, admin["user"]["id"], role=ROLE_ADMIN, closed_contour_access=True)

    target = _register(client, email="target@example.com")
    resp = client.patch(
        f"/auth/users/{target['user']['id']}/role",
        json={"role": "проектировщик", "closed_contour_access": True},
        headers=_auth_headers(admin["token"]),
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "проектировщик"
    assert resp.json()["closed_contour_access"] is True


def test_anonymous_job_creation_still_works_without_token(client):
    """Регрессия: существующее поведение Шага 1.3 (задача без авторизации)
    не должно сломаться после добавления авторизации."""
    resp = client.post("/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500})
    assert resp.status_code == 201


def test_job_created_with_token_appears_in_projects(client):
    body = _register(client, email="owner@example.com")
    token = body["token"]

    create_resp = client.post(
        "/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500}, headers=_auth_headers(token),
    )
    job_id = create_resp.json()["id"]

    projects = client.get("/projects", headers=_auth_headers(token))
    assert projects.status_code == 200
    ids = [p["id"] for p in projects.json()]
    assert job_id in ids


def test_projects_requires_authorization(client):
    resp = client.get("/projects")
    assert resp.status_code == 401


def test_projects_only_shows_own_jobs(client):
    a = _register(client, email="a@example.com")
    b = _register(client, email="b@example.com")

    client.post("/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500}, headers=_auth_headers(a["token"]))

    projects_b = client.get("/projects", headers=_auth_headers(b["token"]))
    assert projects_b.json() == []


def test_job_completion_creates_notification_for_owner(client):
    """Шаг 4.10, п. 3: задача без seed-данных OSM падает (см.
    test_create_job_then_get_shows_failed_without_seed_data в test_api.py) -
    это тоже терминальный статус и тоже должно породить уведомление."""
    body = _register(client, email="notif@example.com")
    token = body["token"]

    create_resp = client.post(
        "/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500}, headers=_auth_headers(token),
    )
    job_id = create_resp.json()["id"]
    assert client.get(f"/jobs/{job_id}").json()["status"] == "failed"

    notifications = client.get("/me/notifications", headers=_auth_headers(token))
    assert notifications.status_code == 200
    messages = [n["message"] for n in notifications.json()]
    assert any("ошибкой" in m for m in messages)


def test_mark_notification_read(client, pg_test_db):
    from topology_geo.auth.store import create_notification, ensure_schema, get_user_by_token

    ensure_schema(pg_test_db)
    body = _register(client, email="read@example.com")
    token = body["token"]
    user = get_user_by_token(pg_test_db, token)
    notification_id = create_notification(pg_test_db, user_id=user.id, message="Тест")

    unread = client.get("/me/notifications?unread_only=true", headers=_auth_headers(token))
    assert len(unread.json()) == 1

    mark_resp = client.post(f"/me/notifications/{notification_id}/read", headers=_auth_headers(token))
    assert mark_resp.status_code == 204

    unread_after = client.get("/me/notifications?unread_only=true", headers=_auth_headers(token))
    assert unread_after.json() == []


def test_share_link_requires_authorization(client):
    resp = client.post("/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500})
    job_id = resp.json()["id"]

    share_resp = client.post(f"/jobs/{job_id}/share")
    assert share_resp.status_code == 401


def test_share_link_rejects_non_owner(client):
    owner = _register(client, email="shareowner@example.com")
    stranger = _register(client, email="stranger@example.com")

    create_resp = client.post(
        "/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500}, headers=_auth_headers(owner["token"]),
    )
    job_id = create_resp.json()["id"]

    resp = client.post(f"/jobs/{job_id}/share", headers=_auth_headers(stranger["token"]))
    assert resp.status_code == 403


def test_public_link_404_without_ready_model(client):
    body = _register(client, email="pub@example.com")
    create_resp = client.post(
        "/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500}, headers=_auth_headers(body["token"]),
    )
    job_id = create_resp.json()["id"]  # без seed-данных OSM модель не собирается

    share_resp = client.post(f"/jobs/{job_id}/share", headers=_auth_headers(body["token"]))
    assert share_resp.status_code == 200
    token = share_resp.json()["token"]

    public_resp = client.get(f"/public/{token}", follow_redirects=False)
    assert public_resp.status_code == 404


def test_public_link_404_for_unknown_token(client):
    resp = client.get("/public/does-not-exist", follow_redirects=False)
    assert resp.status_code == 404


def test_closed_contour_requires_permission(client):
    resp = client.post("/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500})
    job_id = resp.json()["id"]

    viewer = _register(client, email="cc-viewer@example.com")
    denied = client.get(f"/projects/{job_id}/closed-contour", headers=_auth_headers(viewer["token"]))
    assert denied.status_code == 403

    no_auth = client.get(f"/projects/{job_id}/closed-contour")
    assert no_auth.status_code == 401


def test_closed_contour_returns_real_zones_for_users_with_access(client, pg_test_db):
    from topology_geo.auth.store import ROLE_ADMIN, ensure_schema, set_user_role
    from topology_geo.constraints.store import ensure_schema as ensure_constraints_schema

    ensure_schema(pg_test_db)
    ensure_constraints_schema(pg_test_db)
    body = _register(client, email="cc-admin@example.com")
    set_user_role(pg_test_db, body["user"]["id"], role=ROLE_ADMIN, closed_contour_access=True)

    create_resp = client.post("/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500})
    job_id = create_resp.json()["id"]

    load_zone(pg_test_db, ConstraintZone(
        zone_type="ЗОУИТ", status=STATUS_OFFICIAL, source_name="тест",
        data_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        geom_geojson={
            "type": "MultiPolygon",
            "coordinates": [[[[56.0, 57.9], [56.5, 57.9], [56.5, 58.1], [56.0, 58.1], [56.0, 57.9]]]],
        },
        registry_number="ТЕСТ-1",
    ))

    resp = client.get(f"/projects/{job_id}/closed-contour", headers=_auth_headers(body["token"]))
    assert resp.status_code == 200
    zones = resp.json()["zones"]
    assert any(z["registry_number"] == "ТЕСТ-1" for z in zones)
