"""Тесты Шага 4.11: внешний API и интеграции — ключи доступа, вебхуки,
выгрузка в Pilot-BIM. Тот же приём подключения приложения, что в
`test_api.py`/`test_api_auth.py`."""

from __future__ import annotations

import functools
import http.server
import json
import sys
import threading
from pathlib import Path

import pytest


def _make_client(dbname: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **extra_env: str):
    for name in list(sys.modules):
        if name.startswith("topology_geo.tasks") or name.startswith("topology_geo.api"):
            del sys.modules[name]

    monkeypatch.setenv("POSTGRES_DB", dbname)
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    monkeypatch.setenv("TOPOLOGY_STORAGE_ROOT", str(tmp_path / "storage"))
    for key, value in extra_env.items():
        monkeypatch.setenv(key, value)

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


def test_create_api_key_returns_raw_key_once(client):
    body = _register(client, email="apikey@example.com")
    resp = client.post("/me/api-keys", json={"label": "CI"}, headers=_auth_headers(body["token"]))
    assert resp.status_code == 201
    data = resp.json()
    assert data["key"].startswith("tplg_")
    assert data["label"] == "CI"


def test_list_api_keys_never_includes_raw_key(client):
    body = _register(client, email="apikey2@example.com")
    client.post("/me/api-keys", json={"label": "CI"}, headers=_auth_headers(body["token"]))

    listing = client.get("/me/api-keys", headers=_auth_headers(body["token"]))
    assert listing.status_code == 200
    assert "key" not in listing.json()[0]


def test_api_key_authenticates_requests(client):
    body = _register(client, email="apikey3@example.com")
    key_resp = client.post("/me/api-keys", json={"label": "CI"}, headers=_auth_headers(body["token"]))
    raw_key = key_resp.json()["key"]

    resp = client.get("/auth/me", headers={"X-API-Key": raw_key})
    assert resp.status_code == 200
    assert resp.json()["email"] == "apikey3@example.com"


def test_revoked_api_key_stops_authenticating(client):
    body = _register(client, email="apikey4@example.com")
    key_resp = client.post("/me/api-keys", json={"label": "CI"}, headers=_auth_headers(body["token"]))
    key_id, raw_key = key_resp.json()["id"], key_resp.json()["key"]

    revoke_resp = client.delete(f"/me/api-keys/{key_id}", headers=_auth_headers(body["token"]))
    assert revoke_resp.status_code == 204

    resp = client.get("/auth/me", headers={"X-API-Key": raw_key})
    assert resp.status_code == 401


def test_api_key_can_create_jobs(client):
    """Ключ доступа — полноценная альтернатива сессионному токену (Шаг
    4.11, п. 2: «ключи доступа» для внешних интеграций, не только для
    людей за браузером)."""
    body = _register(client, email="apikey5@example.com")
    key_resp = client.post("/me/api-keys", json={"label": "integration"}, headers=_auth_headers(body["token"]))
    raw_key = key_resp.json()["key"]

    create_resp = client.post(
        "/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500}, headers={"X-API-Key": raw_key},
    )
    job_id = create_resp.json()["id"]

    projects = client.get("/projects", headers=_auth_headers(body["token"]))
    assert job_id in [p["id"] for p in projects.json()]


def test_register_webhook_and_list(client):
    body = _register(client, email="wh@example.com")
    resp = client.post("/me/webhooks", json={"url": "https://example.com/hook"}, headers=_auth_headers(body["token"]))
    assert resp.status_code == 201

    listing = client.get("/me/webhooks", headers=_auth_headers(body["token"]))
    assert len(listing.json()) == 1


def test_register_webhook_rejects_private_url(client):
    body = _register(client, email="wh2@example.com")
    resp = client.post("/me/webhooks", json={"url": "http://127.0.0.1/hook"}, headers=_auth_headers(body["token"]))
    assert resp.status_code == 422


def test_revoke_webhook(client):
    body = _register(client, email="wh3@example.com")
    create_resp = client.post("/me/webhooks", json={"url": "https://example.com/hook"}, headers=_auth_headers(body["token"]))
    webhook_id = create_resp.json()["id"]

    revoke_resp = client.delete(f"/me/webhooks/{webhook_id}", headers=_auth_headers(body["token"]))
    assert revoke_resp.status_code == 204
    assert client.get("/me/webhooks", headers=_auth_headers(body["token"])).json() == []


class _CapturingHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, received: list, **kwargs):
        self._received = received
        super().__init__(*args, **kwargs)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self._received.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass


def test_webhook_fires_on_job_completion(pg_test_db, tmp_path, monkeypatch):
    """Сквозная проверка Шага 4.11, п. 4: реальный HTTP-сервер получает
    настоящий POST, когда задача (без seed-данных OSM) падает.

    Вебхук заведён напрямую через SQL, в обход `register_webhook`: та
    функция намеренно отвергает `127.0.0.1` (см. `_validate_webhook_url` -
    защита от SSRF, `test_register_webhook_rejects_private_url`), а
    локальный тестовый HTTP-сервер как раз на нём и поднят - здесь
    проверяется ДОСТАВКА из `pipeline_tasks`, не сама валидация."""
    received: list = []
    handler = functools.partial(_CapturingHandler, received=received)
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        with _make_client(pg_test_db.info.dbname, tmp_path, monkeypatch) as client:
            body = _register(client, email="webhookfire@example.com")
            with pg_test_db.cursor() as cur:
                cur.execute(
                    "INSERT INTO webhooks (user_id, url) VALUES (%s, %s)",
                    (body["user"]["id"], f"http://127.0.0.1:{port}/hook"),
                )
            create_resp = client.post(
                "/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500},
                headers=_auth_headers(body["token"]),
            )
            job_id = create_resp.json()["id"]
            assert client.get(f"/jobs/{job_id}").json()["status"] == "failed"

        assert len(received) == 1
        assert received[0]["job_id"] == job_id
        assert received[0]["status"] == "failed"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def test_export_to_pilot_bim_requires_configured_folder(client):
    body = _register(client, email="pilot1@example.com")
    create_resp = client.post(
        "/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500}, headers=_auth_headers(body["token"]),
    )
    job_id = create_resp.json()["id"]

    resp = client.post(f"/jobs/{job_id}/export-to-pilot-bim", headers=_auth_headers(body["token"]))
    assert resp.status_code == 503


def test_export_to_pilot_bim_requires_authorization(client):
    resp = client.post("/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500})
    job_id = resp.json()["id"]

    export_resp = client.post(f"/jobs/{job_id}/export-to-pilot-bim")
    assert export_resp.status_code == 401


def test_export_to_pilot_bim_rejects_non_owner(pg_test_db, tmp_path, monkeypatch):
    exchange_dir = tmp_path / "pilot-exchange"
    with _make_client(pg_test_db.info.dbname, tmp_path, monkeypatch, PILOT_BIM_EXCHANGE_DIR=str(exchange_dir)) as client:
        owner = _register(client, email="pilot-owner@example.com")
        stranger = _register(client, email="pilot-stranger@example.com")

        create_resp = client.post(
            "/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500}, headers=_auth_headers(owner["token"]),
        )
        job_id = create_resp.json()["id"]

        resp = client.post(f"/jobs/{job_id}/export-to-pilot-bim", headers=_auth_headers(stranger["token"]))
        assert resp.status_code == 403


def test_export_to_pilot_bim_404_without_ifc_model(pg_test_db, tmp_path, monkeypatch):
    exchange_dir = tmp_path / "pilot-exchange"
    with _make_client(pg_test_db.info.dbname, tmp_path, monkeypatch, PILOT_BIM_EXCHANGE_DIR=str(exchange_dir)) as client:
        body = _register(client, email="pilot2@example.com")
        create_resp = client.post(
            "/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500}, headers=_auth_headers(body["token"]),
        )
        job_id = create_resp.json()["id"]  # без seed-данных OSM модель не собирается

        resp = client.post(f"/jobs/{job_id}/export-to-pilot-bim", headers=_auth_headers(body["token"]))
        assert resp.status_code == 404


def test_openapi_schema_is_served():
    """«Документированный REST API (OpenAPI)» (Шаг 4.11, п. 1) — FastAPI
    генерирует схему сам, здесь только подтверждаем, что она реально
    отдаётся и содержит новые эндпоинты этого шага."""
    from fastapi.testclient import TestClient

    from topology_geo.api.app import app

    client = TestClient(app)
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/me/api-keys" in paths
    assert "/me/webhooks" in paths
    assert "/jobs/{job_id}/export-to-pilot-bim" in paths

    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200
