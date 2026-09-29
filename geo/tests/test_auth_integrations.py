"""Тесты Шага 4.11: API-ключи и вебхуки — на реальном PostGIS (`pg_test_db`)
и настоящем HTTP-сервере для доставки вебхука."""

from __future__ import annotations

import functools
import http.server
import json
import threading

import pytest

from topology_geo.auth.integrations import (
    InvalidWebhookUrlError,
    create_api_key,
    deliver_webhook,
    ensure_schema,
    get_user_by_api_key,
    list_api_keys,
    list_webhooks,
    register_webhook,
    revoke_api_key,
    revoke_webhook,
)
from topology_geo.auth.store import create_user


@pytest.fixture()
def db(pg_test_db):
    ensure_schema(pg_test_db)
    return pg_test_db


def test_create_api_key_roundtrip(db):
    user = create_user(db, email="apikey@example.com", password="secret123")
    raw_key, meta = create_api_key(db, user.id, label="CI-интеграция")

    assert raw_key.startswith("tplg_")
    assert meta.label == "CI-интеграция"

    fetched = get_user_by_api_key(db, raw_key)
    assert fetched is not None
    assert fetched.id == user.id


def test_get_user_by_api_key_rejects_unknown_key(db):
    assert get_user_by_api_key(db, "tplg_not-a-real-key") is None


def test_revoked_api_key_stops_working(db):
    user = create_user(db, email="revoke@example.com", password="secret123")
    raw_key, meta = create_api_key(db, user.id, label="temp")

    assert revoke_api_key(db, meta.id, user.id) is True
    assert get_user_by_api_key(db, raw_key) is None


def test_revoke_api_key_rejects_wrong_owner(db):
    owner = create_user(db, email="owner-key@example.com", password="secret123")
    stranger = create_user(db, email="stranger-key@example.com", password="secret123")
    _, meta = create_api_key(db, owner.id, label="mine")

    assert revoke_api_key(db, meta.id, stranger.id) is False


def test_list_api_keys_does_not_expose_raw_key(db):
    user = create_user(db, email="list-key@example.com", password="secret123")
    create_api_key(db, user.id, label="a")
    create_api_key(db, user.id, label="b")

    keys = list_api_keys(db, user.id)
    assert len(keys) == 2
    assert {k.label for k in keys} == {"a", "b"}
    assert not any(hasattr(k, "raw_key") for k in keys)


def test_register_webhook_rejects_non_http_scheme(db):
    user = create_user(db, email="wh1@example.com", password="secret123")
    with pytest.raises(InvalidWebhookUrlError):
        register_webhook(db, user.id, url="ftp://example.com/hook")


@pytest.mark.parametrize("url", [
    "http://localhost/hook",
    "http://127.0.0.1:8000/hook",
    "http://0.0.0.0/hook",
    "http://192.168.1.5/hook",
    "http://10.0.0.5/hook",
])
def test_register_webhook_rejects_private_and_local_hosts(db, url):
    user = create_user(db, email="wh2@example.com", password="secret123")
    with pytest.raises(InvalidWebhookUrlError):
        register_webhook(db, user.id, url=url)


def test_register_and_list_webhook(db):
    user = create_user(db, email="wh3@example.com", password="secret123")
    webhook = register_webhook(db, user.id, url="https://example.com/hook")

    webhooks = list_webhooks(db, user.id)
    assert len(webhooks) == 1
    assert webhooks[0].id == webhook.id
    assert webhooks[0].url == "https://example.com/hook"


def test_revoke_webhook_removes_it_from_active_list(db):
    user = create_user(db, email="wh4@example.com", password="secret123")
    webhook = register_webhook(db, user.id, url="https://example.com/hook")

    assert revoke_webhook(db, webhook.id, user.id) is True
    assert list_webhooks(db, user.id) == []


def test_revoke_webhook_rejects_wrong_owner(db):
    owner = create_user(db, email="wh-owner@example.com", password="secret123")
    stranger = create_user(db, email="wh-stranger@example.com", password="secret123")
    webhook = register_webhook(db, owner.id, url="https://example.com/hook")

    assert revoke_webhook(db, webhook.id, stranger.id) is False
    assert len(list_webhooks(db, owner.id)) == 1


class _CapturingHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, received: list, **kwargs):
        self._received = received
        super().__init__(*args, **kwargs)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self._received.append(json.loads(body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format, *args):
        pass


def test_deliver_webhook_makes_real_http_post():
    received: list = []
    handler = functools.partial(_CapturingHandler, received=received)
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        success, detail = deliver_webhook(f"http://127.0.0.1:{port}/hook", {"job_id": "abc", "status": "done"})
        assert success is True
        assert "200" in detail
        assert received == [{"job_id": "abc", "status": "done"}]
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def test_deliver_webhook_reports_connection_failure():
    success, detail = deliver_webhook("http://127.0.0.1:1/hook", {"x": 1}, timeout_s=1.0)
    assert success is False
    assert detail
