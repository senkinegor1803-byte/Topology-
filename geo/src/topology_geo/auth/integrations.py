"""API-ключи и вебхуки для внешних интеграций (Шаг 4.11, п. 2 «ключи
доступа», п. 4 «вебхуки о готовности задачи»).

`INTEGRATIONS_SCHEMA_SQL` — встроенная копия `geo/sql/011_integrations.sql`
(синхронность проверена `geo/tests/test_integrations_sql_migrations.py`).

API-ключи хешируются `SHA-256` (не PBKDF2, как пароли пользователей, Шаг
4.10): ключ сам по себе — случайная строка высокой энтропии
(`secrets.token_urlsafe(32)`, не пароль, который человек мог придумать
коротким/предсказуемым) — медленное хеширование здесь не добавляет
защиты сверх той, что уже даёт энтропия самого ключа (тот же подход, что
у большинства платформ с API-ключами, например GitHub personal access
tokens)."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from topology_geo.auth import store as auth_store

API_KEY_PREFIX = "tplg_"


class _Connection(Protocol):
    def cursor(self) -> Any: ...


INTEGRATIONS_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS api_keys (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS api_keys_user_id_idx ON api_keys (user_id);

CREATE TABLE IF NOT EXISTS webhooks (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS webhooks_user_id_idx ON webhooks (user_id);
"""


def ensure_schema(conn: _Connection) -> None:
    auth_store.ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute(INTEGRATIONS_SCHEMA_SQL)


def _hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ApiKey:
    id: int
    label: str
    created_at: datetime
    revoked_at: datetime | None


def create_api_key(conn: _Connection, user_id, *, label: str) -> tuple[str, ApiKey]:
    """Возвращает `(реальный_ключ, метаданные)` — реальный ключ отдаётся
    ТОЛЬКО в момент создания, в базе хранится только его хеш (тот же
    принцип «показать один раз», что у GitHub/Stripe/большинства
    платформ с API-ключами)."""
    raw_key = API_KEY_PREFIX + secrets.token_urlsafe(32)
    key_hash = _hash_api_key(raw_key)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO api_keys (user_id, key_hash, label) VALUES (%s, %s, %s) RETURNING id, created_at",
            (str(user_id), key_hash, label),
        )
        key_id, created_at = cur.fetchone()
    return raw_key, ApiKey(id=key_id, label=label, created_at=created_at, revoked_at=None)


def get_user_by_api_key(conn: _Connection, raw_key: str) -> auth_store.User | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_id FROM api_keys WHERE key_hash = %s AND revoked_at IS NULL",
            (_hash_api_key(raw_key),),
        )
        row = cur.fetchone()
    return None if row is None else auth_store.get_user_by_id(conn, row[0])


def list_api_keys(conn: _Connection, user_id) -> list[ApiKey]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, label, created_at, revoked_at FROM api_keys WHERE user_id = %s ORDER BY created_at DESC",
            (str(user_id),),
        )
        rows = cur.fetchall()
    return [ApiKey(id=r[0], label=r[1], created_at=r[2], revoked_at=r[3]) for r in rows]


def revoke_api_key(conn: _Connection, key_id: int, user_id) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE api_keys SET revoked_at = now() WHERE id = %s AND user_id = %s AND revoked_at IS NULL",
            (key_id, str(user_id)),
        )
        return cur.rowcount > 0


_BLOCKED_HOSTNAMES = {"localhost", "0.0.0.0"}


class InvalidWebhookUrlError(ValueError):
    pass


def _validate_webhook_url(url: str) -> None:
    """Базовая защита от совсем очевидных SSRF-целей (localhost, частные
    сети по буквальному хосту) — НЕ полная защита: DNS rebinding и
    редиректы на приватные адреса не перехватываются (потребовали бы
    резолвить DNS при каждой доставке и проверять IP каждого редиректа —
    отдельная работа, честно не сделанная в этом проходе)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise InvalidWebhookUrlError("URL вебхука должен быть http:// или https://")
    hostname = parsed.hostname or ""
    if hostname.lower() in _BLOCKED_HOSTNAMES:
        raise InvalidWebhookUrlError(f"хост {hostname!r} запрещён")
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return  # не буквальный IP - за DNS-резолвинг не отвечаем (см. докстринг)
    if ip.is_private or ip.is_loopback or ip.is_link_local:
        raise InvalidWebhookUrlError(f"адрес {hostname!r} — приватный/локальный, запрещён")


@dataclass(frozen=True)
class Webhook:
    id: int
    url: str
    created_at: datetime


def register_webhook(conn: _Connection, user_id, *, url: str) -> Webhook:
    _validate_webhook_url(url)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO webhooks (user_id, url) VALUES (%s, %s) RETURNING id, created_at",
            (str(user_id), url),
        )
        webhook_id, created_at = cur.fetchone()
    return Webhook(id=webhook_id, url=url, created_at=created_at)


def list_webhooks(conn: _Connection, user_id) -> list[Webhook]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, url, created_at FROM webhooks WHERE user_id = %s AND revoked_at IS NULL ORDER BY created_at DESC",
            (str(user_id),),
        )
        rows = cur.fetchall()
    return [Webhook(id=r[0], url=r[1], created_at=r[2]) for r in rows]


def revoke_webhook(conn: _Connection, webhook_id: int, user_id) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE webhooks SET revoked_at = now() WHERE id = %s AND user_id = %s AND revoked_at IS NULL",
            (webhook_id, str(user_id)),
        )
        return cur.rowcount > 0


def deliver_webhook(url: str, payload: dict, *, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Настоящий HTTP POST (`urllib` из стандартной библиотеки — без новой
    зависимости), не имитация доставки. Без повторов при неудаче — честно:
    очередь с ретраями/экспоненциальной задержкой — отдельная работа, не
    часть этого прохода."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status < 400, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, str(exc.reason)
