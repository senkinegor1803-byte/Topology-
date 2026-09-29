"""Авторизация, роли, проекты пользователя, уведомления (Шаг 4.10, п. 1-3).

`AUTH_SCHEMA_SQL` — встроенная копия `geo/sql/010_auth.sql` (тот же приём,
что и `jobs.store.JOBS_SCHEMA_SQL`; синхронность проверена тестом
`geo/tests/test_auth_sql_migrations.py`).

Владение задачей (`job_ownership`) и уведомления живут в ОТДЕЛЬНЫХ таблицах,
не колонкой `jobs.owner_id`: `jobs/store.py` — уже существующая, широко
используемая (десятки тестов Шагов 1.3-3.13) инфраструктура, где `owner_id`
всегда `NULL` (анонимная задача); добавление обязательной колонки с внешним
ключом на `users` создало бы обратную зависимость миграции 003 от миграции
010 (порядок номеров нарушился бы) и требовало бы трогать `create_job`,
рискуя регрессией по всему проекту. Отдельная таблица-связка — тот же приём
расширения без изменения существующей схемы, что уже применялся в проекте
(например, `constraints.store` не трогает `osm_import_log`)."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import psycopg.errors

from topology_geo.jobs import store as jobs_store

ROLE_VIEWER = "просмотр"
ROLE_DESIGNER = "проектировщик"
ROLE_ADMIN = "администратор"
ROLES = (ROLE_VIEWER, ROLE_DESIGNER, ROLE_ADMIN)

# «Доступ к закрытому контуру» (п. 1) — не отдельная роль, а флаг поверх
# ролей: по формулировке плана это ПРАВО (доступ к данным сетей/изысканий,
# см. `docs/plan.md` п. 10.4 — «доступ к сетям и изысканиям — только по
# ролям»), а не должность вроде «просмотр»/«администратор».
SESSION_TTL_HOURS = 24
# 260 000 итераций - текущая (2023+) рекомендация OWASP для PBKDF2-HMAC-SHA256.
PBKDF2_ITERATIONS = 260_000


class _Connection(Protocol):
    def cursor(self) -> Any: ...


AUTH_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('просмотр', 'проектировщик', 'администратор')),
    closed_contour_access BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions (user_id);

CREATE TABLE IF NOT EXISTS job_ownership (
    job_id UUID PRIMARY KEY REFERENCES jobs (id) ON DELETE CASCADE,
    owner_id UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS job_ownership_owner_id_idx ON job_ownership (owner_id);

CREATE TABLE IF NOT EXISTS notifications (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    job_id UUID REFERENCES jobs (id) ON DELETE CASCADE,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    read_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS notifications_user_id_idx ON notifications (user_id, read_at);

CREATE TABLE IF NOT EXISTS public_links (
    token TEXT PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS public_links_job_id_idx ON public_links (job_id);
"""


def ensure_schema(conn: _Connection) -> None:
    jobs_store.ensure_schema(conn)  # job_ownership/public_links ссылаются на jobs
    with conn.cursor() as cur:
        cur.execute(AUTH_SCHEMA_SQL)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    salt_hex, _, digest_hex = password_hash.partition("$")
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(digest_hex)
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return secrets.compare_digest(actual, expected)


@dataclass(frozen=True)
class User:
    id: uuid.UUID
    email: str
    role: str
    closed_contour_access: bool
    created_at: datetime


class EmailAlreadyRegisteredError(ValueError):
    pass


def create_user(
    conn: _Connection, *, email: str, password: str,
    role: str = ROLE_VIEWER, closed_contour_access: bool = False,
) -> User:
    if role not in ROLES:
        raise ValueError(f"неизвестная роль {role!r}, ожидается одна из {ROLES}")
    user_id = uuid.uuid4()
    password_hash = hash_password(password)
    with conn.cursor() as cur:
        try:
            cur.execute(
                "INSERT INTO users (id, email, password_hash, role, closed_contour_access) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING created_at",
                (str(user_id), email, password_hash, role, closed_contour_access),
            )
        except psycopg.errors.UniqueViolation as exc:
            raise EmailAlreadyRegisteredError(f"email {email!r} уже зарегистрирован") from exc
        (created_at,) = cur.fetchone()
    return User(id=user_id, email=email, role=role, closed_contour_access=closed_contour_access, created_at=created_at)


def _user_from_row(row) -> User:
    return User(id=row[0], email=row[1], role=row[2], closed_contour_access=row[3], created_at=row[4])


def get_user_by_id(conn: _Connection, user_id: uuid.UUID | str) -> User | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, role, closed_contour_access, created_at FROM users WHERE id = %s",
            (str(user_id),),
        )
        row = cur.fetchone()
    return None if row is None else _user_from_row(row)


def authenticate(conn: _Connection, email: str, password: str) -> User | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, password_hash, role, closed_contour_access, created_at "
            "FROM users WHERE email = %s",
            (email,),
        )
        row = cur.fetchone()
    if row is None or not verify_password(password, row[2]):
        return None
    return User(id=row[0], email=row[1], role=row[3], closed_contour_access=row[4], created_at=row[5])


def create_session(conn: _Connection, user_id: uuid.UUID | str, *, ttl_hours: int = SESSION_TTL_HOURS) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (%s, %s, %s)",
            (token, str(user_id), expires_at),
        )
    return token


def get_user_by_token(conn: _Connection, token: str) -> User | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT u.id, u.email, u.role, u.closed_contour_access, u.created_at "
            "FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token = %s AND s.expires_at > now()",
            (token,),
        )
        row = cur.fetchone()
    return None if row is None else _user_from_row(row)


def revoke_session(conn: _Connection, token: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM sessions WHERE token = %s", (token,))


def set_user_role(conn: _Connection, user_id: uuid.UUID | str, *, role: str, closed_contour_access: bool) -> None:
    if role not in ROLES:
        raise ValueError(f"неизвестная роль {role!r}, ожидается одна из {ROLES}")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE users SET role = %s, closed_contour_access = %s WHERE id = %s",
            (role, closed_contour_access, str(user_id)),
        )


def record_job_ownership(conn: _Connection, job_id: uuid.UUID | str, owner_id: uuid.UUID | str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO job_ownership (job_id, owner_id) VALUES (%s, %s) ON CONFLICT (job_id) DO NOTHING",
            (str(job_id), str(owner_id)),
        )


def get_job_owner(conn: _Connection, job_id: uuid.UUID | str) -> uuid.UUID | None:
    with conn.cursor() as cur:
        cur.execute("SELECT owner_id FROM job_ownership WHERE job_id = %s", (str(job_id),))
        row = cur.fetchone()
    return None if row is None else row[0]


def list_job_ids_for_user(conn: _Connection, owner_id: uuid.UUID | str) -> list[uuid.UUID]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT job_id FROM job_ownership WHERE owner_id = %s ORDER BY created_at DESC",
            (str(owner_id),),
        )
        return [row[0] for row in cur.fetchall()]


@dataclass(frozen=True)
class Notification:
    id: int
    message: str
    job_id: uuid.UUID | None
    created_at: datetime
    read_at: datetime | None


def create_notification(
    conn: _Connection, *, user_id: uuid.UUID | str, message: str, job_id: uuid.UUID | str | None = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO notifications (user_id, job_id, message) VALUES (%s, %s, %s) RETURNING id",
            (str(user_id), str(job_id) if job_id is not None else None, message),
        )
        (notification_id,) = cur.fetchone()
    return notification_id


def list_notifications(conn: _Connection, user_id: uuid.UUID | str, *, unread_only: bool = False) -> list[Notification]:
    query = "SELECT id, message, job_id, created_at, read_at FROM notifications WHERE user_id = %s"
    if unread_only:
        query += " AND read_at IS NULL"
    query += " ORDER BY created_at DESC"
    with conn.cursor() as cur:
        cur.execute(query, (str(user_id),))
        rows = cur.fetchall()
    return [Notification(id=r[0], message=r[1], job_id=r[2], created_at=r[3], read_at=r[4]) for r in rows]


def mark_notification_read(conn: _Connection, notification_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE notifications SET read_at = now() WHERE id = %s AND read_at IS NULL", (notification_id,))


def create_public_link(conn: _Connection, job_id: uuid.UUID | str) -> str:
    """Публичная ссылка на просмотр модели (п. 4). Честно: остальные
    эндпоинты задачи (`GET /jobs/{id}`, скачивание файлов) и без того не
    требуют авторизации (в проекте ещё нет ролевого разграничения на уровне
    существующих эндпоинтов, см. `docs/rendering.md`/`docs/api.md`) — токен
    здесь даёт стабильную, отзываемую ссылку для вьюера, а не новую границу
    доступа поверх уже открытых данных."""
    token = secrets.token_urlsafe(16)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO public_links (token, job_id) VALUES (%s, %s)", (token, str(job_id)))
    return token


def resolve_public_link(conn: _Connection, token: str) -> uuid.UUID | None:
    with conn.cursor() as cur:
        cur.execute("SELECT job_id FROM public_links WHERE token = %s AND revoked_at IS NULL", (token,))
        row = cur.fetchone()
    return None if row is None else row[0]


def revoke_public_link(conn: _Connection, token: str) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE public_links SET revoked_at = now() WHERE token = %s AND revoked_at IS NULL", (token,))
