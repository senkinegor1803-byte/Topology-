"""Общие фикстуры для тестов, которым нужен реальный Postgres+PostGIS.

Используется тестами Шага 1.3 (jobs/tasks/api). Модули Шагов 1.1/1.2 уже
заводят аналогичную логику самостоятельно (задним числом не трогаем — не
хотим рисковать уже стабильными тестами ради DRY).
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

PG_CONN_PARAMS = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": os.environ.get("POSTGRES_PORT", "5432"),
    "user": os.environ.get("POSTGRES_USER", "topology"),
    "password": os.environ.get("POSTGRES_PASSWORD", "topology"),
}


def postgres_reachable() -> bool:
    try:
        conn = psycopg.connect(dbname="postgres", connect_timeout=3, **PG_CONN_PARAMS)
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture()
def pg_test_db():
    """Свежая одноразовая база с PostGIS; пропускает тест, если Postgres недоступен."""
    if not postgres_reachable():
        pytest.skip("требуется доступный Postgres+PostGIS")

    db_name = f"topology_test_{uuid.uuid4().hex[:8]}"
    admin_conn = psycopg.connect(dbname="postgres", autocommit=True, **PG_CONN_PARAMS)
    try:
        admin_conn.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        admin_conn.close()

    conn = psycopg.connect(dbname=db_name, autocommit=True, **PG_CONN_PARAMS)
    conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    try:
        yield conn
    finally:
        conn.close()
        admin_conn = psycopg.connect(dbname="postgres", autocommit=True, **PG_CONN_PARAMS)
        try:
            admin_conn.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            admin_conn.close()
