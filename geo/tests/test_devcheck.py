"""Тесты логики проверки окружения (Шаг 0.7). Без реальных сервисов: подключения
подменяются фейковыми фабриками — сам Docker Compose стек здесь не поднимался
(в этой среде разработки нет демона Docker)."""

from __future__ import annotations


from topology_geo.devcheck import (
    check_minio,
    check_postgres,
    check_redis,
    load_environment_config,
)


def test_load_environment_config_defaults():
    config = load_environment_config(env={})
    assert config.postgres.host == "postgis"
    assert config.postgres.port == 5432
    assert config.postgres.dsn == "postgresql://topology:topology@postgis:5432/topology"
    assert config.redis_url == "redis://redis:6379/0"
    assert config.minio.endpoint == "minio:9000"
    assert config.minio.secure is False


def test_load_environment_config_overrides():
    env = {
        "POSTGRES_HOST": "db.internal",
        "POSTGRES_PORT": "6543",
        "POSTGRES_DB": "topo_prod",
        "POSTGRES_USER": "svc",
        "POSTGRES_PASSWORD": "s3cr3t",
        "REDIS_URL": "redis://cache:6380/1",
        "MINIO_ENDPOINT": "s3.internal:9000",
        "MINIO_SECURE": "true",
    }
    config = load_environment_config(env=env)
    assert config.postgres.dsn == "postgresql://svc:s3cr3t@db.internal:6543/topo_prod"
    assert config.redis_url == "redis://cache:6380/1"
    assert config.minio.endpoint == "s3.internal:9000"
    assert config.minio.secure is True


def test_check_postgres_success():
    calls = {}

    class FakeConn:
        def close(self):
            calls["closed"] = True

    def fake_connector(dsn, connect_timeout):
        calls["dsn"] = dsn
        return FakeConn()

    config = load_environment_config(env={}).postgres
    ok, detail = check_postgres(config, connector=fake_connector)
    assert ok is True
    assert detail == "ok"
    assert calls["closed"] is True
    assert calls["dsn"] == config.dsn


def test_check_postgres_failure():
    def failing_connector(*args, **kwargs):
        raise RuntimeError("connection refused")

    config = load_environment_config(env={}).postgres
    ok, detail = check_postgres(config, connector=failing_connector)
    assert ok is False
    assert "connection refused" in detail


def test_check_redis_success_and_failure():
    class FakeClient:
        def ping(self):
            return True

    ok, detail = check_redis("redis://redis:6379/0", client_factory=lambda url, **kw: FakeClient())
    assert ok is True and detail == "ok"

    def failing_factory(url, **kwargs):
        raise ConnectionError("no route to host")

    ok, detail = check_redis("redis://redis:6379/0", client_factory=failing_factory)
    assert ok is False
    assert "no route to host" in detail


def test_check_minio_success_and_failure():
    class FakeMinioClient:
        def list_buckets(self):
            return []

    config = load_environment_config(env={}).minio
    ok, detail = check_minio(config, client_factory=lambda *a, **kw: FakeMinioClient())
    assert ok is True and detail == "ok"

    def failing_factory(*a, **kw):
        raise TimeoutError("timed out")

    ok, detail = check_minio(config, client_factory=failing_factory)
    assert ok is False
    assert "timed out" in detail
