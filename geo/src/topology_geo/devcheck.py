"""Проверка готовности сервисов окружения разработки (Шаг 0.7).

Используется точкой входа Docker-воркера (`geo/docker/entrypoint.py`) как
смоук-тест «docker compose up поднимает все сервисы, и воркер их видит».
Формирование параметров подключения (чистая логика) отделено от самих сетевых
вызовов, чтобы быть тестируемым без реального Postgres/Redis/MinIO —
подключения к реальным сервисам в этой изолированной среде разработки не
проверялись (здесь нет демона Docker), тестами покрыта только сама логика.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Callable, Mapping


@dataclass(frozen=True)
class PostgresConfig:
    host: str
    port: int
    db: str
    user: str
    password: str

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class MinioConfig:
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool = False


@dataclass(frozen=True)
class EnvironmentConfig:
    postgres: PostgresConfig
    redis_url: str
    minio: MinioConfig


def load_environment_config(env: Mapping[str, str] | None = None) -> EnvironmentConfig:
    """Собрать конфигурацию подключений из переменных окружения (с дефолтами
    под `infra/docker-compose.yml`)."""
    env = env if env is not None else os.environ
    postgres = PostgresConfig(
        host=env.get("POSTGRES_HOST", "postgis"),
        port=int(env.get("POSTGRES_PORT", "5432")),
        db=env.get("POSTGRES_DB", "topology"),
        user=env.get("POSTGRES_USER", "topology"),
        password=env.get("POSTGRES_PASSWORD", "topology"),
    )
    redis_url = env.get("REDIS_URL", "redis://redis:6379/0")
    minio = MinioConfig(
        endpoint=env.get("MINIO_ENDPOINT", "minio:9000"),
        access_key=env.get("MINIO_ACCESS_KEY", "topology"),
        secret_key=env.get("MINIO_SECRET_KEY", "topology12345"),
        secure=env.get("MINIO_SECURE", "false").lower() == "true",
    )
    return EnvironmentConfig(postgres=postgres, redis_url=redis_url, minio=minio)


CheckResult = tuple[bool, str]


def check_postgres(config: PostgresConfig, *, connector: Callable | None = None) -> CheckResult:
    """`connector` — psycopg.connect-совместимая функция; внедряется в тестах."""
    connector = connector or importlib.import_module("psycopg").connect
    try:
        conn = connector(config.dsn, connect_timeout=5)
        conn.close()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - смоук-тест: любая ошибка -> статус FAIL
        return False, str(exc)


def check_redis(redis_url: str, *, client_factory: Callable | None = None) -> CheckResult:
    if client_factory is None:
        client_factory = importlib.import_module("redis").Redis.from_url
    try:
        client = client_factory(redis_url, socket_connect_timeout=5)
        client.ping()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def check_minio(config: MinioConfig, *, client_factory: Callable | None = None) -> CheckResult:
    if client_factory is None:
        client_factory = importlib.import_module("minio").Minio
    try:
        client = client_factory(
            config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure,
        )
        client.list_buckets()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
