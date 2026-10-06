"""Точка входа контейнера воркера (Шаг 0.7).

Проверяет связь со всеми сервисами Compose-стека (PostGIS, Redis, MinIO) и
остаётся в ожидании — реальная обработка задач (Celery) появится в Шаге 1.3.
Пока это только смоук-тест «docker compose up поднимает все сервисы и воркер
их видит».
"""

from __future__ import annotations

import sys
import time

from topology_geo.devcheck import check_minio, check_postgres, check_redis, load_environment_config


def main() -> int:
    config = load_environment_config()
    critical_checks = {
        "PostGIS": check_postgres(config.postgres),
        "Redis": check_redis(config.redis_url),
    }
    optional_checks = {
        "MinIO": check_minio(config.minio),
    }

    all_critical_ok = True
    for name, (ok, detail) in critical_checks.items():
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
        all_critical_ok = all_critical_ok and ok

    for name, (ok, detail) in optional_checks.items():
        status = "OK" if ok else "WARN"
        print(f"[{status}] {name}: {detail}")

    if not all_critical_ok:
        print("Критичные сервисы недоступны — см. вывод выше.")
        return 1

    print("Критичные сервисы доступны. Ожидание задач (Celery — Шаг 1.3)...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
