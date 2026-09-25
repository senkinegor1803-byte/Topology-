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
    checks = {
        "PostGIS": check_postgres(config.postgres),
        "Redis": check_redis(config.redis_url),
        "MinIO": check_minio(config.minio),
    }

    all_ok = True
    for name, (ok, detail) in checks.items():
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
        all_ok = all_ok and ok

    if not all_ok:
        print("Не все сервисы окружения доступны — см. вывод выше.")
        return 1

    print("Все сервисы окружения доступны. Ожидание задач (Celery — Шаг 1.3)...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
