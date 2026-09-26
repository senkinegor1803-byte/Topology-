"""Celery-приложение (Шаг 1.3, п. 2: очередь Redis + Celery).

Брокер/бэкенд читаются из окружения, совместимого с
`topology_geo.devcheck.load_environment_config` (тот же `REDIS_URL`, что и в
смоук-тесте воркера Docker Compose, Шаг 0.7). Бэкенд результатов Celery не
является источником истины для статуса задачи — им остаются таблицы
`jobs`/`job_steps` (`topology_geo.jobs.store`), переживающие перезапуск
воркера и не зависящие от деталей брокера.
"""

from __future__ import annotations

import os

from celery import Celery

from topology_geo.devcheck import load_environment_config

_config = load_environment_config()
BROKER_URL = os.environ.get("CELERY_BROKER_URL", _config.redis_url)
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", _config.redis_url)

app = Celery(
    "topology",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    # без include воркер, запущенный как `celery -A topology_geo.tasks.celery_app`,
    # никогда не импортирует pipeline_tasks.py и не увидит задачу @app.task в нём
    # (регистрация задач привязана к импорту модуля, а не к самому объекту app).
    include=["topology_geo.tasks.pipeline_tasks", "topology_geo.tasks.tile_tasks"],
)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # для тестов API: выполнить задачу синхронно в том же процессе, без брокера
    task_always_eager=os.environ.get("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true",
    task_eager_propagates=True,
)
