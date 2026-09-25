"""Обёртка Celery над исполнением пайплайна задачи (Шаг 1.3).

Каждый вызов `run_job_step_task` выполняет РОВНО один следующий шаг задачи
(`jobs.pipeline.run_next_step`) и, если задача ещё не done/failed, сам ставит
себя в очередь ещё раз — так падение одного шага не требует пересчёта
предыдущих: после `store.retry_failed_step` следующий `enqueue_job` продолжит
с него же (см. `geo/tests/test_tasks_celery.py`).
"""

from __future__ import annotations

import os

import psycopg

from topology_geo.devcheck import load_environment_config
from topology_geo.jobs import store
from topology_geo.jobs.pipeline import run_next_step
from topology_geo.jobs.steps import DEFAULT_PIPELINE
from topology_geo.storage import FileSystemObjectStorage, InMemoryObjectStorage, ObjectStorage
from topology_geo.tasks.celery_app import app

_storage_singleton: ObjectStorage | None = None


def _connect():
    config = load_environment_config()
    return psycopg.connect(config.postgres.dsn, autocommit=True)


def get_storage() -> ObjectStorage:
    """Хранилище результатов задач.

    Настоящего MinIO в этой среде разработки нет (нет демона Docker) — если
    задан `TOPOLOGY_STORAGE_ROOT`, используется локальный каталог
    (`FileSystemObjectStorage`, работает и из отдельного процесса воркера);
    иначе — хранилище в памяти (годится только внутри одного процесса, для
    `task_always_eager`-тестов API). В проде подставляется `MinioObjectStorage`.
    """
    global _storage_singleton
    if _storage_singleton is None:
        root = os.environ.get("TOPOLOGY_STORAGE_ROOT")
        _storage_singleton = FileSystemObjectStorage(root) if root else InMemoryObjectStorage()
    return _storage_singleton


@app.task(name="topology.run_job_step")
def run_job_step_task(job_id: str) -> str | None:
    conn = _connect()
    try:
        storage = get_storage()
        executed = run_next_step(conn, storage, job_id, DEFAULT_PIPELINE)
        job = store.get_job(conn, job_id)
    finally:
        conn.close()

    if executed is not None and job is not None and job.status == store.STATUS_RUNNING:
        run_job_step_task.delay(job_id)
    return executed


def enqueue_job(job_id: str) -> None:
    """Поставить задачу в очередь — выполнит все её шаги по порядку
    (самоперепланирующаяся цепочка задач, см. `run_job_step_task`)."""
    run_job_step_task.delay(job_id)
