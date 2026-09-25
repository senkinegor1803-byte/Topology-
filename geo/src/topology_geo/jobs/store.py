"""Хранение задач и их шагов (Шаг 1.3, п. 1: `GET /jobs/{id}` — статус,
прогресс, журнал).

Источник истины для состояния конвейера — эти таблицы, не результат-бэкенд
Celery: так статус задачи не зависит от деталей брокера очереди и переживает
падение/перезапуск воркера.

`JOBS_SCHEMA_SQL` — встроенная копия `geo/sql/003_jobs.sql` (тот же приём,
что и `osm.import_log`/`relief.coverage`; синхронность проверена тестом
`geo/tests/test_jobs_sql_migrations.py`).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

_TERMINAL_JOB_STATUSES = (STATUS_DONE, STATUS_FAILED)


class _Connection(Protocol):
    def cursor(self) -> Any: ...


JOBS_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    center_lon DOUBLE PRECISION NOT NULL,
    center_lat DOUBLE PRECISION NOT NULL,
    radius_m DOUBLE PRECISION NOT NULL,
    layers JSONB NOT NULL,
    detail TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS job_steps (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    step_name TEXT NOT NULL,
    step_order INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    error_message TEXT,
    result JSONB,
    UNIQUE (job_id, step_name)
);

CREATE INDEX IF NOT EXISTS job_steps_job_id_idx ON job_steps (job_id, step_order);
"""


@dataclass(frozen=True)
class JobStep:
    step_name: str
    step_order: int
    status: str = STATUS_PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None
    result: dict[str, Any] | None = None


@dataclass(frozen=True)
class Job:
    id: uuid.UUID
    center_lon: float
    center_lat: float
    radius_m: float
    layers: list[str]
    detail: str
    status: str
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
    steps: list[JobStep] = field(default_factory=list)


def ensure_schema(conn: _Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(JOBS_SCHEMA_SQL)


def _derive_job_status(step_statuses: list[str]) -> str:
    """Статус задачи из статусов её шагов, выполняемых по порядку.

    failed, если упал хоть один шаг; done, если все шаги готовы; running, если
    хоть один шаг стартовал (running/done/failed вперемешку с pending);
    иначе pending (ни один шаг ещё не начат).
    """
    if any(s == STATUS_FAILED for s in step_statuses):
        return STATUS_FAILED
    if step_statuses and all(s == STATUS_DONE for s in step_statuses):
        return STATUS_DONE
    if any(s != STATUS_PENDING for s in step_statuses):
        return STATUS_RUNNING
    return STATUS_PENDING


def create_job(
    conn: _Connection,
    *,
    center_lon: float,
    center_lat: float,
    radius_m: float,
    layers: list[str],
    detail: str,
    step_names: list[str],
) -> Job:
    import json

    job_id = uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs (id, center_lon, center_lat, radius_m, layers, detail, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING created_at, updated_at",
            (str(job_id), center_lon, center_lat, radius_m, json.dumps(layers), detail, STATUS_PENDING),
        )
        created_at, updated_at = cur.fetchone()
        for order, name in enumerate(step_names):
            cur.execute(
                "INSERT INTO job_steps (job_id, step_name, step_order, status) VALUES (%s, %s, %s, %s)",
                (str(job_id), name, order, STATUS_PENDING),
            )

    return get_job(conn, job_id)  # type: ignore[return-value]


def get_job(conn: _Connection, job_id: uuid.UUID | str) -> Job | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, center_lon, center_lat, radius_m, layers, detail, status, error_message, "
            "created_at, updated_at FROM jobs WHERE id = %s",
            (str(job_id),),
        )
        row = cur.fetchone()
        if row is None:
            return None

        cur.execute(
            "SELECT step_name, step_order, status, started_at, finished_at, error_message, result "
            "FROM job_steps WHERE job_id = %s ORDER BY step_order",
            (str(job_id),),
        )
        step_rows = cur.fetchall()

    steps = [
        JobStep(
            step_name=r[0], step_order=r[1], status=r[2], started_at=r[3],
            finished_at=r[4], error_message=r[5], result=r[6],
        )
        for r in step_rows
    ]
    return Job(
        id=row[0], center_lon=row[1], center_lat=row[2], radius_m=row[3], layers=row[4],
        detail=row[5], status=row[6], error_message=row[7], created_at=row[8], updated_at=row[9],
        steps=steps,
    )


def next_pending_step(conn: _Connection, job_id: uuid.UUID | str) -> JobStep | None:
    """Первый ещё не завершённый шаг задачи (по порядку), либо None."""
    job = get_job(conn, job_id)
    if job is None or job.status in _TERMINAL_JOB_STATUSES:
        return None
    for step in job.steps:
        if step.status != STATUS_DONE:
            return step
    return None


def _recompute_job_status(conn: _Connection, job_id: uuid.UUID | str) -> None:
    job = get_job(conn, job_id)
    if job is None:
        return
    status = _derive_job_status([s.status for s in job.steps])
    error_message = next((s.error_message for s in job.steps if s.status == STATUS_FAILED), None)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status = %s, error_message = %s, updated_at = now() WHERE id = %s",
            (status, error_message, str(job_id)),
        )


def start_step(conn: _Connection, job_id: uuid.UUID | str, step_name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE job_steps SET status = %s, started_at = now(), error_message = NULL "
            "WHERE job_id = %s AND step_name = %s",
            (STATUS_RUNNING, str(job_id), step_name),
        )
    _recompute_job_status(conn, job_id)


def finish_step(conn: _Connection, job_id: uuid.UUID | str, step_name: str, result: dict[str, Any]) -> None:
    import json

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE job_steps SET status = %s, finished_at = now(), result = %s "
            "WHERE job_id = %s AND step_name = %s",
            (STATUS_DONE, json.dumps(result), str(job_id), step_name),
        )
    _recompute_job_status(conn, job_id)


def fail_step(conn: _Connection, job_id: uuid.UUID | str, step_name: str, error_message: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE job_steps SET status = %s, finished_at = now(), error_message = %s "
            "WHERE job_id = %s AND step_name = %s",
            (STATUS_FAILED, error_message, str(job_id), step_name),
        )
    _recompute_job_status(conn, job_id)


def retry_failed_step(conn: _Connection, job_id: uuid.UUID | str) -> JobStep:
    """Сбросить упавший шаг в pending, не трогая уже готовые (done) шаги —
    следующий прогон конвейера (`jobs.pipeline.run_next_step`) продолжит с
    него же, не пересчитывая предыдущие (критерий приёмки Шага 1.3)."""
    job = get_job(conn, job_id)
    if job is None:
        raise LookupError(f"задача {job_id} не найдена")
    failed = next((s for s in job.steps if s.status == STATUS_FAILED), None)
    if failed is None:
        raise ValueError(f"у задачи {job_id} нет упавшего шага для перезапуска")

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE job_steps SET status = %s, started_at = NULL, finished_at = NULL, error_message = NULL "
            "WHERE job_id = %s AND step_name = %s",
            (STATUS_PENDING, str(job_id), failed.step_name),
        )
    _recompute_job_status(conn, job_id)
    return JobStep(step_name=failed.step_name, step_order=failed.step_order, status=STATUS_PENDING)
