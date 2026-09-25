"""Исполнение пайплайна шагов задачи (Шаг 1.3).

Шаги выполняются по порядку; каждый пишет результат независимо и может быть
перезапущен отдельно без пересчёта предыдущих (`store.retry_failed_step` +
повторный вызов `run_next_step`/`run_all_pending_steps` продолжит с
незавершённого шага — критерий приёмки Шага 1.3).

Не привязано к Celery напрямую: `tasks/pipeline_tasks.py` оборачивает эти же
функции в задачи очереди, а тесты могут исполнять пайплайн синхронно, без
брокера.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Protocol

from topology_geo.jobs import store
from topology_geo.storage import ObjectStorage


class StepFunction(Protocol):
    def __call__(self, conn: Any, storage: ObjectStorage, job: store.Job) -> dict[str, Any]: ...


PipelineSteps = dict[str, Callable[[Any, ObjectStorage, store.Job], dict[str, Any]]]


def run_next_step(
    conn: Any,
    storage: ObjectStorage,
    job_id: uuid.UUID | str,
    steps: PipelineSteps,
) -> str | None:
    """Выполнить следующий незавершённый шаг задачи.

    Возвращает имя выполненного (или упавшего) шага, либо None, если
    выполнять больше нечего (все шаги done, либо задача уже failed).
    """
    pending = store.next_pending_step(conn, job_id)
    if pending is None:
        return None

    step_fn = steps[pending.step_name]
    job = store.get_job(conn, job_id)
    store.start_step(conn, job_id, pending.step_name)
    try:
        result = step_fn(conn, storage, job)
    except Exception as exc:  # noqa: BLE001 - любая ошибка шага фиксируется с понятным сообщением
        store.fail_step(conn, job_id, pending.step_name, str(exc))
        return pending.step_name

    store.finish_step(conn, job_id, pending.step_name, result)
    return pending.step_name


def run_all_pending_steps(
    conn: Any,
    storage: ObjectStorage,
    job_id: uuid.UUID | str,
    steps: PipelineSteps,
) -> None:
    """Выполнить все оставшиеся шаги подряд, пока задача не станет done/failed."""
    while True:
        job = store.get_job(conn, job_id)
        if job is None or job.status in (store.STATUS_DONE, store.STATUS_FAILED):
            return
        executed = run_next_step(conn, storage, job_id, steps)
        if executed is None:
            return
