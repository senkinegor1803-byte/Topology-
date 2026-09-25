"""Тесты движка исполнения пайплайна (Шаг 1.3). Шаговые функции — фейковые
(со счётчиком вызовов), чтобы точно проверить два критерия приёмки: упавший
шаг перезапускается БЕЗ пересчёта предыдущих, и параллельные задачи не мешают
друг другу. Реальные шаги (`jobs.steps.select_osm`/`prepare_relief`) проверены
отдельно в test_jobs_steps.py."""

from __future__ import annotations

import threading

import pytest

from topology_geo.jobs import store
from topology_geo.jobs.pipeline import run_all_pending_steps, run_next_step
from topology_geo.storage import InMemoryObjectStorage

STEP_NAMES = ["step_a", "step_b", "step_c"]


@pytest.fixture()
def db(pg_test_db):
    store.ensure_schema(pg_test_db)
    return pg_test_db


class CountingStep:
    def __init__(self, name: str, *, fail_times: int = 0):
        self.name = name
        self.calls = 0
        self.fail_times = fail_times

    def __call__(self, conn, storage, job):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"{self.name} временно недоступен (попытка {self.calls})")
        return {"step": self.name, "call": self.calls, "job_id": str(job.id)}


def test_run_all_pending_steps_executes_in_order(db):
    job = store.create_job(
        db, center_lon=56.2, center_lat=58.0, radius_m=500, layers=[], detail="LOD1", step_names=STEP_NAMES
    )
    order: list[str] = []

    def make_step(name):
        def step(conn, storage, job):
            order.append(name)
            return {}
        return step

    steps = {name: make_step(name) for name in STEP_NAMES}
    run_all_pending_steps(db, InMemoryObjectStorage(), job.id, steps)

    assert order == STEP_NAMES
    assert store.get_job(db, job.id).status == store.STATUS_DONE


def test_failed_step_stops_pipeline_without_running_later_steps(db):
    job = store.create_job(
        db, center_lon=56.2, center_lat=58.0, radius_m=500, layers=[], detail="LOD1", step_names=STEP_NAMES
    )
    a, b, c = CountingStep("a"), CountingStep("b", fail_times=99), CountingStep("c")
    steps = {"step_a": a, "step_b": b, "step_c": c}

    run_all_pending_steps(db, InMemoryObjectStorage(), job.id, steps)

    assert a.calls == 1
    assert b.calls == 1
    assert c.calls == 0  # шаг после упавшего не запускался
    job_state = store.get_job(db, job.id)
    assert job_state.status == store.STATUS_FAILED
    assert "b временно недоступен" in job_state.error_message


def test_retry_after_failure_does_not_recompute_earlier_steps(db):
    """Регрессия критерия Шага 1.3: «упавший шаг перезапускается без
    пересчёта предыдущих»."""
    job = store.create_job(
        db, center_lon=56.2, center_lat=58.0, radius_m=500, layers=[], detail="LOD1", step_names=STEP_NAMES
    )
    a, b, c = CountingStep("a"), CountingStep("b", fail_times=1), CountingStep("c")
    steps = {"step_a": a, "step_b": b, "step_c": c}

    run_all_pending_steps(db, InMemoryObjectStorage(), job.id, steps)
    assert a.calls == 1
    assert b.calls == 1
    assert store.get_job(db, job.id).status == store.STATUS_FAILED

    store.retry_failed_step(db, job.id)
    run_all_pending_steps(db, InMemoryObjectStorage(), job.id, steps)

    assert a.calls == 1  # шаг a НЕ пересчитан
    assert b.calls == 2  # шаг b перезапущен и на этот раз прошёл
    assert c.calls == 1
    assert store.get_job(db, job.id).status == store.STATUS_DONE


def test_run_next_step_returns_none_when_nothing_pending(db):
    job = store.create_job(
        db, center_lon=56.2, center_lat=58.0, radius_m=500, layers=[], detail="LOD1", step_names=["only"]
    )
    steps = {"only": lambda conn, storage, job: {"ok": True}}
    assert run_next_step(db, InMemoryObjectStorage(), job.id, steps) == "only"
    assert run_next_step(db, InMemoryObjectStorage(), job.id, steps) is None


def test_parallel_jobs_do_not_interfere(pg_test_db):
    """Критерий Шага 1.3: «10 параллельных задач не мешают друг другу» —
    каждая задача пишет собственный результат под своим ключом хранилища и
    видит только свои данные, даже когда исполняются одновременно из разных
    потоков с отдельными соединениями к БД."""
    import os

    conn_params = {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
        "user": os.environ.get("POSTGRES_USER", "topology"),
        "password": os.environ.get("POSTGRES_PASSWORD", "topology"),
    }
    import psycopg

    dbname = pg_test_db.info.dbname
    store.ensure_schema(pg_test_db)

    n_jobs = 10
    job_ids = []
    for i in range(n_jobs):
        job = store.create_job(
            pg_test_db, center_lon=56.0 + i * 0.01, center_lat=58.0, radius_m=500,
            layers=[], detail="LOD1", step_names=["compute"],
        )
        job_ids.append(job.id)

    shared_storage = InMemoryObjectStorage()
    errors: list[Exception] = []

    def worker(job_id, index):
        try:
            conn = psycopg.connect(dbname=dbname, autocommit=True, **conn_params)
            try:
                def compute(conn, storage, job):
                    key = f"jobs/{job.id}/result.txt"
                    storage.upload(key, f"result-for-{index}".encode())
                    return {"index": index, "center_lon": job.center_lon, "key": key}

                run_all_pending_steps(conn, shared_storage, job_id, {"compute": compute})
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(job_id, i)) for i, job_id in enumerate(job_ids)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors

    for i, job_id in enumerate(job_ids):
        job = store.get_job(pg_test_db, job_id)
        assert job.status == store.STATUS_DONE
        result = job.steps[0].result
        assert result["index"] == i
        assert result["center_lon"] == pytest.approx(56.0 + i * 0.01)
        assert shared_storage.download(result["key"]) == f"result-for-{i}".encode()
