"""Тесты модели задач (Шаг 1.3, п. 1). Реальный Postgres — фикстура `pg_test_db`
из conftest.py; пропускается, если БД недоступна."""

from __future__ import annotations

import pytest

from topology_geo.jobs import store

STEP_NAMES = ["select_osm", "prepare_relief"]


@pytest.fixture()
def db(pg_test_db):
    store.ensure_schema(pg_test_db)
    return pg_test_db


def test_create_job_starts_pending_with_all_steps(db):
    job = store.create_job(
        db, center_lon=56.24, center_lat=58.01, radius_m=500, layers=["buildings", "roads"],
        detail="LOD1", step_names=STEP_NAMES,
    )
    assert job.status == store.STATUS_PENDING
    assert [s.step_name for s in job.steps] == STEP_NAMES
    assert all(s.status == store.STATUS_PENDING for s in job.steps)
    assert job.layers == ["buildings", "roads"]


def test_get_job_returns_none_for_unknown_id(db):
    import uuid

    assert store.get_job(db, uuid.uuid4()) is None


def test_next_pending_step_follows_order(db):
    job = store.create_job(
        db, center_lon=56.24, center_lat=58.01, radius_m=500, layers=[], detail="LOD1",
        step_names=STEP_NAMES,
    )
    first = store.next_pending_step(db, job.id)
    assert first.step_name == "select_osm"

    store.start_step(db, job.id, "select_osm")
    store.finish_step(db, job.id, "select_osm", {"ok": True})

    second = store.next_pending_step(db, job.id)
    assert second.step_name == "prepare_relief"


def test_job_status_running_after_first_step_starts(db):
    job = store.create_job(
        db, center_lon=56.24, center_lat=58.01, radius_m=500, layers=[], detail="LOD1",
        step_names=STEP_NAMES,
    )
    store.start_step(db, job.id, "select_osm")
    assert store.get_job(db, job.id).status == store.STATUS_RUNNING


def test_job_status_done_after_all_steps_finish(db):
    job = store.create_job(
        db, center_lon=56.24, center_lat=58.01, radius_m=500, layers=[], detail="LOD1",
        step_names=STEP_NAMES,
    )
    for name in STEP_NAMES:
        store.start_step(db, job.id, name)
        store.finish_step(db, job.id, name, {"n": name})

    job = store.get_job(db, job.id)
    assert job.status == store.STATUS_DONE
    assert all(s.status == store.STATUS_DONE for s in job.steps)
    assert job.steps[0].result == {"n": "select_osm"}


def test_job_status_failed_after_step_fails(db):
    job = store.create_job(
        db, center_lon=56.24, center_lat=58.01, radius_m=500, layers=[], detail="LOD1",
        step_names=STEP_NAMES,
    )
    store.start_step(db, job.id, "select_osm")
    store.fail_step(db, job.id, "select_osm", "нет данных OSM для этой области")

    job = store.get_job(db, job.id)
    assert job.status == store.STATUS_FAILED
    assert job.error_message == "нет данных OSM для этой области"
    assert job.steps[0].status == store.STATUS_FAILED
    assert job.steps[1].status == store.STATUS_PENDING  # второй шаг не запускался


def test_next_pending_step_is_none_for_failed_job(db):
    job = store.create_job(
        db, center_lon=56.24, center_lat=58.01, radius_m=500, layers=[], detail="LOD1",
        step_names=STEP_NAMES,
    )
    store.start_step(db, job.id, "select_osm")
    store.fail_step(db, job.id, "select_osm", "бум")
    assert store.next_pending_step(db, job.id) is None


def test_retry_failed_step_resets_only_that_step(db):
    job = store.create_job(
        db, center_lon=56.24, center_lat=58.01, radius_m=500, layers=[], detail="LOD1",
        step_names=STEP_NAMES,
    )
    store.start_step(db, job.id, "select_osm")
    store.finish_step(db, job.id, "select_osm", {"done": True})
    store.start_step(db, job.id, "prepare_relief")
    store.fail_step(db, job.id, "prepare_relief", "нет данных рельефа в этой области")

    retried = store.retry_failed_step(db, job.id)
    assert retried.step_name == "prepare_relief"

    job = store.get_job(db, job.id)
    assert job.status == store.STATUS_RUNNING  # первый шаг done -> статус не pending
    assert job.steps[0].status == store.STATUS_DONE
    assert job.steps[0].result == {"done": True}  # не пересчитан
    assert job.steps[1].status == store.STATUS_PENDING
    assert job.steps[1].error_message is None


def test_retry_failed_step_raises_when_nothing_failed(db):
    job = store.create_job(
        db, center_lon=56.24, center_lat=58.01, radius_m=500, layers=[], detail="LOD1",
        step_names=STEP_NAMES,
    )
    with pytest.raises(ValueError):
        store.retry_failed_step(db, job.id)


def test_retry_failed_step_raises_for_unknown_job(db):
    import uuid

    with pytest.raises(LookupError):
        store.retry_failed_step(db, uuid.uuid4())
