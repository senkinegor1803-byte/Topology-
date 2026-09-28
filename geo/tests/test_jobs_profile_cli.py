"""Тесты Шага 2.13, п. 1: профилирование пайплайна по времени/памяти
(`jobs.profile_cli`) — часть без БД/реального пайплайна (замер RSS,
парсинг CSV, сопоставление окон, сборка отчёта на синтетических, но
структурно настоящих `store.Job`/`store.JobStep`). Сквозной прогон на
реальном пайплайне — `test_jobs_profile_cli_integration.py`."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from topology_geo.jobs import store
from topology_geo.jobs.profile_cli import (
    _load_memory_samples,
    _peak_rss_in_window,
    _read_rss_kb,
    _MemorySampler,
    build_report,
)


def test_read_rss_kb_returns_positive_value_for_current_process():
    rss = _read_rss_kb()
    assert rss is not None
    assert rss > 0


def test_memory_sampler_writes_csv_that_survives_stop(tmp_path):
    log_path = tmp_path / "mem.csv"
    sampler = _MemorySampler(log_path, interval_s=0.02)
    sampler.start()
    import time

    time.sleep(0.15)
    sampler.stop()

    assert log_path.is_file()
    lines = log_path.read_text().strip().splitlines()
    assert lines[0] == "timestamp_unix,rss_kb"
    assert len(lines) >= 2
    ts0, rss0 = lines[1].split(",")
    assert float(ts0) > 0
    assert int(rss0) > 0


def test_load_memory_samples_parses_csv(tmp_path):
    log_path = tmp_path / "mem.csv"
    log_path.write_text("timestamp_unix,rss_kb\n1000.0,50000\n1000.5,52000\n1001.0,51000\n")

    samples = _load_memory_samples(log_path)

    assert samples == [(1000.0, 50000), (1000.5, 52000), (1001.0, 51000)]


def test_load_memory_samples_missing_file_returns_empty():
    assert _load_memory_samples("/tmp/does-not-exist-topology-profile.csv") == []


def test_peak_rss_in_window_finds_max_within_bounds():
    samples = [(0.0, 100), (1.0, 500), (2.0, 300), (3.0, 900), (4.0, 200)]

    assert _peak_rss_in_window(samples, 1.0, 3.0) == 900
    assert _peak_rss_in_window(samples, 5.0, 6.0) is None
    assert _peak_rss_in_window(samples, 0.0, 0.0) == 100


def _step(name: str, order: int, status: str, started_at, finished_at) -> store.JobStep:
    return store.JobStep(step_name=name, step_order=order, status=status, started_at=started_at, finished_at=finished_at)


def test_build_report_computes_duration_and_peak_rss_per_step():
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    job = store.Job(
        id=uuid.uuid4(), center_lon=56.24, center_lat=58.01, radius_m=500.0, layers=[], detail="LOD1",
        status=store.STATUS_DONE, created_at=t0, updated_at=t0,
        steps=[
            _step("select_osm", 0, store.STATUS_DONE, t0, t0 + timedelta(seconds=2)),
            _step("assemble_ifc", 1, store.STATUS_DONE, t0 + timedelta(seconds=2), t0 + timedelta(seconds=12)),
        ],
    )
    memory_samples = [
        (t0.timestamp() + 0.5, 100_000),
        (t0.timestamp() + 1.5, 120_000),
        (t0.timestamp() + 5.0, 900_000),  # пик внутри окна assemble_ifc
        (t0.timestamp() + 11.0, 700_000),
    ]

    report = build_report(job, memory_samples)

    assert report["job_id"] == str(job.id)
    assert report["radius_m"] == 500.0
    steps_by_name = {s["step_name"]: s for s in report["steps"]}
    assert steps_by_name["select_osm"]["duration_s"] == 2.0
    assert steps_by_name["select_osm"]["peak_rss_kb"] == 120_000
    assert steps_by_name["assemble_ifc"]["duration_s"] == 10.0
    assert steps_by_name["assemble_ifc"]["peak_rss_kb"] == 900_000
    assert report["overall_duration_s"] == 12.0
    assert report["overall_peak_rss_kb"] == 900_000


def test_build_report_handles_unstarted_and_crashed_steps():
    """Шаг без `started_at` (ещё не выполнялся) не даёт длительности; шаг с
    `started_at` без `finished_at` (обрыв на нём - реальная картина при
    OOM-killer, СМ. docstring `profile_cli.py`) получает длительность ДО
    текущего момента, не падает."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    job = store.Job(
        id=uuid.uuid4(), center_lon=56.24, center_lat=58.01, radius_m=500.0, layers=[], detail="LOD1",
        status=store.STATUS_RUNNING, created_at=t0, updated_at=t0,
        steps=[
            _step("select_osm", 0, store.STATUS_DONE, t0, t0 + timedelta(seconds=1)),
            _step("assemble_ifc", 1, store.STATUS_RUNNING, t0 + timedelta(seconds=1), None),
            _step("convert_to_glb", 2, store.STATUS_PENDING, None, None),
        ],
    )

    report = build_report(job, memory_samples=[])

    steps_by_name = {s["step_name"]: s for s in report["steps"]}
    assert steps_by_name["convert_to_glb"]["duration_s"] is None
    assert steps_by_name["assemble_ifc"]["duration_s"] is not None
    assert steps_by_name["assemble_ifc"]["duration_s"] >= 0
