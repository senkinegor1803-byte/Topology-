"""Интеграционный тест реальной очереди Celery+Redis (Шаг 1.3, п. 2).

Поднимает настоящий воркер Celery в ОТДЕЛЬНОМ ПРОЦЕССЕ (реальный Redis-брокер,
не eager-режим) и прогоняет через него задачу целиком — оба реализованных
шага (select_osm, prepare_relief) с реально загруженными предварительными
данными (OSM через osm2pgsql, DEM через файловое хранилище). Доказывает, что
работает именно очередь, а не только Python-логика оркестрации (та отдельно
проверена в test_jobs_pipeline.py без брокера).

Пропускается, если недоступны Redis или системный `osm2pgsql`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rio_cogeo.cogeo import cog_validate

redis = pytest.importorskip("redis")

from topology_geo.jobs import store  # noqa: E402
from topology_geo.jobs.steps import DEFAULT_STEP_NAMES  # noqa: E402
from topology_geo.relief.coverage import CoverageEntry  # noqa: E402
from topology_geo.relief.coverage import ensure_schema as ensure_relief_schema  # noqa: E402
from topology_geo.relief.coverage import register_coverage  # noqa: E402

STYLE_LUA = Path(__file__).resolve().parents[1] / "osm2pgsql" / "style.lua"
TEST_REDIS_DB = 2
JOB_CENTER_LON, JOB_CENTER_LAT = 56.243, 58.0105

SAMPLE_OSM_XML = """\
<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="topology-test">
  <node id="1" lat="58.0100" lon="56.2420" version="1"/>
  <node id="2" lat="58.0100" lon="56.2440" version="1"/>
  <node id="3" lat="58.0110" lon="56.2440" version="1"/>
  <node id="4" lat="58.0110" lon="56.2420" version="1"/>
  <way id="100" version="1">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
    <tag k="building" v="yes"/>
  </way>
</osm>
"""


def _redis_reachable() -> bool:
    try:
        client = redis.Redis(host="localhost", port=6379, db=TEST_REDIS_DB, socket_connect_timeout=2)
        return bool(client.ping())
    except Exception:
        return False


def _osm2pgsql_available() -> bool:
    return shutil.which("osm2pgsql") is not None


pytestmark = pytest.mark.skipif(
    not (_redis_reachable() and _osm2pgsql_available()),
    reason="требуется доступный Redis и системный osm2pgsql (см. docstring модуля)",
)


def _load_sample_osm(dbname: str, osm_path: Path) -> None:
    env = {**os.environ, "PGPASSWORD": os.environ.get("POSTGRES_PASSWORD", "topology")}
    subprocess.run(
        [
            "osm2pgsql", "--output=flex", f"--style={STYLE_LUA}",
            f"--host={os.environ.get('POSTGRES_HOST', 'localhost')}",
            f"--port={os.environ.get('POSTGRES_PORT', '5432')}",
            f"--user={os.environ.get('POSTGRES_USER', 'topology')}",
            f"--database={dbname}", str(osm_path),
        ],
        check=True, capture_output=True, text=True, env=env,
    )


def _seed_dem(storage_root: Path) -> None:
    dem_path = storage_root / "tessadem.tif"
    transform = from_origin(56.0, 58.1, 0.001, 0.001)
    data = np.full((200, 200), 155.0, dtype="float32")
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=200, width=200, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


def _enqueue_with_fresh_celery_app(job_id: str, broker_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Импортировать topology_geo.tasks.* заново с нужным CELERY_BROKER_URL в
    окружении ТЕСТОВОГО процесса — если модуль уже был импортирован раньше
    (другим тестом) с другими настройками, обычный import вернул бы старый
    закэшированный объект Celery-приложения.

    Через `monkeypatch`, а не прямое присваивание `os.environ[...]` — иначе
    значения утекают в другие тесты модуля (реальный баг при первом прогоне
    полного набора: test_api.py оставлял CELERY_TASK_ALWAYS_EAGER=true, из-за
    чего здесь `.delay()` пытался выполниться синхронно в процессе теста, а не
    уйти в очередь настоящему воркеру). Явно выставляем eager=false, чтобы не
    зависеть от того, что оставил после себя предыдущий тест.
    """
    for name in list(sys.modules):
        if name.startswith("topology_geo.tasks"):
            del sys.modules[name]

    monkeypatch.setenv("CELERY_BROKER_URL", broker_url)
    monkeypatch.setenv("CELERY_RESULT_BACKEND", broker_url)
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "false")

    from topology_geo.tasks.pipeline_tasks import enqueue_job

    enqueue_job(job_id)


def test_real_worker_processes_job_end_to_end(pg_test_db, tmp_path, monkeypatch):
    dbname = pg_test_db.info.dbname
    store.ensure_schema(pg_test_db)
    ensure_relief_schema(pg_test_db)

    osm_file = tmp_path / "sample.osm"
    osm_file.write_text(SAMPLE_OSM_XML, encoding="utf-8")
    _load_sample_osm(dbname, osm_file)

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    _seed_dem(storage_root)
    register_coverage(
        pg_test_db,
        CoverageEntry(
            source_name="TessaDEM (тест)", priority=1, storage_key="tessadem.tif",
            resolution_m=30.0, data_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            footprint_wkt="POLYGON((56.0 57.9, 56.5 57.9, 56.5 58.1, 56.0 58.1, 56.0 57.9))",
        ),
    )

    job = store.create_job(
        pg_test_db, center_lon=JOB_CENTER_LON, center_lat=JOB_CENTER_LAT, radius_m=500,
        layers=[], detail="LOD1", step_names=DEFAULT_STEP_NAMES,
    )

    broker_url = f"redis://localhost:6379/{TEST_REDIS_DB}"
    redis.Redis(host="localhost", port=6379, db=TEST_REDIS_DB).flushdb()

    worker_env = {
        **os.environ,
        "POSTGRES_HOST": os.environ.get("POSTGRES_HOST", "localhost"),
        "POSTGRES_PORT": os.environ.get("POSTGRES_PORT", "5432"),
        "POSTGRES_USER": os.environ.get("POSTGRES_USER", "topology"),
        "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD", "topology"),
        "POSTGRES_DB": dbname,
        "TOPOLOGY_STORAGE_ROOT": str(storage_root),
        "CELERY_BROKER_URL": broker_url,
        "CELERY_RESULT_BACKEND": broker_url,
    }

    worker = subprocess.Popen(
        [sys.executable, "-m", "celery", "-A", "topology_geo.tasks.celery_app", "worker",
         "--loglevel=info", "--pool=solo"],
        env=worker_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        _enqueue_with_fresh_celery_app(str(job.id), broker_url, monkeypatch)

        deadline = time.time() + 40
        final_job = None
        while time.time() < deadline:
            final_job = store.get_job(pg_test_db, job.id)
            if final_job.status in (store.STATUS_DONE, store.STATUS_FAILED):
                break
            time.sleep(0.5)

        worker_output = None
        if final_job is None or final_job.status not in (store.STATUS_DONE, store.STATUS_FAILED):
            worker.terminate()
            worker_output = worker.stdout.read() if worker.stdout else ""
        assert final_job is not None, "задача не завершилась вовремя"
        assert final_job.status == store.STATUS_DONE, (final_job.error_message, worker_output)

        select_osm_step = next(s for s in final_job.steps if s.step_name == "select_osm")
        assert select_osm_step.result["counts"]["osm_buildings"] == 1

        relief_step = next(s for s in final_job.steps if s.step_name == "prepare_relief")
        # читаем через то же файловое хранилище, что использовал воркер
        relief_path = storage_root / relief_step.result["storage_key"]
        is_valid, errors, _ = cog_validate(str(relief_path))
        assert is_valid, errors
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=5)
