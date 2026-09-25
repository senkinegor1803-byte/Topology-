"""Тесты FastAPI (Шаг 1.3, п. 1, 3, 4).

Как и test_tasks_celery.py, эти тесты не могут импортировать
`topology_geo.api.app`/`topology_geo.tasks.*` на уровне модуля: приложению
нужен `CELERY_TASK_ALWAYS_EAGER=true` (выполнить пайплайн синхронно в
процессе теста, без отдельного воркера) и правильный `POSTGRES_DB` — оба
читаются один раз при первом импорте `topology_geo.tasks.celery_app`. Поэтому
окружение выставляется, кэш `sys.modules` чистится, и только потом делается
локальный импорт внутри каждого теста/фикстуры.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from topology_geo.relief.coverage import CoverageEntry
from topology_geo.relief.coverage import ensure_schema as ensure_relief_schema
from topology_geo.relief.coverage import register_coverage

STYLE_LUA = Path(__file__).resolve().parents[1] / "osm2pgsql" / "style.lua"

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


def _osm2pgsql_available() -> bool:
    return shutil.which("osm2pgsql") is not None


def _make_client(dbname: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Настроить окружение под eager Celery + нужную тестовую БД и
    (пере)импортировать приложение с нуля, чтобы оно подхватило именно это
    окружение (см. docstring модуля).

    Через `monkeypatch`, а не прямое присваивание `os.environ[...]` — иначе
    переменные (`POSTGRES_DB` и т.д.) утекают в остальные тесты модуля и
    ломают их: именно так проявлялся реальный баг при первом прогоне полного
    набора (test_tasks_celery.py пытался подключиться к уже удалённой БД
    предыдущего теста, потому что POSTGRES_DB не был сброшен).
    """
    for name in list(sys.modules):
        if name.startswith("topology_geo.tasks") or name.startswith("topology_geo.api"):
            del sys.modules[name]

    monkeypatch.setenv("POSTGRES_DB", dbname)
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    monkeypatch.setenv("TOPOLOGY_STORAGE_ROOT", str(tmp_path / "storage"))

    from fastapi.testclient import TestClient

    from topology_geo.api.app import app

    return TestClient(app)


@pytest.fixture()
def client(pg_test_db, tmp_path, monkeypatch):
    with _make_client(pg_test_db.info.dbname, tmp_path, monkeypatch) as c:
        yield c


def test_create_job_rejects_radius_out_of_range(client):
    resp = client.post("/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 50})
    assert resp.status_code == 422


def test_create_job_rejects_coordinates_outside_region(client):
    resp = client.post("/jobs", json={"center": {"lon": 30.0, "lat": 58.01}, "radius_m": 1000})
    assert resp.status_code == 422
    body = resp.json()
    assert "поддерживаемого региона" in str(body)


def test_create_job_then_get_shows_failed_without_seed_data(client):
    resp = client.post("/jobs", json={"center": {"lon": 56.24, "lat": 58.01}, "radius_m": 500})
    assert resp.status_code == 201
    job_id = resp.json()["id"]

    got = client.get(f"/jobs/{job_id}")
    assert got.status_code == 200
    body = got.json()
    assert body["status"] == "failed"
    assert "нет данных OSM" in body["error_message"]
    assert body["steps"][0]["step_name"] == "select_osm"
    assert body["steps"][0]["status"] == "failed"
    assert body["steps"][1]["status"] == "pending"  # второй шаг не запускался


def test_get_job_returns_404_for_unknown_id(client):
    import uuid

    resp = client.get(f"/jobs/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_get_files_returns_404_for_unknown_job(client):
    import uuid

    resp = client.get(f"/models/{uuid.uuid4()}/files")
    assert resp.status_code == 404


@pytest.mark.skipif(not _osm2pgsql_available(), reason="требуется системный osm2pgsql")
def test_full_happy_path_creates_downloadable_files(pg_test_db, tmp_path, monkeypatch):
    dbname = pg_test_db.info.dbname
    osm_file = tmp_path / "sample.osm"
    osm_file.write_text(SAMPLE_OSM_XML, encoding="utf-8")
    env = {**os.environ, "PGPASSWORD": os.environ.get("POSTGRES_PASSWORD", "topology")}
    subprocess.run(
        [
            "osm2pgsql", "--output=flex", f"--style={STYLE_LUA}",
            f"--host={os.environ.get('POSTGRES_HOST', 'localhost')}",
            f"--port={os.environ.get('POSTGRES_PORT', '5432')}",
            f"--user={os.environ.get('POSTGRES_USER', 'topology')}",
            f"--database={dbname}", str(osm_file),
        ],
        check=True, capture_output=True, text=True, env=env,
    )

    ensure_relief_schema(pg_test_db)
    register_coverage(
        pg_test_db,
        CoverageEntry(
            source_name="TessaDEM (тест)", priority=1, storage_key="tessadem.tif",
            resolution_m=30.0, data_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            footprint_wkt="POLYGON((56.0 57.9, 56.5 57.9, 56.5 58.1, 56.0 58.1, 56.0 57.9))",
        ),
    )

    with _make_client(dbname, tmp_path, monkeypatch) as client:
        # DEM-семпл кладём в то же файловое хранилище, которым будет пользоваться приложение
        from topology_geo.tasks.pipeline_tasks import get_storage

        transform = from_origin(56.0, 58.1, 0.001, 0.001)
        data = np.full((200, 200), 155.0, dtype="float32")
        buf_path = tmp_path / "seed_dem.tif"
        with rasterio.open(
            buf_path, "w", driver="GTiff", height=200, width=200, count=1,
            dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)
        get_storage().upload("tessadem.tif", buf_path.read_bytes())

        resp = client.post("/jobs", json={"center": {"lon": 56.243, "lat": 58.0105}, "radius_m": 500})
        assert resp.status_code == 201
        job_id = resp.json()["id"]

        got = client.get(f"/jobs/{job_id}")
        assert got.json()["status"] == "done", got.json()

        files_resp = client.get(f"/models/{job_id}/files")
        assert files_resp.status_code == 200
        files = files_resp.json()["files"]
        assert {f["step_name"] for f in files} == {"select_osm", "prepare_relief", "select_and_normalize"}
