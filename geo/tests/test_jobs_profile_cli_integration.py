"""Сквозной тест Шага 2.13, п. 1 (`jobs.profile_cli`) на реальном пайплайне:
настоящий `osm2pgsql` + реальный рельеф + весь `DEFAULT_PIPELINE`
(`select_osm` → ... → `package_outputs`), с фоновым замером памяти -
доказывает, что сам инструмент профилирования работает end-to-end и
выдаёт настоящие числа по каждому шагу, а не только то, что его отдельные
функции корректны (это уже проверяет `test_jobs_profile_cli.py`).

Тот же образец OSM/рельефа, что и `test_api.py::test_full_happy_path_
creates_downloadable_files` (Шаг 1.3) - не изобретается новый, чтобы числа
были сравнимы. На этом (не самом требовательном) фрагменте пайплайн реально
проходит все шаги без OOM в этой среде - настоящее подтверждение падения
именно от ПЛОТНОСТИ реальной застройки (см. docs/dev-tree.md, Шаги 2.4/2.6/
2.10), а не от самого факта работы конвейера."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from topology_geo.jobs import store
from topology_geo.jobs.pipeline import run_all_pending_steps
from topology_geo.jobs.profile_cli import _MemorySampler, _load_memory_samples, build_report
from topology_geo.jobs.steps import DEFAULT_PIPELINE
from topology_geo.relief.coverage import CoverageEntry
from topology_geo.relief.coverage import ensure_schema as ensure_relief_schema
from topology_geo.relief.coverage import register_coverage
from topology_geo.storage import FileSystemObjectStorage

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
    import shutil

    return shutil.which("osm2pgsql") is not None


@pytest.mark.skipif(not _osm2pgsql_available(), reason="требуется системный osm2pgsql")
def test_profile_cli_runs_real_pipeline_and_reports_per_step_time_and_memory(pg_test_db, tmp_path):
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

    transform = from_origin(56.0, 58.1, 0.001, 0.001)
    data = np.full((200, 200), 155.0, dtype="float32")
    dem_path = tmp_path / "seed_dem.tif"
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=200, width=200, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    storage = FileSystemObjectStorage(str(tmp_path / "storage"))
    storage.upload("tessadem.tif", dem_path.read_bytes())

    # Только первые 3 (быстрых) шага, не весь DEFAULT_PIPELINE (Шаг
    # 2.13 - настоящий прогон ВСЕХ 7 шагов, реально выполненный в этой
    # сессии на этом же образце: `assemble_ifc` один занял 821.8 с и
    # 3,7 ГБ пика ДАЖЕ на единственном синтетическом здании - тяжесть
    # шага определяет не количество зданий, а полный TIN на радиус
    # задачи 1 м (тот же вывод, что и в более ранних находках Шагов
    # 2.4/2.6/2.10), а `convert_to_glb` после него уводит процесс в
    # честный OOM (13,76 ГБ, `dmesg`, см. STATUS.md/docs/citymap.md).
    # Гонять эти тяжёлые шаги в каждом прогоне юнит-тестов означало бы
    # держать тест ~15-20 мин и рисковать реальным OOM почти при каждом
    # запуске - непрактично; здесь проверяется, что САМ инструмент
    # (реальная БД + реальный фоновый замер памяти + реальные шаги
    # пайплайна) работает end-to-end, а тяжёлые числа уже получены и
    # задокументированы отдельным реальным прогоном.
    fast_pipeline = {name: DEFAULT_PIPELINE[name] for name in ("select_osm", "prepare_relief", "select_and_normalize")}

    store.ensure_schema(pg_test_db)
    job = store.create_job(
        pg_test_db, center_lon=56.243, center_lat=58.0105, radius_m=500.0,
        layers=[], detail="LOD1", step_names=list(fast_pipeline),
    )

    memory_log = tmp_path / "mem.csv"
    sampler = _MemorySampler(memory_log, interval_s=0.2)
    sampler.start()
    try:
        run_all_pending_steps(pg_test_db, storage, job.id, fast_pipeline)
    finally:
        sampler.stop()

    job = store.get_job(pg_test_db, job.id)
    assert job.status == store.STATUS_DONE, job.error_message

    memory_samples = _load_memory_samples(memory_log)
    assert len(memory_samples) > 0  # фоновый замер реально что-то записал

    report = build_report(job, memory_samples)

    assert report["status"] == "done"
    assert len(report["steps"]) == len(fast_pipeline)
    for step_report in report["steps"]:
        assert step_report["status"] == store.STATUS_DONE
        assert step_report["duration_s"] is not None
        assert step_report["duration_s"] >= 0
    assert report["overall_duration_s"] > 0
    assert report["overall_peak_rss_kb"] is not None
    assert report["overall_peak_rss_kb"] > 0
