"""Тесты реальных шагов пайплайна (Шаг 1.3): select_osm поверх Шага 1.1,
prepare_relief поверх Шага 1.2. Требуют реальный Postgres+PostGIS (фикстура
`pg_test_db`); select_osm-тесты с данными дополнительно требуют системный
`osm2pgsql` на PATH."""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import io

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rio_cogeo.cogeo import cog_validate

import pyogrio

from topology_geo.jobs import store
from topology_geo.jobs.steps import prepare_relief, select_and_normalize, select_osm
from topology_geo.relief.coverage import CoverageEntry, ensure_schema as ensure_relief_schema, register_coverage
from topology_geo.storage import InMemoryObjectStorage

PILOT_CENTER_LON, PILOT_CENTER_LAT = 56.2431, 58.0105

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


def _make_job(conn, *, radius_m=500.0, step_names):
    store.ensure_schema(conn)
    return store.create_job(
        conn, center_lon=PILOT_CENTER_LON, center_lat=PILOT_CENTER_LAT, radius_m=radius_m,
        layers=[], detail="LOD1", step_names=step_names,
    )


@pytest.mark.skipif(not _osm2pgsql_available(), reason="требуется системный osm2pgsql")
def test_select_osm_returns_counts_and_stores_json(pg_test_db):
    import os

    osm_file = Path("/tmp") / "jobs_steps_sample.osm"
    osm_file.write_text(SAMPLE_OSM_XML, encoding="utf-8")
    dbname = pg_test_db.info.dbname
    subprocess.run(
        [
            "osm2pgsql", "--output=flex", f"--style={STYLE_LUA}",
            f"--host={os.environ.get('POSTGRES_HOST', 'localhost')}",
            f"--port={os.environ.get('POSTGRES_PORT', '5432')}",
            f"--user={os.environ.get('POSTGRES_USER', 'topology')}",
            f"--database={dbname}", str(osm_file),
        ],
        check=True, capture_output=True, text=True,
        env={**os.environ, "PGPASSWORD": os.environ.get("POSTGRES_PASSWORD", "topology")},
    )

    job = _make_job(pg_test_db, step_names=["select_osm"])
    storage = InMemoryObjectStorage()

    result = select_osm(pg_test_db, storage, job)

    assert result["counts"]["osm_buildings"] == 1
    stored = storage.download(result["storage_key"])
    assert b"osm_buildings" in stored


def test_select_osm_raises_clear_error_without_osm_tables(pg_test_db):
    job = _make_job(pg_test_db, step_names=["select_osm"])
    storage = InMemoryObjectStorage()

    with pytest.raises(RuntimeError, match="нет данных OSM"):
        select_osm(pg_test_db, storage, job)


def _make_tif_bytes(bounds, size, value) -> bytes:
    minx, miny, maxx, maxy = bounds
    w, h = size
    px, py = (maxx - minx) / w, (maxy - miny) / h
    transform = from_origin(minx, maxy, px, py)
    data = np.full((h, w), value, dtype="float32")
    buf = io.BytesIO()
    with rasterio.open(
        buf, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return buf.getvalue()


class _StaticStorage(InMemoryObjectStorage):
    def __init__(self, seed: dict[str, bytes]):
        super().__init__()
        self._data.update(seed)


def test_prepare_relief_produces_valid_cog_when_covered(pg_test_db):
    ensure_relief_schema(pg_test_db)
    footprint = (
        "POLYGON((56.0 57.9, 56.5 57.9, 56.5 58.1, 56.0 58.1, 56.0 57.9))"
    )
    register_coverage(
        pg_test_db,
        CoverageEntry(
            source_name="TessaDEM (тест)", priority=1, storage_key="tessadem.tif",
            resolution_m=30.0, data_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            footprint_wkt=footprint,
        ),
    )

    storage = _StaticStorage({"tessadem.tif": _make_tif_bytes((56.0, 57.9, 56.5, 58.1), (200, 200), 150.0)})
    job = _make_job(pg_test_db, radius_m=500.0, step_names=["prepare_relief"])

    result = prepare_relief(pg_test_db, storage, job)

    assert result["coverage_fraction"] == pytest.approx(1.0)
    cog_bytes = storage.download(result["storage_key"])
    tmp_path = Path("/tmp/jobs_steps_relief_out.tif")
    tmp_path.write_bytes(cog_bytes)
    is_valid, errors, _ = cog_validate(str(tmp_path))
    assert is_valid, errors

    with rasterio.open(tmp_path) as f:
        data = f.read(1)
    assert data.mean() == pytest.approx(150.0, abs=1.0)


def test_prepare_relief_raises_clear_error_without_coverage(pg_test_db):
    ensure_relief_schema(pg_test_db)
    storage = InMemoryObjectStorage()
    job = _make_job(pg_test_db, step_names=["prepare_relief"])

    with pytest.raises(RuntimeError, match="нет данных рельефа в этой области"):
        prepare_relief(pg_test_db, storage, job)


@pytest.mark.skipif(not _osm2pgsql_available(), reason="требуется системный osm2pgsql")
def test_select_and_normalize_produces_geopackage(pg_test_db, tmp_path):
    import os

    osm_file = tmp_path / "sample.osm"
    osm_file.write_text(SAMPLE_OSM_XML, encoding="utf-8")
    subprocess.run(
        [
            "osm2pgsql", "--output=flex", f"--style={STYLE_LUA}",
            f"--host={os.environ.get('POSTGRES_HOST', 'localhost')}",
            f"--port={os.environ.get('POSTGRES_PORT', '5432')}",
            f"--user={os.environ.get('POSTGRES_USER', 'topology')}",
            f"--database={pg_test_db.info.dbname}", str(osm_file),
        ],
        check=True, capture_output=True, text=True,
        env={**os.environ, "PGPASSWORD": os.environ.get("POSTGRES_PASSWORD", "topology")},
    )

    job = _make_job(pg_test_db, step_names=["select_and_normalize"])
    storage = InMemoryObjectStorage()

    result = select_and_normalize(pg_test_db, storage, job)

    assert result["feature_count"] == 1
    assert result["layers"] == {"osm_buildings": 1}
    assert result["zone"] == 2

    gpkg_bytes = storage.download(result["storage_key"])
    gpkg_path = tmp_path / "out.gpkg"
    gpkg_path.write_bytes(gpkg_bytes)
    layers = {name for name, _ in pyogrio.list_layers(gpkg_path)}
    assert layers == {"osm_buildings"}


def test_select_and_normalize_raises_clear_error_without_osm_tables(pg_test_db):
    job = _make_job(pg_test_db, step_names=["select_and_normalize"])
    storage = InMemoryObjectStorage()

    with pytest.raises(RuntimeError, match="нет данных OSM"):
        select_and_normalize(pg_test_db, storage, job)


@pytest.mark.skipif(not _osm2pgsql_available(), reason="требуется системный osm2pgsql")
def test_select_and_normalize_raises_when_nothing_in_radius(pg_test_db, tmp_path):
    import os

    osm_file = tmp_path / "sample.osm"
    osm_file.write_text(SAMPLE_OSM_XML, encoding="utf-8")
    subprocess.run(
        [
            "osm2pgsql", "--output=flex", f"--style={STYLE_LUA}",
            f"--host={os.environ.get('POSTGRES_HOST', 'localhost')}",
            f"--port={os.environ.get('POSTGRES_PORT', '5432')}",
            f"--user={os.environ.get('POSTGRES_USER', 'topology')}",
            f"--database={pg_test_db.info.dbname}", str(osm_file),
        ],
        check=True, capture_output=True, text=True,
        env={**os.environ, "PGPASSWORD": os.environ.get("POSTGRES_PASSWORD", "topology")},
    )

    # задача далеко от загруженного здания (но всё ещё в зоне МСК-59) ->
    # в буфере нет объектов
    store.ensure_schema(pg_test_db)
    job = store.create_job(
        pg_test_db, center_lon=57.0, center_lat=58.0, radius_m=500.0,
        layers=[], detail="LOD1", step_names=["select_and_normalize"],
    )
    storage = InMemoryObjectStorage()

    with pytest.raises(ValueError, match="нет ни одного объекта"):
        select_and_normalize(pg_test_db, storage, job)
