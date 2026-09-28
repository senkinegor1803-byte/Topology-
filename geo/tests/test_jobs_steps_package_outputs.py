"""Тесты Шага 2.10, п. 3: `package_outputs` (meta.json + zip-архив пакета).

Не пересобирает геометрию/TIN (дорого, см. OOM в `docs/math-model.md`) —
подсовывает синтетические результаты предыдущих шагов через `store.finish_step`
напрямую и синтетические байты в `InMemoryObjectStorage`, проверяя только
логику САМОГО `package_outputs` (сборка списка файлов, meta.json по схеме,
переименование `roads_internal` -> `roads_local` в архиве)."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import jsonschema
import pytest

from topology_geo.jobs import store
from topology_geo.jobs.steps import IFC_SCHEMAS, package_outputs
from topology_geo.osm.import_log import ImportLogEntry, record_import
from topology_geo.osm.import_log import ensure_schema as ensure_import_log_schema
from topology_geo.relief.coverage import CoverageEntry, register_coverage
from topology_geo.relief.coverage import ensure_schema as ensure_dem_coverage_schema
from topology_geo.storage import InMemoryObjectStorage

PILOT_CENTER_LON, PILOT_CENTER_LAT = 56.2431, 58.0105
META_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "meta.schema.json"
PERM_REGION_WKT = "POLYGON((50 55, 62 55, 62 60, 50 60, 50 55))"


def _make_job(conn, step_names):
    store.ensure_schema(conn)
    return store.create_job(
        conn, center_lon=PILOT_CENTER_LON, center_lat=PILOT_CENTER_LAT, radius_m=500.0,
        layers=[], detail="LOD1", step_names=step_names,
    )


def _seed_storage_and_finish_steps(conn, job_id) -> InMemoryObjectStorage:
    storage = InMemoryObjectStorage()
    prefix = f"jobs/{job_id}"

    normalize_key = f"{prefix}/site.gpkg"
    storage.upload(normalize_key, b"gpkg-bytes")
    store.finish_step(conn, job_id, "select_and_normalize", {"storage_key": normalize_key})

    schemas: dict[str, dict] = {}
    for schema in IFC_SCHEMAS:
        suffix = schema.lower()
        schemas[schema] = {"storage_key": f"{prefix}/site_{suffix}.ifc"}
        for layer in ("roads_backbone", "roads_internal", "relief", "buildings", "power"):
            key = f"{prefix}/{layer}_{suffix}.ifc"
            storage.upload(key, f"{layer}-{suffix}-bytes".encode())
            schemas[schema][f"{layer}_storage_key"] = key
        storage.upload(schemas[schema]["storage_key"], f"site-{suffix}-bytes".encode())

    landxml_key = f"{prefix}/site.landxml"
    cityjson_key = f"{prefix}/site.cityjson"
    dxf_key = f"{prefix}/site.dxf"
    storage.upload(landxml_key, b"<LandXML/>")
    storage.upload(cityjson_key, b'{"type": "CityJSON"}')
    storage.upload(dxf_key, b"dxf-bytes")

    store.finish_step(
        conn, job_id, "assemble_ifc",
        {
            "schemas": schemas,
            "landxml_storage_key": landxml_key,
            "cityjson_storage_key": cityjson_key,
            "dxf_storage_key": dxf_key,
            "base_point": {"x": 2310450.0, "y": -5857320.0, "height": 150.0, "zone": 2},
        },
    )

    glb_key = f"{prefix}/site.glb"
    storage.upload(glb_key, b"glb-bytes")
    store.finish_step(conn, job_id, "convert_to_glb", {"storage_key": glb_key, "size_bytes": 10})

    return storage


@pytest.fixture(scope="module")
def meta_schema() -> dict:
    return json.loads(META_SCHEMA_PATH.read_text(encoding="utf-8"))


def test_package_outputs_returns_storage_keys(pg_test_db):
    job = _make_job(pg_test_db, ["select_and_normalize", "assemble_ifc", "convert_to_glb", "package_outputs"])
    storage = _seed_storage_and_finish_steps(pg_test_db, job.id)
    job = store.get_job(pg_test_db, job.id)

    result = package_outputs(pg_test_db, storage, job)

    assert result["meta_storage_key"] == f"jobs/{job.id}/meta.json"
    assert result["archive_storage_key"] == f"jobs/{job.id}/package.zip"
    assert result["file_count"] > 0
    assert storage.exists(result["meta_storage_key"])
    assert storage.exists(result["archive_storage_key"])


def test_package_outputs_meta_json_validates_against_schema(pg_test_db, meta_schema):
    job = _make_job(pg_test_db, ["select_and_normalize", "assemble_ifc", "convert_to_glb", "package_outputs"])
    storage = _seed_storage_and_finish_steps(pg_test_db, job.id)
    job = store.get_job(pg_test_db, job.id)

    result = package_outputs(pg_test_db, storage, job)
    meta = json.loads(storage.download(result["meta_storage_key"]))

    jsonschema.validate(meta, meta_schema)
    assert meta["task_id"] == str(job.id)
    assert meta["radius_m"] == 500.0
    assert meta["coordinate_system"]["base_point"] == {"x": 2310450.0, "y": -5857320.0, "height": 150.0}


def test_package_outputs_archive_renames_roads_internal_to_roads_local(pg_test_db):
    job = _make_job(pg_test_db, ["select_and_normalize", "assemble_ifc", "convert_to_glb", "package_outputs"])
    storage = _seed_storage_and_finish_steps(pg_test_db, job.id)
    job = store.get_job(pg_test_db, job.id)

    result = package_outputs(pg_test_db, storage, job)
    archive = zipfile.ZipFile(BytesIO(storage.download(result["archive_storage_key"])))
    names = set(archive.namelist())

    assert "roads_local_ifc4.ifc" in names
    assert "roads_local_ifc4x3.ifc" in names
    assert not any("roads_internal" in name for name in names)
    assert "meta.json" in names
    assert "site.landxml" in names
    assert "site.cityjson" in names
    assert "site.dxf" in names
    assert "site.glb" in names
    assert "site.gpkg" in names


def test_package_outputs_archive_content_matches_uploaded_bytes(pg_test_db):
    job = _make_job(pg_test_db, ["select_and_normalize", "assemble_ifc", "convert_to_glb", "package_outputs"])
    storage = _seed_storage_and_finish_steps(pg_test_db, job.id)
    job = store.get_job(pg_test_db, job.id)

    result = package_outputs(pg_test_db, storage, job)
    archive = zipfile.ZipFile(BytesIO(storage.download(result["archive_storage_key"])))

    assert archive.read("site.gpkg") == b"gpkg-bytes"
    assert archive.read("relief_ifc4.ifc") == b"relief-ifc4-bytes"


def test_package_outputs_sources_include_real_osm_import_log_entry(pg_test_db):
    ensure_import_log_schema(pg_test_db)
    record_import(
        pg_test_db,
        ImportLogEntry(
            source_name="OSM Приволжский ФО (Geofabrik)",
            source_file="perm.osm.pbf",
            data_timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        ),
    )
    job = _make_job(pg_test_db, ["select_and_normalize", "assemble_ifc", "convert_to_glb", "package_outputs"])
    storage = _seed_storage_and_finish_steps(pg_test_db, job.id)
    job = store.get_job(pg_test_db, job.id)

    result = package_outputs(pg_test_db, storage, job)
    meta = json.loads(storage.download(result["meta_storage_key"]))

    osm_sources = [s for s in meta["sources"] if s["name"] == "OSM Приволжский ФО (Geofabrik)"]
    assert len(osm_sources) == 1
    assert osm_sources[0]["license"] == "ODbL"
    assert osm_sources[0]["retrieved_at"] == "2026-09-01"


def test_package_outputs_sources_include_real_dem_coverage_entry(pg_test_db):
    ensure_dem_coverage_schema(pg_test_db)
    register_coverage(
        pg_test_db,
        CoverageEntry(
            source_name="TessaDEM", priority=1, storage_key="relief/tessadem.tif",
            resolution_m=30.0, data_timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
            footprint_wkt=PERM_REGION_WKT,
        ),
    )
    job = _make_job(pg_test_db, ["select_and_normalize", "assemble_ifc", "convert_to_glb", "package_outputs"])
    storage = _seed_storage_and_finish_steps(pg_test_db, job.id)
    job = store.get_job(pg_test_db, job.id)

    result = package_outputs(pg_test_db, storage, job)
    meta = json.loads(storage.download(result["meta_storage_key"]))

    dem_sources = [s for s in meta["sources"] if s["name"] == "TessaDEM"]
    assert len(dem_sources) == 1
    assert dem_sources[0]["retrieved_at"] == "2026-06-01"


def test_package_outputs_sources_empty_when_no_logs_recorded(pg_test_db):
    job = _make_job(pg_test_db, ["select_and_normalize", "assemble_ifc", "convert_to_glb", "package_outputs"])
    storage = _seed_storage_and_finish_steps(pg_test_db, job.id)
    job = store.get_job(pg_test_db, job.id)

    result = package_outputs(pg_test_db, storage, job)
    meta = json.loads(storage.download(result["meta_storage_key"]))

    assert meta["sources"] == []
