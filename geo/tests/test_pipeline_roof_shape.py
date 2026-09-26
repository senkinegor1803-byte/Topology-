"""Сквозной тест Шага 2.2: здание с `roof:shape=gabled` в реальном OSM ->
реальный конвейер (osm2pgsql+PostGIS) -> `site.ifc` содержит `Pset_Крыша`
с формой и высотой конька, а не заглушку с плоской крышей."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import ifcopenshell
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from topology_geo.relief.coverage import CoverageEntry, register_coverage
from topology_geo.relief.coverage import ensure_schema as ensure_relief_schema

STYLE_LUA = Path(__file__).resolve().parents[1] / "osm2pgsql" / "style.lua"
CENTER_LON, CENTER_LAT = 56.243, 58.0105

GABLED_BUILDING_OSM_XML = """\
<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="topology-test">
  <node id="1" lat="58.0100" lon="56.2420" version="1"/>
  <node id="2" lat="58.0100" lon="56.2440" version="1"/>
  <node id="3" lat="58.0110" lon="56.2440" version="1"/>
  <node id="4" lat="58.0110" lon="56.2420" version="1"/>
  <way id="100" version="1">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
    <tag k="building" v="house"/>
    <tag k="height" v="8"/>
    <tag k="roof:shape" v="gabled"/>
    <tag k="roof:height" v="2.5"/>
  </way>
</osm>
"""


def _osm2pgsql_available() -> bool:
    return shutil.which("osm2pgsql") is not None


def _import_osm(dbname: str, tmp_path: Path) -> None:
    osm_file = tmp_path / "gabled.osm"
    osm_file.write_text(GABLED_BUILDING_OSM_XML, encoding="utf-8")
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


def _seed_dem(pg_test_db, tmp_path: Path) -> None:
    ensure_relief_schema(pg_test_db)
    register_coverage(
        pg_test_db,
        CoverageEntry(
            source_name="TessaDEM (тест)", priority=1, storage_key="tessadem.tif",
            resolution_m=30.0, data_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            footprint_wkt="POLYGON((56.0 57.9, 56.5 57.9, 56.5 58.1, 56.0 58.1, 56.0 57.9))",
        ),
    )
    transform = from_origin(56.0, 58.1, 0.001, 0.001)
    data = np.full((200, 200), 155.0, dtype="float32")
    buf_path = tmp_path / "seed_dem.tif"
    with rasterio.open(
        buf_path, "w", driver="GTiff", height=200, width=200, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    from topology_geo.tasks.pipeline_tasks import get_storage

    get_storage().upload("tessadem.tif", buf_path.read_bytes())


def _make_client(dbname: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for mod_name in list(sys.modules):
        if mod_name.startswith(("topology_geo.tasks", "topology_geo.api")):
            del sys.modules[mod_name]
    monkeypatch.setenv("POSTGRES_DB", dbname)
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    monkeypatch.setenv("TOPOLOGY_STORAGE_ROOT", str(tmp_path / "storage"))

    from fastapi.testclient import TestClient

    from topology_geo.api.app import app

    return TestClient(app)


@pytest.mark.skipif(not _osm2pgsql_available(), reason="требуется системный osm2pgsql")
def test_gabled_roof_tag_survives_full_pipeline_into_ifc(pg_test_db, tmp_path, monkeypatch):
    dbname = pg_test_db.info.dbname
    _import_osm(dbname, tmp_path)

    with _make_client(dbname, tmp_path, monkeypatch) as client:
        _seed_dem(pg_test_db, tmp_path)

        resp = client.post("/jobs", json={"center": {"lon": CENTER_LON, "lat": CENTER_LAT}, "radius_m": 500})
        assert resp.status_code == 201
        job_id = resp.json()["id"]

        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "done", job

        files = client.get(f"/models/{job_id}/files").json()["files"]
        ifc_file = next(f for f in files if f["step_name"] == "assemble_ifc:IFC4X3")
        ifc_bytes = client.get(ifc_file["download_url"]).content
        model = ifcopenshell.file.from_string(ifc_bytes.decode("utf-8"))

        building = next(e for e in model.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Здание"))
        psets = {
            rel.RelatingPropertyDefinition.Name: {
                prop.Name: prop.NominalValue.wrappedValue for prop in rel.RelatingPropertyDefinition.HasProperties
            }
            for rel in building.IsDefinedBy
            if rel.RelatingPropertyDefinition.is_a("IfcPropertySet")
        }

        assert "Pset_Крыша" in psets, psets
        assert psets["Pset_Крыша"]["Форма"] == "gabled"
        assert psets["Pset_Крыша"]["Высота_конька_м"] == pytest.approx(2.5)
        assert psets["Pset_Здание"]["Высота_м"] == pytest.approx(8.0)
        assert psets["Pset_Здание"]["Источник_высоты"] == "OSM"
