"""Сквозной тест Шага 2.2, п. 1/3: `building:part` и `entrance` в реальном OSM ->
реальный конвейер (osm2pgsql+PostGIS) -> `site.ifc` содержит по отдельному
зданию на каждую часть (без контура-дубликата) со своими входами, а не
единственную призму по контуру `building=yes`."""

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

# Контур building=yes 20x20 м, разбитый на две building:part половины
# (западная - height=6, восточная - height=9 с двускатной крышей) плюс один
# entrance на общей границе. Контур целиком покрыт частями - по конвенции
# OSM (вики Key:building:part) сам он не должен попасть в site.ifc.
BUILDING_PARTS_OSM_XML = """\
<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="topology-test">
  <node id="1" lat="58.0100" lon="56.2420" version="1"/>
  <node id="2" lat="58.0100" lon="56.2440" version="1"/>
  <node id="3" lat="58.0110" lon="56.2440" version="1"/>
  <node id="4" lat="58.0110" lon="56.2420" version="1"/>

  <node id="10" lat="58.0100" lon="56.2420" version="1"/>
  <node id="11" lat="58.0100" lon="56.2430" version="1"/>
  <node id="12" lat="58.0110" lon="56.2430" version="1"/>
  <node id="13" lat="58.0110" lon="56.2420" version="1"/>

  <node id="20" lat="58.0100" lon="56.2430" version="1"/>
  <node id="21" lat="58.0100" lon="56.2440" version="1"/>
  <node id="22" lat="58.0110" lon="56.2440" version="1"/>
  <node id="23" lat="58.0110" lon="56.2430" version="1"/>

  <node id="30" lat="58.0100" lon="56.2430" version="1">
    <tag k="entrance" v="main"/>
  </node>

  <way id="100" version="1">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
    <tag k="building" v="yes"/>
  </way>
  <way id="101" version="1">
    <nd ref="10"/><nd ref="11"/><nd ref="12"/><nd ref="13"/><nd ref="10"/>
    <tag k="building:part" v="yes"/>
    <tag k="height" v="6"/>
  </way>
  <way id="102" version="1">
    <nd ref="20"/><nd ref="21"/><nd ref="22"/><nd ref="23"/><nd ref="20"/>
    <tag k="building:part" v="yes"/>
    <tag k="height" v="9"/>
    <tag k="roof:shape" v="gabled"/>
    <tag k="roof:height" v="2"/>
  </way>
</osm>
"""


def _osm2pgsql_available() -> bool:
    return shutil.which("osm2pgsql") is not None


def _import_osm(dbname: str, tmp_path: Path) -> None:
    osm_file = tmp_path / "parts.osm"
    osm_file.write_text(BUILDING_PARTS_OSM_XML, encoding="utf-8")
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
def test_building_parts_and_entrance_survive_full_pipeline_into_ifc(pg_test_db, tmp_path, monkeypatch):
    dbname = pg_test_db.info.dbname
    _import_osm(dbname, tmp_path)

    with _make_client(dbname, tmp_path, monkeypatch) as client:
        _seed_dem(pg_test_db, tmp_path)

        resp = client.post("/jobs", json={"center": {"lon": CENTER_LON, "lat": CENTER_LAT}, "radius_m": 500})
        assert resp.status_code == 201
        job_id = resp.json()["id"]

        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "done", job
        assemble_step = next(s for s in job["steps"] if s["step_name"] == "assemble_ifc")
        assert assemble_step["result"]["buildings"] == 2  # две части, БЕЗ контура-дубликата

        files = client.get(f"/models/{job_id}/files").json()["files"]
        ifc_file = next(f for f in files if f["step_name"] == "assemble_ifc:IFC4X3")
        ifc_bytes = client.get(ifc_file["download_url"]).content
        model = ifcopenshell.file.from_string(ifc_bytes.decode("utf-8"))

        buildings = [e for e in model.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Здание")]
        assert len(buildings) == 2

        def _psets(element):
            return {
                rel.RelatingPropertyDefinition.Name: {
                    prop.Name: prop.NominalValue.wrappedValue
                    for prop in rel.RelatingPropertyDefinition.HasProperties
                }
                for rel in element.IsDefinedBy
                if rel.RelatingPropertyDefinition.is_a("IfcPropertySet")
            }

        by_height = sorted((_psets(b) for b in buildings), key=lambda p: p["Pset_Здание"]["Высота_м"])
        low, high = by_height

        assert low["Pset_Здание"]["Высота_м"] == pytest.approx(6.0)
        assert low["Pset_Контекст"]["Часть_здания"] is True
        assert "Pset_Крыша" not in low or not low["Pset_Крыша"]

        assert high["Pset_Здание"]["Высота_м"] == pytest.approx(9.0)
        assert high["Pset_Крыша"]["Форма"] == "gabled"
        assert high["Pset_Контекст"]["Часть_здания"] is True

        # Вход на общей границе частей (56.2430, 58.0100) должен попасть хотя
        # бы в одну из двух частей (какую именно - зависит от геометрии
        # буфера, важно само наличие, без потери в никуда).
        entrance_counts = [
            p["Pset_Здание"].get("Входов_всего", 0) for p in (low, high)
        ]
        assert sum(entrance_counts) >= 1
