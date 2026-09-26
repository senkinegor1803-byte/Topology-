"""Сквозной тест Шага 2.3, п. 1: реальный перекрёсток из OSM -> реальный
конвейер (osm2pgsql+PostGIS) -> osm2streets (реальный процесс Node.js) ->
`site.ifc` содержит полосы (Pset_Полоса) и перекрёсток (Pset_Перекрёсток),
а не одну ленту на дорогу (Шаг 1.7)."""

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

from topology_geo.geometry.streets import is_osm2streets_available
from topology_geo.relief.coverage import CoverageEntry, register_coverage
from topology_geo.relief.coverage import ensure_schema as ensure_relief_schema

STYLE_LUA = Path(__file__).resolve().parents[1] / "osm2pgsql" / "style.lua"
CENTER_LON, CENTER_LAT = 56.243, 58.0105

# Перекрёсток из 4 улиц вокруг центра задачи - тот же профиль, что и в
# test_osm_raw_roads.py/test_geometry_streets.py, но теперь через реальный
# osm2pgsql импорт и весь HTTP-пайплайн задачи.
CROSSROADS_OSM_XML = """\
<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="topology-test">
  <node id="1" lat="58.0105" lon="56.2430" version="1"/>
  <node id="2" lat="58.0125" lon="56.2430" version="1"/>
  <node id="3" lat="58.0105" lon="56.2460" version="1"/>
  <node id="4" lat="58.0085" lon="56.2430" version="1"/>
  <node id="5" lat="58.0105" lon="56.2400" version="1"/>
  <way id="10" version="1">
    <nd ref="1"/><nd ref="2"/>
    <tag k="highway" v="residential"/>
    <tag k="lanes" v="2"/>
    <tag k="name" v="North St"/>
  </way>
  <way id="11" version="1">
    <nd ref="1"/><nd ref="3"/>
    <tag k="highway" v="residential"/>
    <tag k="lanes" v="2"/>
    <tag k="name" v="East St"/>
  </way>
  <way id="12" version="1">
    <nd ref="1"/><nd ref="4"/>
    <tag k="highway" v="residential"/>
    <tag k="lanes" v="2"/>
    <tag k="name" v="South St"/>
  </way>
  <way id="13" version="1">
    <nd ref="1"/><nd ref="5"/>
    <tag k="highway" v="residential"/>
    <tag k="lanes" v="2"/>
    <tag k="name" v="West St"/>
  </way>
</osm>
"""


def _osm2pgsql_available() -> bool:
    return shutil.which("osm2pgsql") is not None


def _import_osm(dbname: str, tmp_path: Path) -> None:
    osm_file = tmp_path / "crossroads.osm"
    osm_file.write_text(CROSSROADS_OSM_XML, encoding="utf-8")
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
@pytest.mark.skipif(not is_osm2streets_available(), reason="требуется node и `npm install` в geo/osm2streets")
def test_crossroads_lanes_and_intersection_survive_full_pipeline_into_ifc(pg_test_db, tmp_path, monkeypatch):
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
        assert assemble_step["result"]["lanes"] == 24  # 4 улицы x (2 проезжие + 2 тротуара + 2 бордюра)
        assert assemble_step["result"]["intersections"] == 4
        assert assemble_step["result"]["markings"] > 0

        files = client.get(f"/models/{job_id}/files").json()["files"]
        ifc_file = next(f for f in files if f["step_name"] == "assemble_ifc:IFC4X3")
        ifc_bytes = client.get(ifc_file["download_url"]).content
        model = ifcopenshell.file.from_string(ifc_bytes.decode("utf-8"))

        def _psets(element):
            return {
                rel.RelatingPropertyDefinition.Name: {
                    prop.Name: prop.NominalValue.wrappedValue
                    for prop in rel.RelatingPropertyDefinition.HasProperties
                }
                for rel in element.IsDefinedBy
                if rel.RelatingPropertyDefinition.is_a("IfcPropertySet")
            }

        lanes = [e for e in model.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Полоса ")]
        assert len(lanes) == 24
        lane_types = {_psets(lane)["Pset_Полоса"]["Тип"] for lane in lanes}
        assert lane_types == {"Driving", "Sidewalk", "Curb"}

        intersections = [
            e for e in model.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Перекрёсток")
        ]
        assert len(intersections) == 4

        markings = [e for e in model.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Разметка")]
        assert len(markings) > 0  # один продукт на вид разметки, не на штрих
        assert {_psets(m)["Pset_Разметка"]["Тип"] for m in markings} <= {"center line", "lane arrow"}
