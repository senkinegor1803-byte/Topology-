"""Тесты Шага 2.11, п. 1: шаг пайплайна `generate_tileset` (3D Tiles)."""

from __future__ import annotations

import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pygltflib
import pytest
import rasterio
from rasterio.transform import from_origin

from topology_geo.coords import MSK59_ZONES, pick_msk59_zone, wgs84_to_msk59
from topology_geo.jobs import store
from topology_geo.jobs.steps import generate_tileset
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
    <tag k="height" v="9"/>
  </way>
</osm>
"""


def _osm2pgsql_available() -> bool:
    import shutil

    return shutil.which("osm2pgsql") is not None


def _make_job(conn, *, radius_m=500.0):
    store.ensure_schema(conn)
    return store.create_job(
        conn, center_lon=PILOT_CENTER_LON, center_lat=PILOT_CENTER_LAT, radius_m=radius_m,
        layers=[], detail="LOD1", step_names=["generate_tileset"],
    )


def _make_msk59_relief_tif(zone: int, center_x: float, center_y: float, half_extent: float, elevation: float) -> bytes:
    pixel = 5.0
    size = int(2 * half_extent / pixel)
    transform = from_origin(center_x - half_extent, center_y + half_extent, pixel, pixel)
    data = np.full((size, size), elevation, dtype="float64")
    buf = io.BytesIO()
    with rasterio.open(
        buf, "w", driver="GTiff", height=size, width=size, count=1, dtype="float64",
        crs=MSK59_ZONES[zone].to_proj4(), transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return buf.getvalue()


@pytest.mark.skipif(not _osm2pgsql_available(), reason="требуется системный osm2pgsql")
def test_generate_tileset_produces_valid_tileset_and_glb(pg_test_db, tmp_path):
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

    job = _make_job(pg_test_db)
    zone = pick_msk59_zone(job.center_lon)
    center_x, center_y, _ = wgs84_to_msk59(job.center_lon, job.center_lat, zone=zone)

    storage = InMemoryObjectStorage()
    relief_bytes = _make_msk59_relief_tif(zone, center_x, center_y, job.radius_m + 200.0, elevation=150.0)
    storage.upload(f"jobs/{job.id}/relief.tif", relief_bytes)

    result = generate_tileset(pg_test_db, storage, job)

    assert result["tile_count"] > 0
    # Радиус задачи 500 м = граница LOD2/LOD1 (Шаг 2.1); тайлы 250×250 м,
    # чей КВАДРАТ пересекает круг радиуса 500 м, но ЦЕНТР тайла чуть дальше,
    # честно классифицируются в LOD1 - не все тайлы обязаны быть LOD2.
    assert result["tiles_by_lod"]["LOD2"] > 0
    assert result["tiles_by_lod"]["LOD2"] + result["tiles_by_lod"]["LOD1"] == result["tile_count"]
    assert result["tiles_by_lod"]["LOD0"] == 0  # радиус 500 м не достаёт до LOD0 (>1500 м)

    tileset_doc = json.loads(storage.download(result["tileset_storage_key"]))
    assert tileset_doc["asset"]["version"] == "1.1"
    assert len(tileset_doc["root"]["children"]) == result["tile_count"]

    # Первый тайл содержит реальный валидный GLB
    first_uri = tileset_doc["root"]["children"][0]["content"]["uri"]
    glb_bytes = storage.download(first_uri)
    gltf = pygltflib.GLTF2.load_from_bytes(glb_bytes)
    assert len(gltf.meshes) > 0

    # Хотя бы один тайл содержит здание (узел категории "Здания")
    building_found = False
    for entry in tileset_doc["root"]["children"]:
        doc = pygltflib.GLTF2.load_from_bytes(storage.download(entry["content"]["uri"]))
        if any(node.name == "Здания" for node in doc.nodes):
            building_found = True
            break
    assert building_found


def test_generate_tileset_raises_clear_error_without_osm_tables(pg_test_db):
    job = _make_job(pg_test_db)
    zone = pick_msk59_zone(job.center_lon)
    center_x, center_y, _ = wgs84_to_msk59(job.center_lon, job.center_lat, zone=zone)
    storage = InMemoryObjectStorage()
    storage.upload(
        f"jobs/{job.id}/relief.tif",
        _make_msk59_relief_tif(zone, center_x, center_y, job.radius_m + 200.0, elevation=150.0),
    )

    with pytest.raises(RuntimeError, match="нет данных OSM"):
        generate_tileset(pg_test_db, storage, job)
