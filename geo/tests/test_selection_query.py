"""Тесты выборки объектов OSM в буфере (Шаг 1.4, п. 1). Требуют реальный
osm2pgsql + Postgres+PostGIS (фикстура `pg_test_db`); пропускаются, если
osm2pgsql недоступен на PATH."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from topology_geo.selection.query import fetch_features_in_buffer

STYLE_LUA = Path(__file__).resolve().parents[1] / "osm2pgsql" / "style.lua"

# Здание рядом с центром (~11 м) и здание далеко (~1.1 км) - второе не должно
# попасть в буфер радиуса 500 м с запасом 50 м.
SAMPLE_OSM_XML = """\
<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="topology-test">
  <node id="1" lat="58.0100" lon="56.2420" version="1"/>
  <node id="2" lat="58.0100" lon="56.2422" version="1"/>
  <node id="3" lat="58.0101" lon="56.2422" version="1"/>
  <node id="4" lat="58.0101" lon="56.2420" version="1"/>
  <node id="10" lat="58.0200" lon="56.2420" version="1"/>
  <node id="11" lat="58.0200" lon="56.2422" version="1"/>
  <node id="12" lat="58.0201" lon="56.2422" version="1"/>
  <node id="13" lat="58.0201" lon="56.2420" version="1"/>
  <node id="20" lat="58.0100" lon="56.2400" version="1">
    <tag k="power" v="pole"/>
    <tag k="voltage" v="10000"/>
  </node>

  <way id="100" version="1">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
    <tag k="building" v="yes"/>
    <tag k="building:levels" v="5"/>
  </way>
  <way id="200" version="1">
    <nd ref="10"/><nd ref="11"/><nd ref="12"/><nd ref="13"/><nd ref="10"/>
    <tag k="building" v="yes"/>
  </way>
</osm>
"""

CENTER_LON, CENTER_LAT = 56.2421, 58.01005


def _osm2pgsql_available() -> bool:
    return shutil.which("osm2pgsql") is not None


pytestmark = pytest.mark.skipif(not _osm2pgsql_available(), reason="требуется системный osm2pgsql")


@pytest.fixture()
def loaded_db(pg_test_db, tmp_path):
    osm_file = tmp_path / "sample.osm"
    osm_file.write_text(SAMPLE_OSM_XML, encoding="utf-8")
    env = {**os.environ, "PGPASSWORD": os.environ.get("POSTGRES_PASSWORD", "topology")}
    subprocess.run(
        [
            "osm2pgsql", "--output=flex", f"--style={STYLE_LUA}",
            f"--host={os.environ.get('POSTGRES_HOST', 'localhost')}",
            f"--port={os.environ.get('POSTGRES_PORT', '5432')}",
            f"--user={os.environ.get('POSTGRES_USER', 'topology')}",
            f"--database={pg_test_db.info.dbname}", str(osm_file),
        ],
        check=True, capture_output=True, text=True, env=env,
    )
    return pg_test_db


def test_fetch_features_returns_nearby_building_with_tags_and_geometry(loaded_db):
    features = fetch_features_in_buffer(loaded_db, CENTER_LON, CENTER_LAT, radius_m=500.0)
    buildings = [f for f in features if f.layer == "osm_buildings"]

    assert len(buildings) == 1
    feature = buildings[0]
    assert feature.osm_id == 100
    assert feature.osm_type == "W"
    assert feature.tags == {"building": "yes", "building:levels": "5"}
    assert feature.geometry.geom_type == "Polygon"
    assert feature.geometry.is_valid


def test_fetch_features_excludes_distant_building(loaded_db):
    features = fetch_features_in_buffer(loaded_db, CENTER_LON, CENTER_LAT, radius_m=500.0)
    ids = {f.osm_id for f in features if f.layer == "osm_buildings"}
    assert 200 not in ids


def test_fetch_features_includes_distant_building_with_large_radius(loaded_db):
    features = fetch_features_in_buffer(loaded_db, CENTER_LON, CENTER_LAT, radius_m=3000.0)
    ids = {f.osm_id for f in features if f.layer == "osm_buildings"}
    assert {100, 200} <= ids


def test_fetch_features_returns_power_pole_with_point_geometry(loaded_db):
    features = fetch_features_in_buffer(loaded_db, CENTER_LON, CENTER_LAT, radius_m=500.0)
    poles = [f for f in features if f.layer == "osm_power"]
    assert len(poles) == 1
    assert poles[0].geometry.geom_type == "Point"
    assert poles[0].tags["voltage"] == "10000"


def test_fetch_features_rejects_unknown_table(loaded_db):
    with pytest.raises(ValueError):
        fetch_features_in_buffer(loaded_db, CENTER_LON, CENTER_LAT, radius_m=500.0, tables=["not_a_table"])


def test_fetch_features_respects_custom_table_subset(loaded_db):
    features = fetch_features_in_buffer(
        loaded_db, CENTER_LON, CENTER_LAT, radius_m=3000.0, tables=["osm_power"]
    )
    assert {f.layer for f in features} == {"osm_power"}
