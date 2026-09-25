"""Интеграционный тест оркестрации выборки участка (Шаг 1.4): выборка ->
репроекция -> обрезка -> нормализация целиком, на реальном osm2pgsql+PostGIS."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from topology_geo.selection.service import select_site_data

STYLE_LUA = Path(__file__).resolve().parents[1] / "osm2pgsql" / "style.lua"

CENTER_LON, CENTER_LAT = 56.2421, 58.01005
ZONE = 2

# Здание рядом с центром (~15 м), здание далеко (~1.1 км, вне радиуса 500 м),
# дорога, пересекающая границу круга (должна быть обрезана), дерево без species.
SAMPLE_OSM_XML = """\
<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="topology-test">
  <node id="1" lat="58.0100" lon="56.2420" version="1"/>
  <node id="2" lat="58.0100" lon="56.2423" version="1"/>
  <node id="3" lat="58.0102" lon="56.2423" version="1"/>
  <node id="4" lat="58.0102" lon="56.2420" version="1"/>
  <node id="10" lat="58.0200" lon="56.2420" version="1"/>
  <node id="11" lat="58.0200" lon="56.2422" version="1"/>
  <node id="12" lat="58.0201" lon="56.2422" version="1"/>
  <node id="13" lat="58.0201" lon="56.2420" version="1"/>
  <node id="30" lat="58.0000" lon="56.2000" version="1"/>
  <node id="31" lat="58.0200" lon="56.2800" version="1"/>
  <node id="40" lat="58.0105" lon="56.2425" version="1">
    <tag k="natural" v="tree"/>
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
  <way id="300" version="1">
    <nd ref="30"/><nd ref="31"/>
    <tag k="highway" v="residential"/>
    <tag k="surface" v="asphalt"/>
    <tag k="lanes" v="2"/>
  </way>
</osm>
"""


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


def test_select_site_data_includes_nearby_excludes_far_building(loaded_db):
    dataset = select_site_data(loaded_db, CENTER_LON, CENTER_LAT, radius_m=500.0, zone=ZONE)
    building_ids = {f.osm_id for f in dataset.features if f.layer == "osm_buildings"}
    assert 100 in building_ids
    assert 200 not in building_ids


def test_select_site_data_localizes_geometry_around_center(loaded_db):
    dataset = select_site_data(loaded_db, CENTER_LON, CENTER_LAT, radius_m=500.0, zone=ZONE)
    building = next(f for f in dataset.features if f.osm_id == 100)

    # локализованная геометрия участка должна лежать в пределах его радиуса
    # вокруг (0,0), а не в "мировых" координатах МСК-59 (порядка 10^6)
    minx, miny, maxx, maxy = building.geometry.bounds
    assert abs(minx) < 500.0
    assert abs(maxy) < 500.0


def test_select_site_data_normalizes_building_attributes(loaded_db):
    dataset = select_site_data(loaded_db, CENTER_LON, CENTER_LAT, radius_m=500.0, zone=ZONE)
    building = next(f for f in dataset.features if f.osm_id == 100)
    assert building.attributes["type"] == "yes"
    assert building.attributes["levels"] == 5
    assert building.confidence["levels"] == "факт"


def test_select_site_data_clips_road_crossing_boundary(loaded_db):
    dataset = select_site_data(loaded_db, CENTER_LON, CENTER_LAT, radius_m=500.0, zone=ZONE)
    roads = [f for f in dataset.features if f.layer == "osm_roads"]
    assert len(roads) == 1
    road = roads[0]
    # исходная дорога длиной несколько км, после обрезки кругом 500 м не длиннее диаметра
    assert road.geometry.length <= 1001.0
    assert road.attributes["surface"] == "asphalt"
    assert road.attributes["lanes"] == 2


def test_select_site_data_marks_missing_species_as_default(loaded_db):
    dataset = select_site_data(loaded_db, CENTER_LON, CENTER_LAT, radius_m=500.0, zone=ZONE)
    trees = [f for f in dataset.features if f.layer == "osm_vegetation"]
    assert len(trees) == 1
    assert trees[0].attributes["species"] is None
    assert trees[0].confidence["species"] == "умолчание"


def test_select_site_data_auto_detects_zone_when_not_given(loaded_db):
    dataset = select_site_data(loaded_db, CENTER_LON, CENTER_LAT, radius_m=500.0)
    assert dataset.zone == ZONE
