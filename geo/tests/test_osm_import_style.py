"""Интеграционные тесты флекс-стиля osm2pgsql, журнала импорта и запроса
выборки (Шаг 1.1).

Требуют реальный Postgres+PostGIS (переменные POSTGRES_HOST/PORT/USER/PASSWORD,
как в `topology_geo.devcheck`) и бинарь `osm2pgsql` на PATH. При отсутствии
любого из них модуль целиком пропускается — так `pytest` без системных
пакетов (например, окружение без `apt-get install postgresql osm2pgsql`) не
падает. В CI (`.github/workflows/ci.yml`) оба доступны и тесты реально
прогоняются против настоящего osm2pgsql.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")

from topology_geo.osm.import_log import ImportLogEntry, ensure_schema, latest_import, record_import  # noqa: E402
from topology_geo.osm.queries import TABLES, count_within_radius  # noqa: E402

STYLE_LUA = Path(__file__).resolve().parents[1] / "osm2pgsql" / "style.lua"

CONN_PARAMS = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": os.environ.get("POSTGRES_PORT", "5432"),
    "user": os.environ.get("POSTGRES_USER", "topology"),
    "password": os.environ.get("POSTGRES_PASSWORD", "topology"),
}

SAMPLE_OSM_XML = """\
<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="topology-test">
  <node id="1" lat="58.0010" lon="56.2010" version="1"/>
  <node id="2" lat="58.0010" lon="56.2020" version="1"/>
  <node id="3" lat="58.0020" lon="56.2020" version="1"/>
  <node id="4" lat="58.0020" lon="56.2010" version="1"/>

  <node id="10" lat="58.0000" lon="56.2000" version="1"/>
  <node id="11" lat="58.0005" lon="56.2005" version="1"/>

  <node id="20" lat="58.0030" lon="56.2000" version="1"/>
  <node id="21" lat="58.0035" lon="56.2010" version="1"/>

  <node id="30" lat="58.0040" lon="56.2030" version="1"/>
  <node id="31" lat="58.0040" lon="56.2040" version="1"/>
  <node id="32" lat="58.0050" lon="56.2040" version="1"/>
  <node id="33" lat="58.0050" lon="56.2030" version="1"/>

  <node id="40" lat="58.0045" lon="56.2050" version="1"/>
  <node id="41" lat="58.0050" lon="56.2055" version="1"/>

  <node id="50" lat="58.0015" lon="56.2050" version="1">
    <tag k="natural" v="tree"/>
    <tag k="species" v="Betula pendula"/>
  </node>
  <node id="51" lat="58.0060" lon="56.2000" version="1">
    <tag k="power" v="pole"/>
    <tag k="voltage" v="10000"/>
  </node>
  <node id="52" lat="58.0002" lon="56.2002" version="1">
    <tag k="amenity" v="bench"/>
  </node>

  <node id="60" lat="58.0060" lon="56.2000" version="1"/>
  <node id="61" lat="58.0060" lon="56.2020" version="1"/>

  <node id="70" lat="58.0070" lon="56.2000" version="1"/>
  <node id="71" lat="58.0070" lon="56.2020" version="1"/>
  <node id="72" lat="58.0080" lon="56.2020" version="1"/>
  <node id="73" lat="58.0080" lon="56.2000" version="1"/>

  <node id="80" lat="58.0090" lon="56.2000" version="1"/>
  <node id="81" lat="58.0090" lon="56.2020" version="1"/>
  <node id="82" lat="58.0100" lon="56.2020" version="1"/>
  <node id="83" lat="58.0100" lon="56.2000" version="1"/>
  <node id="84" lat="58.0093" lon="56.2005" version="1"/>
  <node id="85" lat="58.0093" lon="56.2015" version="1"/>
  <node id="86" lat="58.0097" lon="56.2015" version="1"/>
  <node id="87" lat="58.0097" lon="56.2005" version="1"/>

  <way id="100" version="1">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
    <tag k="building" v="yes"/>
    <tag k="building:levels" v="5"/>
  </way>
  <way id="101" version="1">
    <nd ref="10"/><nd ref="11"/>
    <tag k="highway" v="residential"/>
    <tag k="surface" v="asphalt"/>
    <tag k="lanes" v="2"/>
  </way>
  <way id="102" version="1">
    <nd ref="20"/><nd ref="21"/>
    <tag k="railway" v="rail"/>
  </way>
  <way id="103" version="1">
    <nd ref="30"/><nd ref="31"/><nd ref="32"/><nd ref="33"/><nd ref="30"/>
    <tag k="natural" v="water"/>
  </way>
  <way id="104" version="1">
    <nd ref="40"/><nd ref="41"/>
    <tag k="waterway" v="stream"/>
  </way>
  <way id="105" version="1">
    <nd ref="60"/><nd ref="61"/>
    <tag k="power" v="line"/>
    <tag k="voltage" v="10000"/>
  </way>
  <way id="106" version="1">
    <nd ref="70"/><nd ref="71"/><nd ref="72"/><nd ref="73"/><nd ref="70"/>
    <tag k="natural" v="wood"/>
  </way>
  <way id="110" version="1">
    <nd ref="80"/><nd ref="81"/><nd ref="82"/><nd ref="83"/><nd ref="80"/>
  </way>
  <way id="111" version="1">
    <nd ref="84"/><nd ref="85"/><nd ref="86"/><nd ref="87"/><nd ref="84"/>
  </way>

  <relation id="200" version="1">
    <member type="way" ref="110" role="outer"/>
    <member type="way" ref="111" role="inner"/>
    <tag k="type" v="multipolygon"/>
    <tag k="building" v="yes"/>
    <tag k="building:levels" v="9"/>
  </relation>
</osm>
"""


def _osm2pgsql_available() -> bool:
    return shutil.which("osm2pgsql") is not None


def _postgres_reachable() -> bool:
    try:
        conn = psycopg.connect(dbname="postgres", connect_timeout=3, **CONN_PARAMS)
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (_osm2pgsql_available() and _postgres_reachable()),
    reason="требуется установленный osm2pgsql и доступный Postgres+PostGIS (см. docstring модуля)",
)


@pytest.fixture()
def osm_test_db(tmp_path):
    db_name = f"topology_test_{uuid.uuid4().hex[:8]}"

    admin_conn = psycopg.connect(dbname="postgres", autocommit=True, **CONN_PARAMS)
    try:
        admin_conn.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        admin_conn.close()

    conn = psycopg.connect(dbname=db_name, autocommit=True, **CONN_PARAMS)
    conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    osm_file = tmp_path / "sample.osm"
    osm_file.write_text(SAMPLE_OSM_XML, encoding="utf-8")

    subprocess.run(
        [
            "osm2pgsql",
            "--output=flex",
            f"--style={STYLE_LUA}",
            f"--host={CONN_PARAMS['host']}",
            f"--port={CONN_PARAMS['port']}",
            f"--user={CONN_PARAMS['user']}",
            f"--database={db_name}",
            str(osm_file),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PGPASSWORD": CONN_PARAMS["password"]},
    )

    try:
        yield conn
    finally:
        conn.close()
        admin_conn = psycopg.connect(dbname="postgres", autocommit=True, **CONN_PARAMS)
        try:
            admin_conn.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            admin_conn.close()


def _fetchall(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def test_all_layer_tables_created(osm_test_db):
    for table in TABLES:
        rows = _fetchall(osm_test_db, f"SELECT count(*) FROM {table}")  # noqa: S608 - таблица из TABLES
        assert rows[0][0] >= 0  # таблица существует и читается


def test_building_way_and_tags(osm_test_db):
    rows = _fetchall(
        osm_test_db,
        "SELECT tags, ST_GeometryType(geom) FROM osm_buildings WHERE osm_id = 100",
    )
    assert len(rows) == 1
    tags, geom_type = rows[0]
    assert tags == {"building": "yes", "building:levels": "5"}
    assert geom_type == "ST_Polygon"


def test_multipolygon_building_has_hole(osm_test_db):
    rows = _fetchall(
        osm_test_db,
        "SELECT ST_NumInteriorRings(geom), tags->>'building:levels' FROM osm_buildings WHERE osm_id = 200",
    )
    assert len(rows) == 1
    interior_rings, levels = rows[0]
    assert interior_rings == 1
    assert levels == "9"


def test_road_water_railway_present(osm_test_db):
    assert _fetchall(osm_test_db, "SELECT tags FROM osm_roads WHERE osm_id = 101")[0][0] == {
        "highway": "residential",
        "surface": "asphalt",
        "lanes": "2",
    }
    assert _fetchall(osm_test_db, "SELECT tags FROM osm_railways WHERE osm_id = 102")[0][0] == {"railway": "rail"}
    assert _fetchall(osm_test_db, "SELECT tags FROM osm_water_areas WHERE osm_id = 103")[0][0] == {
        "natural": "water"
    }
    assert _fetchall(osm_test_db, "SELECT tags FROM osm_waterways WHERE osm_id = 104")[0][0] == {
        "waterway": "stream"
    }


def test_vegetation_power_landscaping_present(osm_test_db):
    veg = {r[0]: r[1] for r in _fetchall(osm_test_db, "SELECT osm_id, tags FROM osm_vegetation")}
    assert veg[50] == {"natural": "tree", "species": "Betula pendula"}
    assert veg[106] == {"natural": "wood"}

    power = {r[0]: r[1] for r in _fetchall(osm_test_db, "SELECT osm_id, tags FROM osm_power")}
    assert power[51]["power"] == "pole"
    assert power[105]["power"] == "line"

    landscaping = _fetchall(osm_test_db, "SELECT tags FROM osm_landscaping WHERE osm_id = 52")
    assert landscaping[0][0] == {"amenity": "bench"}


def test_geom_columns_have_gist_index(osm_test_db):
    rows = _fetchall(
        osm_test_db,
        "SELECT indexdef FROM pg_indexes WHERE tablename = 'osm_buildings' AND indexname LIKE '%geom%'",
    )
    assert rows, "нет индекса на geom у osm_buildings"
    assert "gist" in rows[0][0].lower()


def test_count_within_radius_finds_only_nearby_building(osm_test_db):
    counts_near = count_within_radius(osm_test_db, lon=56.201, lat=58.001, radius_m=200)
    assert counts_near["osm_buildings"] == 1  # только way 100, не мультиполигон 200 (далеко)

    counts_far = count_within_radius(osm_test_db, lon=56.201, lat=58.001, radius_m=3000)
    assert counts_far["osm_buildings"] == 2  # оба здания попадают в 3 км


def test_count_within_radius_rejects_unknown_table(osm_test_db):
    with pytest.raises(ValueError):
        count_within_radius(osm_test_db, lon=56.2, lat=58.0, radius_m=500, tables=["not_a_real_table"])


def test_import_log_roundtrip(osm_test_db):
    ensure_schema(osm_test_db)
    entry = ImportLogEntry(
        source_name="Geofabrik Приволжский ФО (тест)",
        source_file="sample.osm",
        data_timestamp=datetime(2026, 9, 20, tzinfo=timezone.utc),
        notes="интеграционный тест Шага 1.1",
    )
    row_id = record_import(osm_test_db, entry)
    assert row_id > 0

    fetched = latest_import(osm_test_db, entry.source_name)
    assert fetched is not None
    assert fetched.source_file == "sample.osm"
    assert fetched.data_timestamp == entry.data_timestamp
    assert fetched.notes == entry.notes


def test_import_log_returns_none_for_unknown_source(osm_test_db):
    ensure_schema(osm_test_db)
    assert latest_import(osm_test_db, "неизвестный источник") is None
