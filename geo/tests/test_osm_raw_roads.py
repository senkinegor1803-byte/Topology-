"""Тесты восстановления графа узлов дорог из PostGIS (Шаг 2.3, п. 1).

Критичное свойство, которое здесь проверяется: два way, делящие общий узел
на перекрёстке в исходном OSM, после `fetch_raw_roads_in_buffer` +
`build_osm_xml` должны и дальше делить этот же ID узла — иначе osm2streets
(geometry/streets.py) не увидит перекрёсток вовсе.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")

from topology_geo.osm.raw_roads import build_osm_xml, fetch_raw_roads_in_buffer  # noqa: E402

STYLE_LUA = Path(__file__).resolve().parents[1] / "osm2pgsql" / "style.lua"

CONN_PARAMS = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": os.environ.get("POSTGRES_PORT", "5432"),
    "user": os.environ.get("POSTGRES_USER", "topology"),
    "password": os.environ.get("POSTGRES_PASSWORD", "topology"),
}

# Перекрёсток из 4 улиц, сходящихся в узле 1 (та же геометрия, что смоук-тест
# при разработке этого шага) - реальная проверка того, что топология не
# теряется между исходным OSM и osm_roads.nodes.
INTERSECTION_OSM_XML = """\
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
    <tag k="name" v="East &amp; Co. St"/>
  </way>
  <way id="12" version="1">
    <nd ref="1"/><nd ref="4"/>
    <tag k="highway" v="residential"/>
    <tag k="lanes" v="2"/>
  </way>
  <way id="13" version="1">
    <nd ref="1"/><nd ref="5"/>
    <tag k="highway" v="residential"/>
    <tag k="lanes" v="2"/>
  </way>
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
    reason="требуется установленный osm2pgsql и доступный Postgres+PostGIS",
)


@pytest.fixture()
def intersection_db(pg_test_db, tmp_path):
    osm_file = tmp_path / "intersection.osm"
    osm_file.write_text(INTERSECTION_OSM_XML, encoding="utf-8")
    dbname = pg_test_db.info.dbname
    subprocess.run(
        [
            "osm2pgsql", "--output=flex", f"--style={STYLE_LUA}",
            f"--host={CONN_PARAMS['host']}", f"--port={CONN_PARAMS['port']}",
            f"--user={CONN_PARAMS['user']}", f"--database={dbname}", str(osm_file),
        ],
        check=True, capture_output=True, text=True,
        env={**os.environ, "PGPASSWORD": CONN_PARAMS["password"]},
    )
    return pg_test_db


def test_fetch_raw_roads_returns_node_ids_matching_geometry_order(intersection_db):
    roads = fetch_raw_roads_in_buffer(intersection_db, lon=56.243, lat=58.0105, radius_m=1000.0)
    assert len(roads) == 4

    by_id = {r.osm_id: r for r in roads}
    assert by_id[10].node_ids == [1, 2]
    assert by_id[11].node_ids == [1, 3]
    assert list(by_id[10].geometry.coords) == [(56.243, 58.0105), (56.243, 58.0125)]
    assert by_id[11].tags["name"] == "East & Co. St"  # спецсимвол пережил jsonb-хранение


def test_fetch_raw_roads_respects_buffer_radius(intersection_db):
    near = fetch_raw_roads_in_buffer(intersection_db, lon=56.243, lat=58.0105, radius_m=1.0, margin_m=1.0)
    far = fetch_raw_roads_in_buffer(intersection_db, lon=56.243, lat=58.0105, radius_m=1000.0)
    assert len(near) <= len(far)
    assert len(far) == 4


def test_build_osm_xml_preserves_shared_intersection_node(intersection_db):
    roads = fetch_raw_roads_in_buffer(intersection_db, lon=56.243, lat=58.0105, radius_m=1000.0)
    xml_text = build_osm_xml(roads)

    root = ET.fromstring(xml_text)
    node_ids = [n.attrib["id"] for n in root.findall("node")]
    way_ids = [w.attrib["id"] for w in root.findall("way")]
    assert sorted(node_ids, key=int) == ["1", "2", "3", "4", "5"]
    assert sorted(way_ids, key=int) == ["10", "11", "12", "13"]

    # узел "1" - общий центр перекрёстка - встречается как nd ref в ролях
    # первого узла у ВСЕХ четырёх way (топология сохранена).
    for way in root.findall("way"):
        refs = [nd.attrib["ref"] for nd in way.findall("nd")]
        assert refs[0] == "1"

    node_1 = next(n for n in root.findall("node") if n.attrib["id"] == "1")
    assert node_1.attrib["lat"] == "58.0105"
    assert node_1.attrib["lon"] == "56.243"

    # все узлы объявлены раньше всех way (порядок важен для большинства
    # OSM-парсеров, тот же приём, что и синтетические .osm фикстуры Шага 1.1).
    tags_in_order = [child.tag for child in root]
    last_node_index = max(i for i, t in enumerate(tags_in_order) if t == "node")
    first_way_index = min(i for i, t in enumerate(tags_in_order) if t == "way")
    assert last_node_index < first_way_index

    way_with_amp = next(w for w in root.findall("way") if w.attrib["id"] == "11")
    name_tag = next(t for t in way_with_amp.findall("tag") if t.attrib["k"] == "name")
    assert name_tag.attrib["v"] == "East & Co. St"


def test_build_osm_xml_empty_input_produces_valid_empty_osm():
    xml_text = build_osm_xml([])
    root = ET.fromstring(xml_text)
    assert root.tag == "osm"
    assert root.findall("node") == []
    assert root.findall("way") == []
