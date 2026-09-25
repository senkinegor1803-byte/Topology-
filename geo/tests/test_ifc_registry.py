"""Тесты реестра GlobalId (Шаг 1.8, п. 2) на реальном Postgres (`pg_test_db`)."""

from __future__ import annotations

from topology_geo.ifc.registry import (
    ensure_schema,
    find_by_global_id,
    find_global_id,
    register_global_ids,
)


def test_register_and_find_global_id(pg_test_db):
    conn = pg_test_db
    ensure_schema(conn)

    entries = [
        ("osm_buildings", 100, "1abcDEF2345678901234gg"),
        ("osm_roads", 200, "2abcDEF2345678901234gh"),
    ]
    register_global_ids(conn, "model-1", entries)

    assert find_global_id(conn, "model-1", "osm_buildings", 100) == "1abcDEF2345678901234gg"
    assert find_global_id(conn, "model-1", "osm_roads", 200) == "2abcDEF2345678901234gh"
    assert find_global_id(conn, "model-1", "osm_buildings", 999) is None
    # другая модель не видит регистрацию первой
    assert find_global_id(conn, "model-2", "osm_buildings", 100) is None


def test_find_by_global_id_reverse_lookup(pg_test_db):
    conn = pg_test_db
    ensure_schema(conn)
    register_global_ids(conn, "model-1", [("osm_buildings", 42, "gid-42-abc")])

    assert find_by_global_id(conn, "gid-42-abc") == ("model-1", "osm_buildings", 42)
    assert find_by_global_id(conn, "not-there") is None


def test_register_global_ids_upserts_on_rerun(pg_test_db):
    conn = pg_test_db
    ensure_schema(conn)

    register_global_ids(conn, "model-1", [("osm_buildings", 1, "old-gid")])
    assert find_global_id(conn, "model-1", "osm_buildings", 1) == "old-gid"

    # пересборка модели с тем же исходным объектом -> новый GlobalId заменяет старый,
    # без ошибки уникальности и без дублирования строки
    register_global_ids(conn, "model-1", [("osm_buildings", 1, "new-gid")])
    assert find_global_id(conn, "model-1", "osm_buildings", 1) == "new-gid"

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM ifc_globalid_registry WHERE model_id = %s AND osm_id = %s",
            ("model-1", 1),
        )
        assert cur.fetchone()[0] == 1


def test_register_global_ids_noop_for_empty_list(pg_test_db):
    conn = pg_test_db
    ensure_schema(conn)
    register_global_ids(conn, "model-1", [])
    assert find_global_id(conn, "model-1", "osm_buildings", 1) is None
