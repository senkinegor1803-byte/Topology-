"""Тесты `osm.import_log.all_latest_sources` (Шаг 2.10, п. 3: журнал
источников пакета выгрузки читает РЕАЛЬНЫЕ записи `osm_import_log`,
Шаг 1.1, п. 4)."""

from __future__ import annotations

from datetime import datetime, timezone

from topology_geo.osm.import_log import ImportLogEntry, all_latest_sources, ensure_schema, record_import


def test_all_latest_sources_empty_without_records(pg_test_db):
    ensure_schema(pg_test_db)
    assert all_latest_sources(pg_test_db) == []


def test_all_latest_sources_returns_one_entry_per_distinct_source(pg_test_db):
    ensure_schema(pg_test_db)
    record_import(
        pg_test_db,
        ImportLogEntry(
            source_name="OSM Приволжский ФО (Geofabrik)", source_file="perm-2026-06-01.osm.pbf",
            data_timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
    )
    record_import(
        pg_test_db,
        ImportLogEntry(
            source_name="Топосъёмка 1:500", source_file="topo.gpkg",
            data_timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
        ),
    )

    sources = {s.source_name: s for s in all_latest_sources(pg_test_db)}
    assert set(sources) == {"OSM Приволжский ФО (Geofabrik)", "Топосъёмка 1:500"}


def test_all_latest_sources_picks_most_recent_per_source(pg_test_db):
    ensure_schema(pg_test_db)
    record_import(
        pg_test_db,
        ImportLogEntry(
            source_name="OSM", source_file="perm-old.osm.pbf",
            data_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    )
    record_import(
        pg_test_db,
        ImportLogEntry(
            source_name="OSM", source_file="perm-new.osm.pbf",
            data_timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
    )

    sources = all_latest_sources(pg_test_db)
    assert len(sources) == 1
    assert sources[0].source_file == "perm-new.osm.pbf"
    assert sources[0].data_timestamp == datetime(2026, 6, 1, tzinfo=timezone.utc)
