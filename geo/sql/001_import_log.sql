-- Служебный журнал загрузок OSM (Шаг 1.1, п. 4 плана).
-- Канонический источник — geo/src/topology_geo/osm/import_log.py::IMPORT_LOG_SCHEMA_SQL
-- (должны совпадать, см. geo/tests/test_osm_import_log.py::test_sql_file_matches_embedded_schema).

CREATE TABLE IF NOT EXISTS osm_import_log (
    id BIGSERIAL PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_file TEXT NOT NULL,
    data_timestamp TIMESTAMPTZ NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes TEXT
);

CREATE INDEX IF NOT EXISTS osm_import_log_data_timestamp_idx
    ON osm_import_log (data_timestamp DESC);
