-- Индекс покрытия рельефа (Шаг 1.2, п. 4).
-- Канонический источник — geo/src/topology_geo/relief/coverage.py::DEM_COVERAGE_SCHEMA_SQL
-- (должны совпадать, см. geo/tests/test_osm_sql_migrations.py-подобный тест
-- geo/tests/test_relief_sql_migrations.py).

CREATE TABLE IF NOT EXISTS dem_coverage (
    id BIGSERIAL PRIMARY KEY,
    source_name TEXT NOT NULL,
    priority INTEGER NOT NULL,
    storage_key TEXT NOT NULL,
    resolution_m DOUBLE PRECISION NOT NULL,
    data_timestamp TIMESTAMPTZ NOT NULL,
    geom GEOMETRY(Polygon, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS dem_coverage_geom_idx ON dem_coverage USING GIST (geom);
CREATE INDEX IF NOT EXISTS dem_coverage_priority_idx ON dem_coverage (priority DESC);
