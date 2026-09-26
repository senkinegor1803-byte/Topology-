-- Кеш готовых тайлов (Шаг 2.1, п. 2-3).
-- Канонический источник — geo/src/topology_geo/tiling/cache.py::TILE_CACHE_SCHEMA_SQL
-- (должны совпадать, см. geo/tests/test_tiling_sql_migrations.py).

CREATE TABLE IF NOT EXISTS tile_cache (
    id BIGSERIAL PRIMARY KEY,
    tile_key TEXT NOT NULL UNIQUE,
    zone INTEGER NOT NULL,
    tx INTEGER NOT NULL,
    ty INTEGER NOT NULL,
    layer TEXT NOT NULL,
    data_version TEXT NOT NULL,
    generator_version TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS tile_cache_zone_tx_ty_idx ON tile_cache (zone, tx, ty, layer);
