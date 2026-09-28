-- Территориальные зоны ПЗЗ (Шаг 3.3, п. 1).
CREATE TABLE IF NOT EXISTS pzz_zones (
    id BIGSERIAL PRIMARY KEY,
    zone_code TEXT NOT NULL,
    vri TEXT[] NOT NULL,
    max_height_m DOUBLE PRECISION,
    max_building_percent DOUBLE PRECISION,
    setback_m DOUBLE PRECISION NOT NULL,
    document_basis TEXT,
    source_name TEXT NOT NULL,
    data_timestamp TIMESTAMPTZ NOT NULL,
    geom GEOMETRY(MultiPolygon, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS pzz_zones_geom_idx ON pzz_zones USING GIST (geom);
CREATE INDEX IF NOT EXISTS pzz_zones_zone_code_idx ON pzz_zones (zone_code);
