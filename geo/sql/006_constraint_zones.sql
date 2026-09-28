-- Официальные и расчётные зоны ограничений (Шаг 3.1, п. 3-4; Шаг 3.2).
CREATE TABLE IF NOT EXISTS constraint_zones (
    id BIGSERIAL PRIMARY KEY,
    zone_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('официально', 'расчётно')),
    regime TEXT,
    registry_number TEXT,
    document_basis TEXT,
    source_name TEXT NOT NULL,
    data_timestamp TIMESTAMPTZ NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    version INTEGER NOT NULL DEFAULT 1,
    superseded_at TIMESTAMPTZ,
    geom GEOMETRY(MultiPolygon, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS constraint_zones_geom_idx ON constraint_zones USING GIST (geom);
CREATE INDEX IF NOT EXISTS constraint_zones_zone_type_idx ON constraint_zones (zone_type);
CREATE INDEX IF NOT EXISTS constraint_zones_registry_number_idx ON constraint_zones (registry_number);
CREATE INDEX IF NOT EXISTS constraint_zones_current_idx ON constraint_zones (registry_number) WHERE superseded_at IS NULL;
