-- Связь исходных объектов OSM с их GlobalId в собранной IFC-модели (Шаг 1.8, п. 2).
-- Канонический источник — geo/src/topology_geo/ifc/registry.py::IFC_GLOBALID_REGISTRY_SCHEMA_SQL
-- (должны совпадать, см. geo/tests/test_ifc_registry_sql_migrations.py).

CREATE TABLE IF NOT EXISTS ifc_globalid_registry (
    id BIGSERIAL PRIMARY KEY,
    model_id TEXT NOT NULL,
    layer TEXT NOT NULL,
    osm_id BIGINT NOT NULL,
    global_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_id, layer, osm_id)
);

CREATE INDEX IF NOT EXISTS ifc_globalid_registry_model_idx ON ifc_globalid_registry (model_id);
CREATE INDEX IF NOT EXISTS ifc_globalid_registry_global_id_idx ON ifc_globalid_registry (global_id);
