-- API-ключи и вебхуки для внешних интеграций (Шаг 4.11).
-- Канонический источник — geo/src/topology_geo/auth/integrations.py::INTEGRATIONS_SCHEMA_SQL
-- (должны совпадать, см. geo/tests/test_integrations_sql_migrations.py).
-- Требует уже существующую таблицу users (миграция 010).

CREATE TABLE IF NOT EXISTS api_keys (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS api_keys_user_id_idx ON api_keys (user_id);

CREATE TABLE IF NOT EXISTS webhooks (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS webhooks_user_id_idx ON webhooks (user_id);
