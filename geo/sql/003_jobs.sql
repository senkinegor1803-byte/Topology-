-- Модель задач конвейера (Шаг 1.3, п. 1: GET /jobs/{id} - статус, прогресс, журнал).
-- Канонический источник — geo/src/topology_geo/jobs/store.py::JOBS_SCHEMA_SQL
-- (должны совпадать, см. geo/tests/test_jobs_sql_migrations.py).

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    center_lon DOUBLE PRECISION NOT NULL,
    center_lat DOUBLE PRECISION NOT NULL,
    radius_m DOUBLE PRECISION NOT NULL,
    layers JSONB NOT NULL,
    detail TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS job_steps (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    step_name TEXT NOT NULL,
    step_order INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    error_message TEXT,
    result JSONB,
    UNIQUE (job_id, step_name)
);

CREATE INDEX IF NOT EXISTS job_steps_job_id_idx ON job_steps (job_id, step_order);
