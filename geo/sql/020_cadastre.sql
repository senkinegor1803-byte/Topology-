-- Таблица для хранения кадастровых данных участков

CREATE TABLE IF NOT EXISTS cadastre_data (
    id SERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    cadastre_number VARCHAR(50) NOT NULL,
    center_lon DOUBLE PRECISION NOT NULL,
    center_lat DOUBLE PRECISION NOT NULL,
    area_m2 DOUBLE PRECISION,
    owner VARCHAR(500),
    address VARCHAR(500),
    boundary_geojson JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_id, cadastre_number)
);

CREATE INDEX IF NOT EXISTS idx_cadastre_job ON cadastre_data(job_id);
CREATE INDEX IF NOT EXISTS idx_cadastre_number ON cadastre_data(cadastre_number);
CREATE INDEX IF NOT EXISTS idx_cadastre_geom ON cadastre_data USING GIST (
    st_geomfromgeojson(boundary_geojson::text)
);

-- Таблица для логирования запросов к НСПД API
CREATE TABLE IF NOT EXISTS cadastre_api_log (
    id SERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    search_lon DOUBLE PRECISION NOT NULL,
    search_lat DOUBLE PRECISION NOT NULL,
    search_radius_m INTEGER,
    results_count INTEGER,
    error_message VARCHAR(500),
    request_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cadastre_api_log_job ON cadastre_api_log(job_id);
