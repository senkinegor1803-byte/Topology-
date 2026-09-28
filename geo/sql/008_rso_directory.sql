-- Справочник РСО (ресурсоснабжающих организаций) - Шаг 3.10, п. 1.
CREATE TABLE IF NOT EXISTS rso_directory (
    id BIGSERIAL PRIMARY KEY,
    network_type TEXT NOT NULL,
    organization_name TEXT NOT NULL,
    contacts TEXT,
    application_channel TEXT NOT NULL CHECK (application_channel IN ('письмо', 'личный кабинет', 'Госуслуги', 'МФЦ')),
    verified_at DATE NOT NULL,
    service_area GEOMETRY(MultiPolygon, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS rso_directory_service_area_idx ON rso_directory USING GIST (service_area);
CREATE INDEX IF NOT EXISTS rso_directory_network_type_idx ON rso_directory (network_type);
