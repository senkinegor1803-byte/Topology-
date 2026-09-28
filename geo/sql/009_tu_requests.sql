-- Реестр запросов ТУ (Шаг 3.12).
CREATE TABLE IF NOT EXISTS tu_requests (
    id BIGSERIAL PRIMARY KEY,
    organization_name TEXT NOT NULL,
    network_type TEXT NOT NULL,
    channel TEXT NOT NULL,
    sent_at DATE NOT NULL,
    deadline DATE NOT NULL,
    incoming_number TEXT,
    status TEXT NOT NULL DEFAULT 'отправлен' CHECK (status IN ('отправлен', 'получен', 'просрочен')),
    response_received_at DATE,
    connection_point GEOMETRY(Point, 4326),
    connection_params TEXT
);

CREATE INDEX IF NOT EXISTS tu_requests_status_idx ON tu_requests (status);
CREATE INDEX IF NOT EXISTS tu_requests_deadline_idx ON tu_requests (deadline);
