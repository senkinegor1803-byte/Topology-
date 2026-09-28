"""Реестр запросов и привязка ТУ (Шаг 3.12).

Срок ответа по умолчанию — 30 календарных дней (ГрК РФ, ст. 48, ч. 7:
«срок предоставления технических условий... не может превышать 30 дней»)
— реальная норма, не произвольное число; вызывающая сторона может передать
другой срок, если для конкретной сети/организации в её данных указан
иной регламентный срок.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Protocol

DEFAULT_RESPONSE_DEADLINE_DAYS = 30

STATUS_SENT = "отправлен"
STATUS_RECEIVED = "получен"
STATUS_OVERDUE = "просрочен"


class _Connection(Protocol):
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...


TU_REQUESTS_SCHEMA_SQL = """\
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
"""


@dataclass(frozen=True)
class TuRequest:
    organization_name: str
    network_type: str
    channel: str
    sent_at: date
    deadline: date
    status: str = STATUS_SENT
    incoming_number: str | None = None
    response_received_at: date | None = None
    connection_point_geojson: dict[str, Any] | None = None
    connection_params: str | None = None
    id: int | None = None


def ensure_schema(conn: _Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(TU_REQUESTS_SCHEMA_SQL)
    conn.commit()


def register_tu_request(
    conn: _Connection, *, organization_name: str, network_type: str, channel: str,
    sent_at: date, response_deadline_days: int = DEFAULT_RESPONSE_DEADLINE_DAYS,
) -> int:
    """Действие п. 1: реестр (организация, дата, канал...). `deadline`
    считается сразу при регистрации (`sent_at + response_deadline_days`),
    не хранится готовой строкой откуда-то ещё — единый источник правды для
    напоминаний (`overdue_requests`)."""
    deadline = sent_at + timedelta(days=response_deadline_days)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO tu_requests (organization_name, network_type, channel, sent_at, deadline, status) "
            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (organization_name, network_type, channel, sent_at, deadline, STATUS_SENT),
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    return new_id


def record_response(
    conn: _Connection, request_id: int, *, incoming_number: str, received_at: date,
    connection_point_geojson: dict[str, Any] | None = None, connection_params: str | None = None,
) -> None:
    """Действие: «загрузка полученных ТУ; точка подключения и параметры из
    ТУ отображаются на сети в модели» — `connection_point`/`connection_
    params` пишутся сюда же, откуда их сможет прочитать модель (тот же
    принцип, что и остальные геослои проекта: один источник в PostGIS)."""
    import json

    with conn.cursor() as cur:
        if connection_point_geojson is not None:
            cur.execute(
                "UPDATE tu_requests SET status = %s, incoming_number = %s, response_received_at = %s, "
                "connection_point = ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), connection_params = %s "
                "WHERE id = %s",
                (STATUS_RECEIVED, incoming_number, received_at, json.dumps(connection_point_geojson),
                 connection_params, request_id),
            )
        else:
            cur.execute(
                "UPDATE tu_requests SET status = %s, incoming_number = %s, response_received_at = %s, "
                "connection_params = %s WHERE id = %s",
                (STATUS_RECEIVED, incoming_number, received_at, connection_params, request_id),
            )
    conn.commit()


def _row_to_request(row) -> TuRequest:
    import json

    (id_, organization_name, network_type, channel, sent_at, deadline, incoming_number, status,
     response_received_at, connection_point_geojson, connection_params) = row
    return TuRequest(
        id=id_, organization_name=organization_name, network_type=network_type, channel=channel,
        sent_at=sent_at, deadline=deadline, incoming_number=incoming_number, status=status,
        response_received_at=response_received_at,
        connection_point_geojson=json.loads(connection_point_geojson) if connection_point_geojson else None,
        connection_params=connection_params,
    )


def overdue_requests(conn: _Connection, *, as_of: date | None = None) -> list[TuRequest]:
    """Действие: «напоминания о сроках ответа» — запросы, чей срок истёк, а
    ответ так и не получен. НЕ меняет `status` в базе на 'просрочен' сама
    (это отдельное, явное действие пользователя/крон-задачи — иначе один
    вызов этой функции для отчёта тихо менял бы состояние реестра, что
    честнее делать явным шагом)."""
    as_of = as_of or date.today()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, organization_name, network_type, channel, sent_at, deadline, incoming_number, status, "
            "response_received_at, ST_AsGeoJSON(connection_point), connection_params "
            "FROM tu_requests WHERE status = %s AND deadline < %s ORDER BY deadline",
            (STATUS_SENT, as_of),
        )
        rows = cur.fetchall()
    return [_row_to_request(r) for r in rows]


def mark_overdue(conn: _Connection, *, as_of: date | None = None) -> int:
    """Явно пометить просроченные запросы `status='просрочен'` (действие
    п. 1, статус — часть реестра). Возвращает число обновлённых строк."""
    as_of = as_of or date.today()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE tu_requests SET status = %s WHERE status = %s AND deadline < %s",
            (STATUS_OVERDUE, STATUS_SENT, as_of),
        )
        updated = cur.rowcount
    conn.commit()
    return updated


def get_tu_request(conn: _Connection, request_id: int) -> TuRequest | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, organization_name, network_type, channel, sent_at, deadline, incoming_number, status, "
            "response_received_at, ST_AsGeoJSON(connection_point), connection_params "
            "FROM tu_requests WHERE id = %s",
            (request_id,),
        )
        row = cur.fetchone()
    return _row_to_request(row) if row is not None else None
