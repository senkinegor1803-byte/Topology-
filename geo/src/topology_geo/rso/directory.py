"""Справочник РСО (ресурсоснабжающих организаций) и определение владельца
сети рядом с участком (Шаг 3.10).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol

CHANNEL_LETTER = "письмо"
CHANNEL_PERSONAL_ACCOUNT = "личный кабинет"
CHANNEL_GOSUSLUGI = "Госуслуги"
CHANNEL_MFC = "МФЦ"
APPLICATION_CHANNELS = (CHANNEL_LETTER, CHANNEL_PERSONAL_ACCOUNT, CHANNEL_GOSUSLUGI, CHANNEL_MFC)

STATUS_NEEDS_CLARIFICATION = "требует уточнения"


class _Connection(Protocol):
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...


RSO_DIRECTORY_SCHEMA_SQL = """\
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
"""


@dataclass(frozen=True)
class RsoEntry:
    network_type: str  # К/В/Т/Г/Кл, тот же словарь, что networks.dxf_import
    organization_name: str
    application_channel: str
    verified_at: date
    geom_geojson: dict[str, Any]
    contacts: str | None = None
    id: int | None = None


def ensure_schema(conn: _Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(RSO_DIRECTORY_SCHEMA_SQL)
    conn.commit()


def register_rso(conn: _Connection, entry: RsoEntry) -> int:
    import json

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO rso_directory (network_type, organization_name, contacts, application_channel, "
            "verified_at, service_area) VALUES (%s, %s, %s, %s, %s, "
            "ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))) RETURNING id",
            (entry.network_type, entry.organization_name, entry.contacts, entry.application_channel,
             entry.verified_at, json.dumps(entry.geom_geojson)),
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    return new_id


def find_rso_for_point(conn: _Connection, network_type: str, lon: float, lat: float) -> RsoEntry | None:
    """Действие п. 1-2 (последний шаг каскада): организация из справочника,
    чья зона деятельности содержит точку, для данного типа сети."""
    import json

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, network_type, organization_name, contacts, application_channel, verified_at, "
            "ST_AsGeoJSON(service_area) FROM rso_directory "
            "WHERE network_type = %s AND ST_Contains(service_area, ST_SetSRID(ST_MakePoint(%s, %s), 4326)) "
            "LIMIT 1",
            (network_type, lon, lat),
        )
        row = cur.fetchone()
    if row is None:
        return None
    id_, network_type_, organization_name, contacts, application_channel, verified_at, geom_geojson = row
    return RsoEntry(
        id=id_, network_type=network_type_, organization_name=organization_name, contacts=contacts,
        application_channel=application_channel, verified_at=verified_at, geom_geojson=json.loads(geom_geojson),
    )


@dataclass(frozen=True)
class OwnerResolution:
    network_type: str
    owner_name: str | None
    source: str  # "OSM operator" | "справочник РСО" | STATUS_NEEDS_CLARIFICATION
    application_channel: str | None = None
    contacts: str | None = None


def resolve_network_owner(
    conn: _Connection, network_type: str, lon: float, lat: float, *, osm_operator: str | None = None,
) -> OwnerResolution:
    """Действие п. 2, каскад: `operator` в OSM → правообладатель по НСПД/
    ЕГРН → зоны ЕТО/гарантирующих организаций по схемам тепло-/
    водоснабжения → справочник (п. 1). Реализованы РЕАЛЬНО первый шаг
    (тег OSM, уже приходит с выборкой Шага 1.4 — этот модуль его не
    запрашивает сам, только принимает) и последний (справочник, п. 1
    этого же шага). Средние два шага (НСПД/ЕГРН, схемы тепло-/
    водоснабжения муниципалитета) ЧЕСТНО пропущены — не реализованы: оба
    требуют живых внешних порталов, недоступных из этой сетевой среды (тот
    же класс ограничения, что Geofabrik/Overpass/НСПД в Шагах 2.12/3.1) —
    каскад не подделывает эти шаги, просто идёт от первого сразу к
    последнему, честно называя source «OSM operator» или «справочник РСО»,
    никогда не «НСПД» или «схема водоснабжения», которые не выполнялись."""
    if osm_operator:
        return OwnerResolution(network_type=network_type, owner_name=osm_operator, source="OSM operator")

    entry = find_rso_for_point(conn, network_type, lon, lat)
    if entry is not None:
        return OwnerResolution(
            network_type=network_type, owner_name=entry.organization_name, source="справочник РСО",
            application_channel=entry.application_channel, contacts=entry.contacts,
        )

    return OwnerResolution(network_type=network_type, owner_name=None, source=STATUS_NEEDS_CLARIFICATION)
