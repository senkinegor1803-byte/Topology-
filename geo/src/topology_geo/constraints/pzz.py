"""Территориальные зоны ПЗЗ (Шаг 3.3, п. 1: «для участка определить
территориальную зону ПЗЗ и её параметры: ВРИ, предельная высота или
этажность, процент застройки, отступы»).

Отдельная таблица от `constraints.store.constraint_zones` (не тот же
`zone_type="ПЗЗ"`, что мог бы завестись там) — параметры ПЗЗ типизированы
(высота, %, отступ как числа), а не свободный текст `regime`: Шагу 3.3
нужно СЧИТАТЬ по ним (`envelope.py`), а не только показать в карточке."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


class _Connection(Protocol):
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...


PZZ_ZONES_SCHEMA_SQL = """\
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
"""


@dataclass(frozen=True)
class PzzZone:
    zone_code: str  # например "Ж-1"
    vri: list[str]
    setback_m: float
    source_name: str
    data_timestamp: datetime
    geom_geojson: dict[str, Any]
    max_height_m: float | None = None
    max_building_percent: float | None = None
    document_basis: str | None = None
    id: int | None = None


def ensure_schema(conn: _Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(PZZ_ZONES_SCHEMA_SQL)
    conn.commit()


def load_pzz_zone(conn: _Connection, zone: PzzZone) -> int:
    import json

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pzz_zones (zone_code, vri, max_height_m, max_building_percent, "
            "setback_m, document_basis, source_name, data_timestamp, geom) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, "
            "ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))) RETURNING id",
            (zone.zone_code, zone.vri, zone.max_height_m, zone.max_building_percent,
             zone.setback_m, zone.document_basis, zone.source_name, zone.data_timestamp,
             json.dumps(zone.geom_geojson)),
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    return new_id


def find_pzz_zone_for_point(conn: _Connection, lon: float, lat: float) -> PzzZone | None:
    """Территориальная зона, покрывающая точку (участок определяется своим
    центром — действие п. 1 «для участка определить территориальную
    зону»). `None`, если точка ни в одной зоне ПЗЗ (например, зона не
    загружена для этого места) — честно, не подставляется умолчание."""
    import json

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, zone_code, vri, max_height_m, max_building_percent, setback_m, "
            "document_basis, source_name, data_timestamp, ST_AsGeoJSON(geom) FROM pzz_zones "
            "WHERE ST_Contains(geom, ST_SetSRID(ST_MakePoint(%s, %s), 4326)) LIMIT 1",
            (lon, lat),
        )
        row = cur.fetchone()
    if row is None:
        return None
    (id_, zone_code, vri, max_height_m, max_building_percent, setback_m,
     document_basis, source_name, data_timestamp, geom_geojson) = row
    return PzzZone(
        id=id_, zone_code=zone_code, vri=list(vri), max_height_m=max_height_m,
        max_building_percent=max_building_percent, setback_m=setback_m,
        document_basis=document_basis, source_name=source_name, data_timestamp=data_timestamp,
        geom_geojson=json.loads(geom_geojson),
    )
