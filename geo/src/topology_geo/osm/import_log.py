"""Служебный журнал загрузок OSM (Шаг 1.1, п. 4: «Записывать дату данных в
служебную таблицу»).

`data_timestamp` — это дата актуальности данных источника (например, дата
выгрузки Geofabrik), а не момент запуска импорта (`imported_at`) — эти два
времени осознанно разделены, чтобы можно было отследить отставание базы от
реального состояния OSM.

`IMPORT_LOG_SCHEMA_SQL` — встроенная копия `geo/sql/001_import_log.sql`
(нужна, чтобы приложение могло создать схему само, не полагаясь на то, что
файл миграции попал в установленный пакет); тест
`geo/tests/test_osm_import_log.py::test_sql_file_matches_embedded_schema`
не даёт им разойтись.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


class _Connection(Protocol):
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...


IMPORT_LOG_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS osm_import_log (
    id BIGSERIAL PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_file TEXT NOT NULL,
    data_timestamp TIMESTAMPTZ NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes TEXT
);

CREATE INDEX IF NOT EXISTS osm_import_log_data_timestamp_idx
    ON osm_import_log (data_timestamp DESC);
"""


@dataclass(frozen=True)
class ImportLogEntry:
    source_name: str
    source_file: str
    data_timestamp: datetime
    notes: str | None = None


def ensure_schema(conn: _Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(IMPORT_LOG_SCHEMA_SQL)
    conn.commit()


def record_import(conn: _Connection, entry: ImportLogEntry) -> int:
    """Записать факт импорта, вернуть id новой строки."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO osm_import_log (source_name, source_file, data_timestamp, notes) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (entry.source_name, entry.source_file, entry.data_timestamp, entry.notes),
        )
        row = cur.fetchone()
    conn.commit()
    return row[0]


def latest_import(conn: _Connection, source_name: str) -> ImportLogEntry | None:
    """Последняя (по data_timestamp) запись для источника, если есть."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source_name, source_file, data_timestamp, notes FROM osm_import_log "
            "WHERE source_name = %s ORDER BY data_timestamp DESC LIMIT 1",
            (source_name,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return ImportLogEntry(source_name=row[0], source_file=row[1], data_timestamp=row[2], notes=row[3])
