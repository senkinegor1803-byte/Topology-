"""Индекс покрытия рельефа в PostGIS (Шаг 1.2, п. 4).

Хранит, какой источник (топосъёмка, TessaDEM, ...) покрывает какую область,
с каким приоритетом и под каким ключом лежит в объектном хранилище (MinIO) —
`service.get_dem` использует это, чтобы найти и слить нужные растры для bbox.

`DEM_COVERAGE_SCHEMA_SQL` — встроенная копия `geo/sql/002_dem_coverage.sql`
(тот же приём, что и в `topology_geo.osm.import_log`, синхронность проверена
тестом `geo/tests/test_relief_sql_migrations.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


class _Connection(Protocol):
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...


DEM_COVERAGE_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS dem_coverage (
    id BIGSERIAL PRIMARY KEY,
    source_name TEXT NOT NULL,
    priority INTEGER NOT NULL,
    storage_key TEXT NOT NULL,
    resolution_m DOUBLE PRECISION NOT NULL,
    data_timestamp TIMESTAMPTZ NOT NULL,
    geom GEOMETRY(Polygon, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS dem_coverage_geom_idx ON dem_coverage USING GIST (geom);
CREATE INDEX IF NOT EXISTS dem_coverage_priority_idx ON dem_coverage (priority DESC);
"""


@dataclass(frozen=True)
class CoverageEntry:
    source_name: str
    priority: int
    storage_key: str
    resolution_m: float
    data_timestamp: datetime
    footprint_wkt: str  # WGS-84 (EPSG:4326), например из shapely geometry.wkt
    id: int | None = None


def ensure_schema(conn: _Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(DEM_COVERAGE_SCHEMA_SQL)
    conn.commit()


def register_coverage(conn: _Connection, entry: CoverageEntry) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO dem_coverage (source_name, priority, storage_key, resolution_m, "
            "data_timestamp, geom) VALUES (%s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326)) "
            "RETURNING id",
            (
                entry.source_name,
                entry.priority,
                entry.storage_key,
                entry.resolution_m,
                entry.data_timestamp,
                entry.footprint_wkt,
            ),
        )
        row = cur.fetchone()
    conn.commit()
    return row[0]


def find_coverage(conn: _Connection, bbox: tuple[float, float, float, float]) -> list[CoverageEntry]:
    """Все покрытия, пересекающие `bbox` (minx, miny, maxx, maxy, EPSG:4326),
    от наивысшего приоритета к наименьшему (топосъёмка раньше TessaDEM)."""
    minx, miny, maxx, maxy = bbox
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, source_name, priority, storage_key, resolution_m, data_timestamp, "
            "ST_AsText(geom) FROM dem_coverage "
            "WHERE ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 4326)) "
            "ORDER BY priority DESC",
            (minx, miny, maxx, maxy),
        )
        rows = cur.fetchall()
    return [
        CoverageEntry(
            id=row[0],
            source_name=row[1],
            priority=row[2],
            storage_key=row[3],
            resolution_m=row[4],
            data_timestamp=row[5],
            footprint_wkt=row[6],
        )
        for row in rows
    ]
