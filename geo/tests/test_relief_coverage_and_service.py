"""Интеграционные тесты индекса покрытия рельефа и сервиса get_dem (Шаг 1.2, п. 4).

Требуют реальный Postgres+PostGIS (см. `topology_geo.devcheck` / те же
переменные, что и `test_osm_import_style.py`); при недоступности БД модуль
целиком пропускается.
"""

from __future__ import annotations

import io
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.transform import from_origin

psycopg = pytest.importorskip("psycopg")

from topology_geo.coords import MSK59_ZONES, wgs84_to_msk59  # noqa: E402
from topology_geo.relief.coverage import (  # noqa: E402
    CoverageEntry,
    ensure_schema,
    find_coverage,
    register_coverage,
)
from topology_geo.relief.service import Grid, get_dem  # noqa: E402

CONN_PARAMS = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": os.environ.get("POSTGRES_PORT", "5432"),
    "user": os.environ.get("POSTGRES_USER", "topology"),
    "password": os.environ.get("POSTGRES_PASSWORD", "topology"),
}

ZONE = 2
PERM_LON, PERM_LAT = 56.2431, 58.0105

LOW_PRIORITY_FOOTPRINT = "POLYGON((56.0 57.9, 56.5 57.9, 56.5 58.1, 56.0 58.1, 56.0 57.9))"
HIGH_PRIORITY_FOOTPRINT = "POLYGON((56.23 58.00, 56.26 58.00, 56.26 58.02, 56.23 58.02, 56.23 58.00))"


def _postgres_reachable() -> bool:
    try:
        conn = psycopg.connect(dbname="postgres", connect_timeout=3, **CONN_PARAMS)
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(), reason="требуется доступный Postgres+PostGIS (см. docstring модуля)"
)


@pytest.fixture()
def relief_test_db():
    db_name = f"topology_relief_test_{uuid.uuid4().hex[:8]}"

    admin_conn = psycopg.connect(dbname="postgres", autocommit=True, **CONN_PARAMS)
    try:
        admin_conn.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        admin_conn.close()

    conn = psycopg.connect(dbname=db_name, autocommit=True, **CONN_PARAMS)
    conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    ensure_schema(conn)

    try:
        yield conn
    finally:
        conn.close()
        admin_conn = psycopg.connect(dbname="postgres", autocommit=True, **CONN_PARAMS)
        try:
            admin_conn.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            admin_conn.close()


def _register_sample_coverage(conn) -> None:
    register_coverage(
        conn,
        CoverageEntry(
            source_name="TessaDEM (тест)",
            priority=1,
            storage_key="tessadem.tif",
            resolution_m=30.0,
            data_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            footprint_wkt=LOW_PRIORITY_FOOTPRINT,
        ),
    )
    register_coverage(
        conn,
        CoverageEntry(
            source_name="Топосъёмка (тест)",
            priority=10,
            storage_key="survey.tif",
            resolution_m=1.0,
            data_timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
            footprint_wkt=HIGH_PRIORITY_FOOTPRINT,
        ),
    )


def test_find_coverage_orders_by_priority(relief_test_db):
    _register_sample_coverage(relief_test_db)
    bbox = (56.2, 57.98, 56.29, 58.03)

    entries = find_coverage(relief_test_db, bbox)

    assert [e.source_name for e in entries] == ["Топосъёмка (тест)", "TessaDEM (тест)"]


def test_find_coverage_filters_by_bbox(relief_test_db):
    _register_sample_coverage(relief_test_db)
    far_bbox = (10.0, 10.0, 10.1, 10.1)

    entries = find_coverage(relief_test_db, far_bbox)

    assert entries == []


def _make_tif_bytes(bounds, size, value) -> bytes:
    minx, miny, maxx, maxy = bounds
    w, h = size
    px, py = (maxx - minx) / w, (maxy - miny) / h
    transform = from_origin(minx, maxy, px, py)
    data = np.full((h, w), value, dtype="float32")
    buf = io.BytesIO()
    with rasterio.open(
        buf, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return buf.getvalue()


class _FakeStorage:
    """Хранилище в памяти — реального MinIO в этой среде нет (нет демона Docker)."""

    def __init__(self, data: dict[str, bytes]):
        self._data = data

    def download(self, key: str) -> bytes:
        return self._data[key]


@pytest.fixture()
def sample_storage():
    return _FakeStorage(
        {
            "tessadem.tif": _make_tif_bytes((56.0, 57.9, 56.5, 58.1), (100, 100), 100.0),
            "survey.tif": _make_tif_bytes((56.23, 58.00, 56.26, 58.02), (60, 40), 120.0),
        }
    )


def _msk59_grid(width=500, height=300, pixel=20.0) -> Grid:
    dst_crs = MSK59_ZONES[ZONE].to_proj4()
    x0, y0, _ = wgs84_to_msk59(56.2, 58.03, zone=ZONE)
    transform = Affine(pixel, 0.0, x0, 0.0, -pixel, y0)
    return Grid(transform=transform, width=width, height=height, crs=dst_crs)


def test_get_dem_prefers_high_priority_source_deep_inside(relief_test_db, sample_storage):
    _register_sample_coverage(relief_test_db)
    bbox = (56.2, 57.98, 56.29, 58.03)
    grid = _msk59_grid()

    values, coverage = get_dem(relief_test_db, sample_storage, bbox, grid, transition_width_m=40.0)

    assert coverage.all()
    sx, sy, _ = wgs84_to_msk59(56.245, 58.01, zone=ZONE)  # deep inside survey footprint
    col = int((sx - grid.transform.c) / grid.transform.a)
    row = int((grid.transform.f - sy) / grid.transform.a)
    assert values[row, col] == pytest.approx(120.0, abs=0.5)


def test_get_dem_falls_back_to_low_priority_far_away(relief_test_db, sample_storage):
    _register_sample_coverage(relief_test_db)
    bbox = (56.2, 57.98, 56.29, 58.03)
    grid = _msk59_grid()

    values, coverage = get_dem(relief_test_db, sample_storage, bbox, grid, transition_width_m=40.0)

    assert values[5, 5] == pytest.approx(100.0, abs=0.5)
    assert coverage[5, 5]


def test_get_dem_raises_when_no_coverage(relief_test_db, sample_storage):
    bbox = (10.0, 10.0, 10.1, 10.1)
    grid = _msk59_grid()

    with pytest.raises(LookupError):
        get_dem(relief_test_db, sample_storage, bbox, grid)
