"""Реализация шагов пайплайна задачи (Шаг 1.3).

`select_osm`, `prepare_relief` и `select_and_normalize` — реальные шаги,
использующие уже реализованные Шаги 1.1, 1.2 и 1.4. Более поздние шаги
конвейера (TIN участка, здания, дороги, сборка IFC, веб-конвертация —
Шаги 1.5-1.9) сюда пока не входят: `DEFAULT_PIPELINE` расширится вместе с их
реализацией.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine

from topology_geo.coords import MSK59_ZONES, pick_msk59_zone, wgs84_to_msk59
from topology_geo.jobs import store
from topology_geo.osm.queries import count_within_radius
from topology_geo.relief.cog import to_cog
from topology_geo.relief.service import Grid, get_dem
from topology_geo.selection.geopackage import dataset_to_geopackage_bytes
from topology_geo.selection.service import select_site_data
from topology_geo.storage import ObjectStorage

RELIEF_PIXEL_SIZE_M = 10.0
RELIEF_MARGIN_M = 200.0


def select_osm(conn: Any, storage: ObjectStorage, job: store.Job) -> dict:
    """Шаг 1: посчитать объекты OSM в радиусе задачи (Шаг 1.1) и сохранить сводку."""
    try:
        counts = count_within_radius(conn, lon=job.center_lon, lat=job.center_lat, radius_m=job.radius_m)
    except Exception as exc:  # noqa: BLE001 - таблиц может не быть, если импорт (Шаг 1.1) не запускался
        raise RuntimeError(
            "нет данных OSM для этой области (проверьте, что выполнен импорт по Шагу 1.1)"
        ) from exc

    key = f"jobs/{job.id}/osm_selection.json"
    storage.upload(key, json.dumps(counts, ensure_ascii=False).encode("utf-8"), content_type="application/json")
    return {"storage_key": key, "counts": counts}


def _wgs84_bbox_for_radius(lon: float, lat: float, radius_m: float, zone: int) -> tuple[float, float, float, float]:
    """Bbox в WGS-84, покрывающий круг `radius_m` вокруг (lon, lat), считая
    через МСК-59 (метры) — точнее, чем наивный градусный отступ, который
    искажается с широтой (переиспользует Шаг 0.4)."""
    x, y, _ = wgs84_to_msk59(lon, lat, zone=zone)
    corners = [(x - radius_m, y - radius_m), (x + radius_m, y - radius_m), (x - radius_m, y + radius_m), (x + radius_m, y + radius_m)]
    lons, lats = [], []
    for cx, cy in corners:
        clon, clat = _msk59_to_wgs84_local(cx, cy, zone)
        lons.append(clon)
        lats.append(clat)
    return (min(lons), min(lats), max(lons), max(lats))


def _msk59_to_wgs84_local(x: float, y: float, zone: int):
    from topology_geo.coords import msk59_to_wgs84

    return msk59_to_wgs84(x, y, zone=zone)


def _array_to_geotiff_bytes(values: np.ndarray, grid: Grid, nodata: float = -9999.0) -> bytes:
    from rasterio.io import MemoryFile

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=grid.height, width=grid.width, count=1, dtype="float64",
            crs=grid.crs, transform=grid.transform, nodata=nodata,
        ) as dst:
            dst.write(values, 1)
        return memfile.read()


def prepare_relief(conn: Any, storage: ObjectStorage, job: store.Job) -> dict:
    """Шаг 2: собрать лучший доступный рельеф по радиусу задачи (Шаг 1.2) и
    сохранить как COG."""
    zone = pick_msk59_zone(job.center_lon)
    bbox = _wgs84_bbox_for_radius(job.center_lon, job.center_lat, job.radius_m, zone)

    half_extent = job.radius_m + RELIEF_MARGIN_M
    cx, cy, _ = wgs84_to_msk59(job.center_lon, job.center_lat, zone=zone)
    width = height = max(int(2 * half_extent / RELIEF_PIXEL_SIZE_M), 2)
    transform = Affine(RELIEF_PIXEL_SIZE_M, 0.0, cx - half_extent, 0.0, -RELIEF_PIXEL_SIZE_M, cy + half_extent)
    grid = Grid(transform=transform, width=width, height=height, crs=MSK59_ZONES[zone].to_proj4())

    try:
        values, coverage = get_dem(conn, storage, bbox, grid)
    except LookupError as exc:
        raise RuntimeError("нет данных рельефа в этой области") from exc

    raw_bytes = _array_to_geotiff_bytes(values, grid)
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "relief_raw.tif"
        dst_path = Path(tmp) / "relief.tif"
        src_path.write_bytes(raw_bytes)
        to_cog(src_path, dst_path)
        cog_bytes = dst_path.read_bytes()

    key = f"jobs/{job.id}/relief.tif"
    storage.upload(key, cog_bytes, content_type="image/tiff")
    return {
        "storage_key": key,
        "coverage_fraction": float(coverage.mean()),
        "width": width,
        "height": height,
        "pixel_size_m": RELIEF_PIXEL_SIZE_M,
    }


def select_and_normalize(conn: Any, storage: ObjectStorage, job: store.Job) -> dict:
    """Шаг 3: выборка, обрезка кругом и нормализация атрибутов данных участка
    (Шаг 1.4) — «чистый набор данных участка, одинаковый для всех дальнейших
    генераторов», сохранён как GeoPackage."""
    try:
        dataset = select_site_data(conn, job.center_lon, job.center_lat, job.radius_m)
    except Exception as exc:  # noqa: BLE001 - таблиц может не быть, если импорт (Шаг 1.1) не запускался
        raise RuntimeError(
            "нет данных OSM для этой области (проверьте, что выполнен импорт по Шагу 1.1)"
        ) from exc

    gpkg_bytes = dataset_to_geopackage_bytes(dataset)

    key = f"jobs/{job.id}/site.gpkg"
    storage.upload(key, gpkg_bytes, content_type="application/geopackage+sqlite3")
    return {
        "storage_key": key,
        "zone": dataset.zone,
        "feature_count": len(dataset.features),
        "layers": {layer: len(features) for layer, features in dataset.by_layer().items()},
    }


DEFAULT_PIPELINE: dict[str, Any] = {
    "select_osm": select_osm,
    "prepare_relief": prepare_relief,
    "select_and_normalize": select_and_normalize,
}

DEFAULT_STEP_NAMES: list[str] = list(DEFAULT_PIPELINE)
