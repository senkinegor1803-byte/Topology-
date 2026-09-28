"""Тайлы рельефа terrain-RGB (Шаг 2.12, п. 2).

Формат Mapbox Terrain-RGB — общепринятая кодировка, НЕ придуманная для этого
проекта: высота в каждом пикселе восстанавливается по формуле
`height = -10000 + (R*256*256 + G*256 + B) * 0.1` (метры, шаг 0.1 м,
диапазон примерно от -10000 до +6553.5 м) — тот же формат, что понимают
готовые терраин-рендереры MapLibre GL (`source.encoding = "mapbox"`).

Схема тайлов — стандартная XYZ/slippy-map в Web Mercator (EPSG:3857, y=0
сверху) — расчёт границ тайла по стандартной формуле, БЕЗ зависимости
`mercantile` (в проекте уже есть `rasterio`/GDAL, которых достаточно для
репроекции и записи PNG — добавлять вторую библиотеку с тем же результатом
не нужно).

Источник высот — любой уже загруженный в проекте растр (тот же `Grid` +
values, что использует `relief.service`/`relief.tin`), не отдельный формат:
`render_terrain_rgb_tile` сам репроецирует его на сетку конкретного тайла
через `rasterio.warp.reproject` (тот же приём, что `relief.service.
_read_aligned`)."""

from __future__ import annotations

import math

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject

from topology_geo.relief.service import Grid

WEB_MERCATOR_CRS = "EPSG:3857"
EARTH_CIRCUMFERENCE_M = 40_075_016.685_578_5
ORIGIN_SHIFT_M = EARTH_CIRCUMFERENCE_M / 2.0

MIN_ENCODABLE_M = -10_000.0
MAX_ENCODABLE_M = MIN_ENCODABLE_M + (256**3 - 1) * 0.1  # R=G=B=255, ~1 667 721.5 м - заведомо не достигается


def tile_bounds_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Границы XYZ-тайла (Google/OSM-схема, y=0 сверху) в метрах Web Mercator
    (minx, miny, maxx, maxy) — стандартная формула слайпи-карт."""
    tile_size = EARTH_CIRCUMFERENCE_M / (2**z)
    minx = x * tile_size - ORIGIN_SHIFT_M
    maxx = (x + 1) * tile_size - ORIGIN_SHIFT_M
    maxy = ORIGIN_SHIFT_M - y * tile_size
    miny = ORIGIN_SHIFT_M - (y + 1) * tile_size
    return minx, miny, maxx, maxy


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    """WGS-84 (lon, lat) -> индекс XYZ-тайла на уровне `z` (стандартная
    формула слайпи-карт, обратная к `tile_bounds_3857`)."""
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tiles_covering_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float, z: int) -> list[tuple[int, int]]:
    """Все тайлы уровня `z`, чей квадрат пересекает bbox WGS-84 — для сборки
    пирамиды terrain-RGB по территории (Шаг 2.12, п. 2)."""
    x_min, y_min = lonlat_to_tile(min_lon, max_lat, z)  # север-запад -> меньшие x/y
    x_max, y_max = lonlat_to_tile(max_lon, min_lat, z)  # юг-восток -> большие x/y
    n = 2**z
    x_min, x_max = max(x_min, 0), min(x_max, n - 1)
    y_min, y_max = max(y_min, 0), min(y_max, n - 1)
    return [(x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]


def encode_terrain_rgb(elevations_m: np.ndarray) -> np.ndarray:
    """Высоты (метры) -> RGB-массив (H, W, 3) uint8 по формуле Mapbox
    Terrain-RGB. Значения вне кодируемого диапазона обрезаются (clip) —
    честно, не переполняются молча."""
    clipped = np.clip(elevations_m, MIN_ENCODABLE_M, MAX_ENCODABLE_M)
    value = np.round((clipped - MIN_ENCODABLE_M) / 0.1).astype(np.uint32)
    r = (value // (256 * 256)) % 256
    g = (value // 256) % 256
    b = value % 256
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def decode_terrain_rgb(rgb: np.ndarray) -> np.ndarray:
    """Обратное преобразование (для тестов - проверить, что кодирование
    обратимо с точностью до шага 0.1 м)."""
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return MIN_ENCODABLE_M + (r * 256 * 256 + g * 256 + b) * 0.1


def render_terrain_rgb_tile(
    values: np.ndarray, grid: Grid, z: int, x: int, y: int, *, tile_size: int = 256, nodata_elevation: float = 0.0
) -> bytes:
    """Собрать PNG-тайл terrain-RGB `(z, x, y)`. `values`/`grid` — любой уже
    загруженный в проекте растр рельефа (метры, произвольная исходная CRS) —
    репроецируется на сетку тайла в Web Mercator. Точки вне покрытия
    источника получают `nodata_elevation` (честная типовая плоскость, не
    NaN — PNG-тайл должен быть полностью заполнен для клиента-рендерера)."""
    minx, miny, maxx, maxy = tile_bounds_3857(z, x, y)
    pixel_size = (maxx - minx) / tile_size
    dst_transform = rasterio.transform.from_origin(minx, maxy, pixel_size, pixel_size)

    dst = np.full((tile_size, tile_size), np.nan, dtype="float64")
    reproject(
        source=values,
        destination=dst,
        src_transform=grid.transform,
        src_crs=grid.crs,
        dst_transform=dst_transform,
        dst_crs=WEB_MERCATOR_CRS,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    dst = np.where(np.isnan(dst), nodata_elevation, dst)

    rgb = encode_terrain_rgb(dst)
    with MemoryFile() as memfile:
        with memfile.open(
            driver="PNG", height=tile_size, width=tile_size, count=3, dtype="uint8",
        ) as dataset:
            dataset.write(rgb[:, :, 0], 1)
            dataset.write(rgb[:, :, 1], 2)
            dataset.write(rgb[:, :, 2], 3)
        return memfile.read()
