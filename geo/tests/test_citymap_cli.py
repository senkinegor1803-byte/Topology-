"""Тесты Шага 2.12, п. 2: CLI сборки пирамиды тайлов terrain-RGB."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from topology_geo.citymap.cli import build_terrain_tile_pyramid
from topology_geo.citymap.terrain_rgb import decode_terrain_rgb

PERM_LON, PERM_LAT = 56.2431, 58.0105


def _write_dem_geotiff(path, elevation: float, half_extent_deg: float = 0.05) -> None:
    size = 40
    step = 2 * half_extent_deg / size
    transform = from_origin(PERM_LON - half_extent_deg, PERM_LAT + half_extent_deg, step, step)
    data = np.full((size, size), elevation, dtype="float64")
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1, dtype="float64",
        crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


def test_build_terrain_tile_pyramid_writes_expected_layout(tmp_path):
    dem_path = tmp_path / "dem.tif"
    _write_dem_geotiff(dem_path, elevation=180.0)
    output_dir = tmp_path / "tiles"

    count = build_terrain_tile_pyramid(
        dem_path, output_dir,
        min_lon=PERM_LON - 0.02, min_lat=PERM_LAT - 0.02, max_lon=PERM_LON + 0.02, max_lat=PERM_LAT + 0.02,
        min_zoom=11, max_zoom=12,
    )

    assert count > 0
    pngs = list(output_dir.rglob("*.png"))
    assert len(pngs) == count
    for png_path in pngs:
        z = int(png_path.parent.parent.name)
        assert z in (11, 12)


def test_build_terrain_tile_pyramid_tiles_decode_to_real_elevation(tmp_path):
    dem_path = tmp_path / "dem.tif"
    _write_dem_geotiff(dem_path, elevation=180.0)
    output_dir = tmp_path / "tiles"

    build_terrain_tile_pyramid(
        dem_path, output_dir,
        min_lon=PERM_LON - 0.01, min_lat=PERM_LAT - 0.01, max_lon=PERM_LON + 0.01, max_lat=PERM_LAT + 0.01,
        min_zoom=13, max_zoom=13,
    )

    png_path = next(output_dir.rglob("*.png"))
    with rasterio.open(png_path) as src:
        data = src.read()
    rgb = np.moveaxis(data, 0, -1)
    decoded = decode_terrain_rgb(rgb)
    center = decoded[decoded.shape[0] // 2, decoded.shape[1] // 2]
    assert abs(center - 180.0) < 5.0
