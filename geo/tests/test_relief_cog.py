"""Тесты конвертации в Cloud Optimized GeoTIFF (Шаг 1.2, п. 1)."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rio_cogeo.cogeo import cog_validate

from topology_geo.relief.cog import to_cog


@pytest.fixture()
def sample_geotiff(tmp_path):
    path = tmp_path / "sample.tif"
    width, height = 64, 64
    transform = from_origin(56.0, 58.1, 0.001, 0.001)
    data = np.linspace(100, 200, width * height, dtype="float32").reshape(height, width)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path, data


def test_to_cog_produces_valid_cog(sample_geotiff, tmp_path):
    src_path, _ = sample_geotiff
    dst_path = tmp_path / "out_cog.tif"

    to_cog(src_path, dst_path)

    is_valid, errors, warnings = cog_validate(str(dst_path))
    assert is_valid, f"errors={errors} warnings={warnings}"


def test_to_cog_preserves_pixel_values(sample_geotiff, tmp_path):
    src_path, src_data = sample_geotiff
    dst_path = tmp_path / "out_cog.tif"

    to_cog(src_path, dst_path)

    with rasterio.open(dst_path) as f:
        out_data = f.read(1)
        assert f.crs == rasterio.crs.CRS.from_epsg(4326)

    np.testing.assert_allclose(out_data, src_data, rtol=1e-5)


def test_to_cog_uses_requested_blocksize(sample_geotiff, tmp_path):
    src_path, _ = sample_geotiff
    dst_path = tmp_path / "out_cog.tif"

    to_cog(src_path, dst_path, blocksize=32)

    with rasterio.open(dst_path) as f:
        assert f.block_shapes[0] == (32, 32)
