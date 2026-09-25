"""Тесты пересчёта высот растра рельефа (Шаг 1.2, п. 2; модель — Шаг 0.4)."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine

from topology_geo.coords import MSK59_ZONES, HeightOffsetModel, wgs84_to_msk59
from topology_geo.relief.heights import correct_heights_plane

PERM_LON, PERM_LAT = 56.2431, 58.0105
ZONE = 2


def _make_msk59_raster(path, value_fn, width=40, height=40, pixel=10.0):
    x0, y0, _ = wgs84_to_msk59(PERM_LON, PERM_LAT, zone=ZONE)
    transform = Affine(pixel, 0.0, x0, 0.0, -pixel, y0)
    rows, cols = np.indices((height, width))
    xs = transform.a * (cols + 0.5) + transform.c
    ys = transform.e * (rows + 0.5) + transform.f
    data = value_fn(xs, ys).astype("float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs=MSK59_ZONES[ZONE].to_proj4(), transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return transform


def test_constant_offset_shifts_every_pixel(tmp_path):
    src_path = tmp_path / "src.tif"
    _make_msk59_raster(src_path, lambda xs, ys: np.full_like(xs, 100.0))

    model = HeightOffsetModel(lon0=PERM_LON, lat0=PERM_LAT, offset=-14.7, grad_x=0.0, grad_y=0.0)
    dst_path = tmp_path / "out.tif"
    correct_heights_plane(src_path, dst_path, model, zone=ZONE)

    with rasterio.open(dst_path) as f:
        out = f.read(1)
    np.testing.assert_allclose(out, 100.0 - 14.7, rtol=1e-4)


def test_gradient_varies_spatially_as_expected(tmp_path):
    src_path = tmp_path / "src.tif"
    _make_msk59_raster(src_path, lambda xs, ys: np.full_like(xs, 100.0))

    grad_x, grad_y = 0.001, -0.0005
    model = HeightOffsetModel(lon0=PERM_LON, lat0=PERM_LAT, offset=0.0, grad_x=grad_x, grad_y=grad_y)
    dst_path = tmp_path / "out.tif"
    correct_heights_plane(src_path, dst_path, model, zone=ZONE)

    x0, y0, _ = wgs84_to_msk59(PERM_LON, PERM_LAT, zone=ZONE)
    with rasterio.open(dst_path) as f:
        out = f.read(1)
        transform = f.transform

    # проверить формулу в двух произвольных пикселях напрямую
    for row, col in [(5, 5), (30, 35)]:
        x = transform.a * (col + 0.5) + transform.c
        y = transform.e * (row + 0.5) + transform.f
        expected = 100.0 + grad_x * (x - x0) + grad_y * (y - y0)
        assert out[row, col] == pytest.approx(expected, abs=1e-2)


def test_nodata_pixels_are_not_corrected(tmp_path):
    src_path = tmp_path / "src.tif"

    def value_fn(xs, ys):
        data = np.full_like(xs, 100.0)
        data[0, 0] = -9999.0
        return data

    _make_msk59_raster(src_path, value_fn)

    model = HeightOffsetModel(lon0=PERM_LON, lat0=PERM_LAT, offset=-14.7)
    dst_path = tmp_path / "out.tif"
    correct_heights_plane(src_path, dst_path, model, zone=ZONE)

    with rasterio.open(dst_path) as f:
        out = f.read(1, masked=True)
    assert out[0, 0] is np.ma.masked or out.data[0, 0] == pytest.approx(-9999.0)
    assert out.data[1, 1] == pytest.approx(100.0 - 14.7, abs=1e-2)
