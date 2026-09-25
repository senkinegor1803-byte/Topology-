"""Тесты репроекции растра рельефа в МСК-59 (Шаг 1.2, п. 1-2 подготовки)."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from topology_geo.coords import MSK59_ZONES
from topology_geo.relief.reproject import reproject_to_msk59

PERM_LON, PERM_LAT = 56.2431, 58.0105


@pytest.fixture()
def wgs84_geotiff(tmp_path):
    path = tmp_path / "src_wgs84.tif"
    width, height = 50, 50
    pixel_deg = 0.0005
    transform = from_origin(
        PERM_LON - width / 2 * pixel_deg, PERM_LAT + height / 2 * pixel_deg, pixel_deg, pixel_deg
    )
    data = np.full((height, width), 123.0, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.mark.parametrize("zone", sorted(MSK59_ZONES))
def test_reproject_sets_target_crs(wgs84_geotiff, tmp_path, zone):
    dst_path = tmp_path / f"reproj_{zone}.tif"
    reproject_to_msk59(wgs84_geotiff, dst_path, zone=zone)

    with rasterio.open(dst_path) as f:
        # сравниваем CRS семантически (через rasterio.crs.CRS), а не строкой
        # proj4 - GDAL нормализует форматирование (0 -> 0.0, +no_defs=True) при
        # обратном round-trip, что не значит несовпадение системы координат.
        assert f.crs == rasterio.crs.CRS.from_proj4(MSK59_ZONES[zone].to_proj4())
        # окрестности осевого меридиана зоны -> координаты порядка false_easting/northing
        assert abs(f.bounds.left - MSK59_ZONES[zone].false_easting) < 200_000


def test_reproject_preserves_constant_values(wgs84_geotiff, tmp_path):
    dst_path = tmp_path / "reproj.tif"
    reproject_to_msk59(wgs84_geotiff, dst_path, zone=2)

    with rasterio.open(dst_path) as f:
        data = f.read(1, masked=True)
        # значение константное в источнике -> после билинейной репроекции
        # (без изменения самого значения, только геометрии) должно остаться ~123
        assert data.compressed().mean() == pytest.approx(123.0, abs=0.5)


def test_reproject_pixel_size_is_metric(wgs84_geotiff, tmp_path):
    dst_path = tmp_path / "reproj.tif"
    reproject_to_msk59(wgs84_geotiff, dst_path, zone=2)

    with rasterio.open(dst_path) as f:
        # исходный пиксель 0.0005 град ~ 30-55 м в зависимости от широты/долготы;
        # после репроекции пиксель должен быть в разумных метрах, не в градусах.
        assert 10.0 < abs(f.transform.a) < 100.0
