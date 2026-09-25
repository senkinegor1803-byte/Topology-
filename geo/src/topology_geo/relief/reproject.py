"""Репроекция растра рельефа в проекцию МСК-59 (Шаг 1.2).

Источники вроде TessaDEM обычно приходят в географических координатах
(WGS-84); дальнейшие шаги конвейера (пересчёт высот `heights.py`, слияние
`merge.py`, TIN на Шаге 1.5) работают в метрах — поэтому репроекция в МСК-59
выполняется один раз здесь, а не откладывается на потом.
"""

from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from topology_geo.coords import MSK59_ZONES


def reproject_to_msk59(
    src_path: str | Path,
    dst_path: str | Path,
    zone: int,
    *,
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """Репроецировать растр рельефа (любая исходная CRS) в МСК-59 зоны `zone`."""
    dst_crs = MSK59_ZONES[zone].to_proj4()

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )
