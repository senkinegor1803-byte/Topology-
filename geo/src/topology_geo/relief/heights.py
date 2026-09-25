"""Пересчёт высот растра рельефа EGM96 -> Балтийская система (Шаг 1.2, п. 2).

Модель поправки калибруется по контрольным точкам на Шаге 0.4
(`topology_geo.coords.HeightOffsetModel`) — эта плоскость линейна в
проекционных координатах (МСК-59, метры), поэтому применение к растру
полностью векторизуется через numpy, если растр уже репроецирован в МСК-59
(`reproject.reproject_to_msk59`) — без per-pixel обратной проекции в WGS-84.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from topology_geo.coords import HeightOffsetModel, wgs84_to_msk59


def _pixel_center_coords(transform: Affine, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices((height, width))
    xs = transform.a * (cols + 0.5) + transform.b * (rows + 0.5) + transform.c
    ys = transform.d * (cols + 0.5) + transform.e * (rows + 0.5) + transform.f
    return xs, ys


def correct_heights_plane(
    src_path: str | Path,
    dst_path: str | Path,
    model: HeightOffsetModel,
    zone: int,
) -> None:
    """Применить плоскость-поправку `model` к каждому пикселю растра.

    Растр должен быть в проекции МСК-59 зоны `zone` (в тех же единицах, в
    которых откалиброван `model`).
    """
    with rasterio.open(src_path) as src:
        data = src.read(1, masked=True)
        xs, ys = _pixel_center_coords(src.transform, src.width, src.height)

        x0, y0, _ = wgs84_to_msk59(model.lon0, model.lat0, zone=zone)
        corrected = data.astype("float64") + model.offset + model.grad_x * (xs - x0) + model.grad_y * (ys - y0)

        fill_value = src.nodata if src.nodata is not None else 0.0
        out = np.ma.filled(corrected, fill_value)

        profile = src.profile.copy()
        profile.update({"dtype": "float64"})
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(out, 1)
