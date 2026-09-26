"""Сервис `get_dem(bbox)` — лучший доступный рельеф по области (Шаг 1.2, итог шага).

Оркестрирует: найти покрытия (`coverage.find_coverage`) -> скачать нужные
растры из объектного хранилища -> выровнять на общую сетку -> слить по
приоритету (`merge.merge_with_transition`, от низкого приоритета к высокому).

Хранилище абстрагировано протоколом `Storage` (тот же приём, что и
`devcheck.check_minio`): в этой среде разработки нет демона Docker, поэтому
настоящего MinIO для интеграционных тестов оркестрации нет — здесь
тестируется сама логика с фейковым хранилищем в памяти, а не сетевой слой.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import rasterio
from affine import Affine
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject

from topology_geo.relief.coverage import find_coverage
from topology_geo.relief.merge import merge_with_transition


class Storage(Protocol):
    def download(self, key: str) -> bytes: ...


class _Connection(Protocol):
    def cursor(self) -> Any: ...


@dataclass(frozen=True)
class Grid:
    """Целевая сетка, на которую выравниваются все источники перед слиянием.

    `crs` должна быть проекционной (метры), например МСК-59
    (`topology_geo.coords.MSK59_ZONES[zone].to_proj4()`) — `pixel_size_m` и
    ширина перехода в `merge.merge_with_transition` считаются в метрах по
    `transform`; географическая CRS (градусы) даст неверный масштаб перехода.
    """

    transform: Affine
    width: int
    height: int
    crs: str

    @property
    def pixel_size_m(self) -> float:
        return abs(self.transform.a)


def _read_aligned(data: bytes, grid: Grid, resampling: Resampling = Resampling.bilinear) -> tuple[np.ndarray, np.ndarray]:
    """Прочитать растр из байтов и выровнять на `grid`. Возвращает (values, valid_mask)."""
    with MemoryFile(data) as memfile, memfile.open() as src:
        dst_array = np.full((grid.height, grid.width), np.nan, dtype="float64")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            src_nodata=src.nodata,
            resampling=resampling,
        )
    valid = ~np.isnan(dst_array)
    return dst_array, valid


def get_dem(
    conn: _Connection,
    storage: Storage,
    bbox: tuple[float, float, float, float],
    grid: Grid,
    *,
    transition_width_m: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Собрать лучший доступный рельеф по `bbox` на сетке `grid`.

    Возвращает `(values, coverage_mask)` — `coverage_mask=False` там, где
    вообще не нашлось источника.
    """
    entries = find_coverage(conn, bbox)
    if not entries:
        raise LookupError(f"нет покрытия рельефа для {bbox}")

    entries_ascending = sorted(entries, key=lambda e: e.priority)  # низкий приоритет -> overlay сверху

    result = np.zeros((grid.height, grid.width), dtype="float64")
    coverage_mask = np.zeros((grid.height, grid.width), dtype=bool)

    for entry in entries_ascending:
        values, valid = _read_aligned(storage.download(entry.storage_key), grid)

        if not coverage_mask.any():
            result = np.where(valid, values, result)
            coverage_mask = valid
            continue

        blended = merge_with_transition(
            result, values, valid, pixel_size_m=grid.pixel_size_m, transition_width_m=transition_width_m
        )
        both = coverage_mask & valid
        only_new = valid & ~coverage_mask
        result = np.where(both, blended, result)
        result = np.where(only_new, values, result)
        coverage_mask = coverage_mask | valid

    return result, coverage_mask


def read_relief_from_storage(storage: Storage, storage_key: str) -> tuple[np.ndarray, Grid]:
    """Прочитать COG рельефа (например, сохранённый `jobs.steps.prepare_relief`
    или тайловым генератором Этапа 2) обратно в массив высот + `Grid` — не
    пересчитывать слияние источников заново, единственный источник истины уже
    в хранилище."""
    with MemoryFile(storage.download(storage_key)) as memfile, memfile.open() as src:
        values = src.read(1)
        crs = src.crs.to_proj4() if src.crs else ""
        return values, Grid(transform=src.transform, width=src.width, height=src.height, crs=crs)
