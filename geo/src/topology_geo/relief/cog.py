"""Конвертация растра рельефа в Cloud Optimized GeoTIFF (Шаг 1.2, п. 1).

Обёртка над `rio-cogeo` — конвертация в COG не переизобретается поверх GDAL
вручную, используется реальная проверенная библиотека (её же `cog_validate`
используется в тестах для проверки результата).
"""

from __future__ import annotations

from pathlib import Path

from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles


def to_cog(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    blocksize: int = 512,
    resampling: str = "bilinear",
    overview_resampling: str = "bilinear",
) -> None:
    """Сконвертировать растр рельефа (GeoTIFF) в Cloud Optimized GeoTIFF.

    Непрерывные значения высот -> билинейная передискретизация и для данных,
    и для строящихся оверов (в отличие от категориальных слоёв, где уместнее
    `nearest`).
    """
    dst_profile = cog_profiles.get("deflate")
    dst_profile.update({"blockxsize": blocksize, "blockysize": blocksize})
    cog_translate(
        str(src_path),
        str(dst_path),
        dst_profile,
        resampling=resampling,
        overview_resampling=overview_resampling,
        in_memory=False,
        quiet=True,
    )
