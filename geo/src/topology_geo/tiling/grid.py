"""Кольца детализации и тайловая сетка (Шаг 2.1, п. 1-2).

Тайлы адресуются в МИРОВЫХ координатах МСК-59 зоны (не в локальных
координатах задачи, которые у каждой задачи свои условные (0,0) в центре
участка) — иначе один и тот же физический тайл, запрошенный из двух разных
задач с разными центрами, не совпадал бы по ключу и кеш (`tiling/cache.py`)
никогда бы не срабатывал. Смысл тайлов по плану («повторный запрос по той
же зоне ≤ 30 с») именно в переиспользовании между задачами.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TILE_SIZE_M = 250.0

# Границы колец LOD (Шаг 2.1, п. 1; таблица п. 4.3 ТЗ недоступна AI-сессии
# отдельно от текста самого шага плана — используются ровно те границы,
# что явно перечислены в действии шага).
LOD2_MAX_M = 500.0
LOD1_MAX_M = 1500.0
LOD0_MAX_M = 3000.0

LOD2 = "LOD2"
LOD1 = "LOD1"
LOD0 = "LOD0"


def classify_lod_ring(distance_m: float) -> str:
    """Кольцо детализации по расстоянию от центра участка (Шаг 2.1, п. 1)."""
    if distance_m < 0:
        raise ValueError(f"расстояние не может быть отрицательным: {distance_m}")
    if distance_m <= LOD2_MAX_M:
        return LOD2
    if distance_m <= LOD1_MAX_M:
        return LOD1
    if distance_m <= LOD0_MAX_M:
        return LOD0
    raise ValueError(f"расстояние {distance_m} м вне зоны Этапа 2 (радиус до {LOD0_MAX_M} м)")


@dataclass(frozen=True)
class TileIndex:
    """Индекс тайла 250×250 м в мировых координатах МСК-59 зоны `zone`."""

    zone: int
    tx: int
    ty: int

    def bounds(self, tile_size_m: float = TILE_SIZE_M) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) тайла в мировых координатах МСК-59."""
        minx = self.tx * tile_size_m
        miny = self.ty * tile_size_m
        return minx, miny, minx + tile_size_m, miny + tile_size_m

    def key(self, layer: str, data_version: str, generator_version: str, tile_size_m: float = TILE_SIZE_M) -> str:
        """Ключ кеша — «координаты + версия данных + версия генератора» (Шаг 2.1, п. 2),
        плюс слой: рельеф/дороги/здания одного и того же квадрата — отдельные
        артефакты со своим циклом обновления (здания Шага 2.2 не должны
        протухать в кеше, когда меняется только генератор дорог Шага 2.3), а
        не один ключ на всех. Ключ меняется, если сдвинулась граница тайла
        (другой `tile_size_m`), обновились исходные данные (`data_version`,
        например дата OSM/DEM) или сам код генерации тайла
        (`generator_version`, например хеш/версия пакета)."""
        return f"msk59-{self.zone}/{tile_size_m:g}/{self.tx}_{self.ty}/{layer}/{data_version}/{generator_version}"


def world_to_tile_index(zone: int, world_x: float, world_y: float, tile_size_m: float = TILE_SIZE_M) -> TileIndex:
    return TileIndex(zone=zone, tx=math.floor(world_x / tile_size_m), ty=math.floor(world_y / tile_size_m))


def tiles_covering_circle(
    zone: int, center_x: float, center_y: float, radius_m: float, tile_size_m: float = TILE_SIZE_M
) -> list[TileIndex]:
    """Все тайлы, чей квадрат пересекает круг `radius_m` вокруг (center_x, center_y)
    (мировые координаты МСК-59 зоны `zone`) — Шаг 2.1, п. 2."""
    if radius_m <= 0:
        raise ValueError(f"радиус должен быть положительным: {radius_m}")

    min_tx = math.floor((center_x - radius_m) / tile_size_m)
    max_tx = math.floor((center_x + radius_m) / tile_size_m)
    min_ty = math.floor((center_y - radius_m) / tile_size_m)
    max_ty = math.floor((center_y + radius_m) / tile_size_m)

    tiles = []
    for tx in range(min_tx, max_tx + 1):
        for ty in range(min_ty, max_ty + 1):
            tile = TileIndex(zone=zone, tx=tx, ty=ty)
            minx, miny, maxx, maxy = tile.bounds(tile_size_m)
            nearest_x = min(max(center_x, minx), maxx)
            nearest_y = min(max(center_y, miny), maxy)
            if math.hypot(center_x - nearest_x, center_y - nearest_y) <= radius_m:
                tiles.append(tile)
    return tiles
