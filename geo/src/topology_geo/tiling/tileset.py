"""3D Tiles 1.1 `tileset.json` — иерархия детализации по кольцам (Шаг 2.11,
п. 1).

Плоский список тайлов-листьев (без вложенной REPLACE-иерархии): каждый лист
— один тайл 250×250 м (Шаг 2.1) с содержимым (`content.uri` — Storage-ключ
GLB тайла) и `geometricError`, зависящим от кольца LOD (`tiling.grid.
classify_lod_ring`). Концентрические кольца НЕ вложены друг в друга как
parent/child (REPLACE) — дочерний тайл покрывал бы только часть площади
родителя (внутренний круг), а `REPLACE` в 3D Tiles подразумевает, что дети
ПОЛНОСТЬЮ покрывают площадь родителя; вместо этого все тайлы — сиблинги
корня с `refine="ADD"` (родитель без содержимого — просто группирующий
узел), тот же принцип, что и у «плоского» списка тайлов в `tiling.cache`
(Шаг 2.1, п. 2) — самый простой вариант, который остаётся корректным 3D
Tiles JSON.

`geometricError` тайла = шаг фоновой сетки рельефа этого кольца
(`RING_BACKGROUND_STEP_M`) — честная метрика: чем крупнее шаг сетки, тем
грубее приближение, тем больше мировая ошибка. Свой вьюер проекта (не
Cesium) не реализует полный расчёт screen-space error по спецификации —
использует более простые пороги по расстоянию камеры до тайла (см.
`web/viewer/index.html`); сам `tileset.json` при этом остаётся валидным
JSON по спецификации 3D Tiles 1.1 и мог бы открыться в стороннем
3D-Tiles-совместимом клиенте."""

from __future__ import annotations

from dataclasses import dataclass

from topology_geo.tiling.grid import LOD0, LOD1, LOD2, TileIndex

TILESET_VERSION = "1.1"

# Шаг фоновой сетки рельефа тайла по кольцу (Шаг 2.1, п. 4, `tiling.terrain.
# build_tile_terrain`, `background_step_m`) — ближнее кольцо (LOD2) сохраняет
# прежний шаг 1 м (без изменений по сравнению с Шагом 1.5/2.1 для этой зоны);
# дальние кольца сознательно грубее — реальное снижение детализации с
# расстоянием, а не просто другое имя той же геометрии.
RING_BACKGROUND_STEP_M: dict[str, float] = {LOD2: 1.0, LOD1: 5.0, LOD0: 15.0}


@dataclass(frozen=True)
class TileContentEntry:
    tile: TileIndex
    lod: str
    storage_key: str
    # Границы содержимого тайла В ЛОКАЛЬНЫХ координатах участка (центр
    # задачи = (0,0), та же система, что у `site.glb`/IFC, Шаг 1.8) — не
    # мировые МСК-59 (те в GLB как float32 потеряли бы точность на
    # координатах порядка миллионов метров).
    local_minx: float
    local_miny: float
    local_maxx: float
    local_maxy: float
    z_min: float
    z_max: float


def _box_bounding_volume(minx: float, miny: float, maxx: float, maxy: float, minz: float, maxz: float) -> list[float]:
    """Ось-выровненный `box` bounding volume 3D Tiles: центр + 3 полу-оси
    (12 чисел, спецификация 3D Tiles `boundingVolume.box`)."""
    cx, cy, cz = (minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2
    hx, hy, hz = (maxx - minx) / 2, (maxy - miny) / 2, max((maxz - minz) / 2, 0.5)
    return [cx, cy, cz, hx, 0.0, 0.0, 0.0, hy, 0.0, 0.0, 0.0, hz]


def build_tileset_json(entries: list[TileContentEntry]) -> dict:
    """Собрать `tileset.json` (Шаг 2.11, п. 1) — плоский список тайлов-листьев
    под общим корнем-группой."""
    children = []
    for entry in entries:
        children.append(
            {
                "boundingVolume": {
                    "box": _box_bounding_volume(
                        entry.local_minx, entry.local_miny, entry.local_maxx, entry.local_maxy,
                        entry.z_min, entry.z_max,
                    )
                },
                "geometricError": RING_BACKGROUND_STEP_M[entry.lod],
                "content": {"uri": entry.storage_key},
                "extras": {"lod": entry.lod, "tx": entry.tile.tx, "ty": entry.tile.ty, "zone": entry.tile.zone},
            }
        )

    if entries:
        root_minx = min(e.local_minx for e in entries)
        root_miny = min(e.local_miny for e in entries)
        root_maxx = max(e.local_maxx for e in entries)
        root_maxy = max(e.local_maxy for e in entries)
        root_minz = min(e.z_min for e in entries)
        root_maxz = max(e.z_max for e in entries)
        root_error = max(RING_BACKGROUND_STEP_M.values())
    else:
        root_minx = root_miny = root_maxx = root_maxy = 0.0
        root_minz = root_maxz = 0.0
        root_error = 0.0

    return {
        "asset": {"version": TILESET_VERSION, "generator": "topology-geo (Шаг 2.11)"},
        "geometricError": root_error,
        "root": {
            "boundingVolume": {
                "box": _box_bounding_volume(root_minx, root_miny, root_maxx, root_maxy, root_minz, root_maxz)
            },
            "geometricError": root_error,
            "refine": "ADD",
            "children": children,
        },
    }
