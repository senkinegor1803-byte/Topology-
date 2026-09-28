"""Содержимое одного тайла 3D Tiles: рельеф (Шаг 2.1, п. 4) + здания в
границах тайла, на уровне детализации своего кольца (Шаг 2.11, п. 1).

Рельеф — существующий `tiling.terrain.build_tile_terrain` (Шаг 2.1) с шагом
фоновой сетки по кольцу (`tileset.RING_BACKGROUND_STEP_M`) — переиспользуется
как есть, без изменений. Здания — та же меш-логика по LOD, что и CityJSON
(Шаг 2.10, `export.cityjson`): LOD2 (ближнее кольцо) — с крышей
(`_building_lod2_mesh`); LOD1 (среднее) — блок на всю высоту
(`extrude_polygon_mesh`); LOD0 (дальнее) — плоский контур пятна без высоты
(`triangulate_polygon`, тот же приём, что `export.cityjson._lod0_footprint_
boundaries`, но триангулированный, а не как границы CityJSON-поверхности —
тайлу нужны настоящие треугольники для рендера, не индексированные кольца).

Координаты содержимого — ЛОКАЛЬНЫЕ (центр задачи = (0,0), та же система,
что у `site.glb`): рельеф хранится в мировых МСК-59 (`prepare_relief`),
поэтому его вершины смещаются на `-(center_x, center_y)`; контуры зданий уже
в локальных координатах (Шаг 1.4) и не трогаются."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topology_geo.geometry.buildings import BuildingSolid
from topology_geo.geometry.roofs import ROOF_FLAT
from topology_geo.ifc.assemble import extrude_polygon_mesh, triangulate_polygon
from topology_geo.relief.service import Grid
from topology_geo.tiling.gltf_mesh import MeshPart, build_glb
from topology_geo.tiling.grid import LOD0, LOD1, TileIndex
from topology_geo.tiling.terrain import build_tile_terrain
from topology_geo.tiling.tileset import RING_BACKGROUND_STEP_M


def _building_mesh_for_lod(building: BuildingSolid, lod: str) -> tuple[list, list]:
    if lod == LOD0:
        flat_vertices, triangles = triangulate_polygon(building.footprint)
        verts = [(x, y, building.base_z) for x, y in flat_vertices]
        return verts, triangles
    if lod == LOD1 or building.roof_shape == ROOF_FLAT:
        return extrude_polygon_mesh(building.footprint, building.base_z, building.base_z + building.height_m)
    from topology_geo.export.cityjson import _building_lod2_mesh  # см. докстринг модуля

    return _building_lod2_mesh(building)


@dataclass
class TileContentResult:
    glb_bytes: bytes
    z_min: float
    z_max: float


def build_tile_content(
    relief_values: np.ndarray,
    relief_grid: Grid,
    tile: TileIndex,
    lod: str,
    buildings_in_tile: list[BuildingSolid],
    *,
    center_x: float,
    center_y: float,
) -> TileContentResult | None:
    """Собрать GLB одного тайла. `None`, если у тайла нет ни покрытия
    рельефа, ни зданий (пустой тайл — в `tileset.json` не попадает)."""
    parts: list[MeshPart] = []
    z_values: list[float] = []

    try:
        tin = build_tile_terrain(relief_values, relief_grid, tile, background_step_m=RING_BACKGROUND_STEP_M[lod])
    except ValueError:
        tin = None  # тайл вне покрытия DEM (запас RELIEF_MARGIN_M мог не дотянуться до края радиуса)

    if tin is not None:
        local_verts = tin.vertices.copy()
        local_verts[:, 0] -= center_x
        local_verts[:, 1] -= center_y
        parts.append(MeshPart(name="Рельеф", category="Рельеф", vertices=local_verts, faces=tin.triangles))
        z_values.extend([float(tin.vertices[:, 2].min()), float(tin.vertices[:, 2].max())])

    for building in buildings_in_tile:
        verts, faces = _building_mesh_for_lod(building, lod)
        if not verts or not faces:
            continue
        parts.append(
            MeshPart(
                name=f"Здание {building.osm_id}", category="Здания", vertices=verts, faces=faces,
                extras={"osmId": building.osm_id, "тип": building.building_type},
            )
        )
        zs = [v[2] for v in verts]
        z_values.extend([min(zs), max(zs)])

    if not parts:
        return None

    return TileContentResult(glb_bytes=build_glb(parts), z_min=min(z_values), z_max=max(z_values))
