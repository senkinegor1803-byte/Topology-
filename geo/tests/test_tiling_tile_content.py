"""Тесты Шага 2.11, п. 1: содержимое тайла 3D Tiles (рельеф + здания по LOD)."""

from __future__ import annotations

import numpy as np
from affine import Affine
from shapely.geometry import box

import pygltflib

from topology_geo.geometry.buildings import CONFIDENCE_FACT, SOURCE_OSM, BuildingSolid
from topology_geo.geometry.roofs import ROOF_FLAT, ROOF_GABLED
from topology_geo.relief.service import Grid
from topology_geo.tiling.grid import LOD0, LOD1, LOD2, TileIndex, world_to_tile_index
from topology_geo.tiling.tile_content import _building_mesh_for_lod, build_tile_content

CENTER_X, CENTER_Y = 10_000.0, 20_000.0
PLANE_A, PLANE_B, PLANE_C = 0.01, 0.02, 100.0
PIXEL = 2.0


def _flat_relief_grid(half_extent: float = 300.0) -> tuple[np.ndarray, Grid]:
    size = int(2 * half_extent / PIXEL)
    transform = Affine(PIXEL, 0.0, CENTER_X - half_extent, 0.0, -PIXEL, CENTER_Y + half_extent)
    grid = Grid(transform=transform, width=size, height=size, crs="EPSG:3857")
    rows, cols = np.indices((size, size))
    xs = transform.a * (cols + 0.5) + transform.c
    ys = transform.e * (rows + 0.5) + transform.f
    values = PLANE_A * xs + PLANE_B * ys + PLANE_C
    return values, grid


def _center_tile() -> TileIndex:
    return world_to_tile_index(zone=2, world_x=CENTER_X, world_y=CENTER_Y)


def _flat_building(local_x=0.0, local_y=0.0, osm_id=1, roof_shape=ROOF_FLAT) -> BuildingSolid:
    return BuildingSolid(
        osm_id=osm_id, footprint=box(local_x - 5, local_y - 5, local_x + 5, local_y + 5), height_m=9.0,
        height_confidence=CONFIDENCE_FACT, height_source=SOURCE_OSM, base_z=100.0,
        building_type="жилой", roof_shape=roof_shape, roof_height_m=2.5 if roof_shape != ROOF_FLAT else 0.0,
        roof_height_confidence=CONFIDENCE_FACT,
    )


def test_build_tile_content_terrain_only_returns_local_coordinates():
    values, grid = _flat_relief_grid()
    tile = _center_tile()
    result = build_tile_content(values, grid, tile, LOD2, [], center_x=CENTER_X, center_y=CENTER_Y)

    assert result is not None
    assert result.z_max >= result.z_min


def test_build_tile_content_glb_is_well_formed():
    values, grid = _flat_relief_grid()
    tile = _center_tile()
    result = build_tile_content(values, grid, tile, LOD2, [], center_x=CENTER_X, center_y=CENTER_Y)
    assert result.glb_bytes[:4] == b"glTF"


def test_build_tile_content_returns_none_when_empty():
    values, grid = _flat_relief_grid()
    far_tile = TileIndex(zone=2, tx=10_000, ty=10_000)  # далеко за пределами растра
    result = build_tile_content(values, grid, far_tile, LOD0, [], center_x=CENTER_X, center_y=CENTER_Y)
    assert result is None


def test_build_tile_content_includes_building_in_tile():
    values, grid = _flat_relief_grid()
    tile = _center_tile()
    building = _flat_building()
    result = build_tile_content(values, grid, tile, LOD2, [building], center_x=CENTER_X, center_y=CENTER_Y)
    assert result is not None
    assert result.z_max > result.z_min  # здание выше базовой отметки


def test_lod0_building_mesh_is_flat_footprint():
    # Меш здания напрямую (без рельефа тайла, который иначе доминирует
    # общий z_min/z_max в build_tile_content) - LOD0 плоский, LOD1 - призма
    # на всю высоту.
    building = _flat_building()
    verts0, faces0 = _building_mesh_for_lod(building, LOD0)
    verts1, faces1 = _building_mesh_for_lod(building, LOD1)
    assert faces0 and faces1
    z0 = [v[2] for v in verts0]
    z1 = [v[2] for v in verts1]
    assert max(z0) - min(z0) == 0.0  # LOD0 - все вершины на одной отметке base_z
    assert max(z1) - min(z1) > 0.0  # LOD1 - призма на всю height_m


def test_gabled_lod2_mesh_taller_than_lod1_block():
    building = _flat_building(roof_shape=ROOF_GABLED)
    verts1, _ = _building_mesh_for_lod(building, LOD1)
    verts2, _ = _building_mesh_for_lod(building, LOD2)
    assert max(v[2] for v in verts2) >= max(v[2] for v in verts1)


def test_build_tile_content_terrain_vertices_offset_to_local_origin():
    values, grid = _flat_relief_grid()
    tile = _center_tile()
    result = build_tile_content(values, grid, tile, LOD2, [], center_x=CENTER_X, center_y=CENTER_Y)
    assert result is not None

    gltf = pygltflib.GLTF2.load_from_bytes(result.glb_bytes)
    pos_accessor = next(a for a in gltf.accessors if a.type == "VEC3")
    # Вершины смещены на -(center_x, center_y) - локальные координаты тайла
    # (в мировых МСК-59 x/y были бы ~10000/20000, что сразу видно, если
    # смещение не применено).
    assert abs(pos_accessor.min[0]) < 300.0
    assert abs(pos_accessor.min[1]) < 300.0
    assert abs(pos_accessor.max[0]) < 300.0
    assert abs(pos_accessor.max[1]) < 300.0
