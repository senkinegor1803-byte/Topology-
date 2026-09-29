"""Тест Шага 4.2: запекание освещения в вершинные цвета — единственная
часть, которой реально нужен `bpy` (Blender Cycles). Собирает настоящий
GLB тем же путём, что и остальной пайплайн (`ifc.to_glb`, Шаг 1.9): плоский
участок рельефа (TIN) с одним зданием на нём, запекает AO и тень от
солнца, проверяет результат напрямую через `pygltflib` (не через сам
`bpy`) — реальный физический эффект (тень под зданием темнее открытого
участка), а не просто «не упало».

`bpy` — тяжёлая (374 МБ) опциональная зависимость, не входит в
`pyproject.toml` (см. `docs/rendering.md`); модуль целиком пропускается,
если пакет не установлен."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

bpy = pytest.importorskip("bpy")

from pygltflib import GLTF2

from topology_geo.geometry.buildings import BuildingSolid
from topology_geo.ifc.assemble import BasePoint, SiteModel, build_site_ifc
from topology_geo.ifc.to_glb import convert_ifc_to_glb
from topology_geo.relief.tin import SiteTin
from topology_geo.rendering.lightmap_bake import bake_lighting_to_vertex_colors

BASE_POINT = BasePoint(lon=56.25, lat=58.0, zone=2, x=2_310_450.0, y=-5_857_320.0, height=150.0)


def _make_flat_tin(half_extent: float = 40.0, n: int = 25) -> SiteTin:
    xs, ys = np.meshgrid(np.linspace(-half_extent, half_extent, n), np.linspace(-half_extent, half_extent, n))
    zs = np.zeros_like(xs)
    vertices = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    delaunay = Delaunay(vertices[:, :2])
    return SiteTin(vertices=vertices, triangles=delaunay.simplices, _delaunay=delaunay)


def _make_terrain_and_building_glb() -> bytes:
    building = BuildingSolid(
        osm_id=1, footprint=Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)]),
        height_m=10.0, height_confidence="факт", height_source="OSM", base_z=0.0, building_type="жилой",
    )
    model = SiteModel(tin=_make_flat_tin(), buildings=[building])
    model_ifc, _ = build_site_ifc("IFC4", model, BASE_POINT)
    return convert_ifc_to_glb(model_ifc)


_COMPONENT_FORMATS = {
    5126: ("f", 4, 1.0),   # FLOAT
    5123: ("H", 2, 65535.0),  # UNSIGNED_SHORT, normalized
    5121: ("B", 1, 255.0),  # UNSIGNED_BYTE, normalized
}


def _read_color0(gltf: GLTF2, blob: bytes, mesh_index: int) -> list[tuple[float, float, float, float]]:
    """Blender по умолчанию пишет COLOR_0 как нормализованный
    UNSIGNED_SHORT (не FLOAT) — читаем по реальному componentType
    accessor'а, а не предполагаем формат."""
    primitive = gltf.meshes[mesh_index].primitives[0]
    accessor = gltf.accessors[primitive.attributes.COLOR_0]
    view = gltf.bufferViews[accessor.bufferView]
    offset = view.byteOffset or 0
    n_components = 4 if accessor.type == "VEC4" else 3
    fmt_char, comp_size, max_value = _COMPONENT_FORMATS[accessor.componentType]
    stride = comp_size * n_components
    colors = []
    for i in range(accessor.count):
        chunk = blob[offset + i * stride: offset + i * stride + stride]
        values = struct.unpack("<" + fmt_char * n_components, chunk)
        normalized = tuple(v / max_value for v in values)
        colors.append(normalized if len(normalized) == 4 else (*normalized, 1.0))
    return colors


def _mesh_positions(gltf: GLTF2, blob: bytes, mesh_index: int) -> list[tuple[float, float, float]]:
    primitive = gltf.meshes[mesh_index].primitives[0]
    accessor = gltf.accessors[primitive.attributes.POSITION]
    view = gltf.bufferViews[accessor.bufferView]
    offset = view.byteOffset or 0
    positions = []
    for i in range(accessor.count):
        chunk = blob[offset + i * 12: offset + i * 12 + 12]
        positions.append(struct.unpack("<3f", chunk))
    return positions


def test_bake_lighting_produces_real_shadow_under_building(tmp_path: Path):
    glb_path = tmp_path / "scene.glb"
    glb_path.write_bytes(_make_terrain_and_building_glb())
    output_path = tmp_path / "baked.glb"

    # солнце почти в зените с небольшим наклоном по X - тень ляжет узкой
    # полосой рядом со зданием, а не спрячется прямо под его крышей
    bake_lighting_to_vertex_colors(glb_path, output_path, sun_direction=(0.5, 0.0, 0.85), ao_samples=24, shadow_samples=24)

    assert output_path.exists() and output_path.stat().st_size > 0

    gltf = GLTF2.load(str(output_path))
    blob = gltf.binary_blob()
    assert blob is not None

    terrain_mesh_index = None
    for i, mesh in enumerate(gltf.meshes):
        if mesh.primitives[0].attributes.COLOR_0 is not None:
            positions = _mesh_positions(gltf, blob, i)
            xs = [p[0] for p in positions]
            if max(xs) - min(xs) > 30.0:  # рельеф - большой плоский меш (80x80 м)
                terrain_mesh_index = i
                break
    assert terrain_mesh_index is not None, "не нашли меш рельефа с запечёнными вершинными цветами"

    positions = _mesh_positions(gltf, blob, terrain_mesh_index)
    colors = _read_color0(gltf, blob, terrain_mesh_index)
    assert len(colors) == len(positions)

    near_building_values = []
    far_from_building_values = []
    for (x, y, z), color in zip(positions, colors):
        dist_from_building_center = max(abs(x), abs(y))
        brightness = sum(color[:3]) / 3.0
        if dist_from_building_center < 8.0:
            near_building_values.append(brightness)
        elif dist_from_building_center > 25.0:
            far_from_building_values.append(brightness)

    assert near_building_values and far_from_building_values
    near_avg = sum(near_building_values) / len(near_building_values)
    far_avg = sum(far_from_building_values) / len(far_from_building_values)
    # реальный физический эффект: у подножия 10-метрового здания темнее
    # (тень + окклюзия в углу между стеной и землёй), чем на открытом
    # участке рельефа далеко от здания
    assert near_avg < far_avg


def test_bake_lighting_without_sun_direction_only_bakes_ao(tmp_path: Path):
    glb_path = tmp_path / "scene.glb"
    glb_path.write_bytes(_make_terrain_and_building_glb())
    output_path = tmp_path / "baked_ao_only.glb"

    bake_lighting_to_vertex_colors(glb_path, output_path, sun_direction=None, ao_samples=16)

    gltf = GLTF2.load(str(output_path))
    blob = gltf.binary_blob()
    found_color = False
    for i, mesh in enumerate(gltf.meshes):
        if mesh.primitives[0].attributes.COLOR_0 is not None:
            colors = _read_color0(gltf, blob, i)
            assert len(colors) > 0
            found_color = True
    assert found_color


def test_bake_lighting_rejects_glb_with_no_meshes(tmp_path: Path):
    empty_model_ifc, _ = build_site_ifc("IFC4", SiteModel(), BASE_POINT)
    glb_path = tmp_path / "empty.glb"
    glb_path.write_bytes(convert_ifc_to_glb(empty_model_ifc))

    with pytest.raises(ValueError, match="нет ни одного меша"):
        bake_lighting_to_vertex_colors(glb_path, tmp_path / "out.glb", sun_direction=None)
