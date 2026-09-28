"""Универсальный энкодер треугольных мешей в GLB (Шаг 2.11, содержимое тайлов
3D Tiles).

`ifc/to_glb.py` (Шаг 1.9) не переиспользуется напрямую: там источник геометрии
— итератор `ifcopenshell.geom` по реальной IFC-модели, а контент тайла —
сырые (verts, faces) уже готовых мешей (рельеф тайла/здания в тайле, Шаг
2.11), IFC-модели для тайла не существует. Группировка узлов по категориям
(«Рельеф»/«Здания») для дерева объектов вьюера — тот же приём, что и в
`ifc/to_glb.py`, код по структуре похож, но независим (нет общего источника
итерации, из которого его можно было бы буквально вынести без разрастания
параметров)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    GLTF2,
    SCALAR,
    UNSIGNED_INT,
    VEC3,
    Accessor,
    Buffer,
    BufferView,
    Mesh,
    Node,
    Primitive,
    Scene,
)


@dataclass
class MeshPart:
    name: str
    category: str
    vertices: Any  # (N, 3) float
    faces: Any  # (M, 3) int
    extras: dict = field(default_factory=dict)


def build_glb(parts: list[MeshPart]) -> bytes:
    """Собрать один самодостаточный `.glb` из частей, сгруппированных по
    `category` (узлы-папки — «слои» дерева объектов вьюера, тот же приём,
    что `ifc/to_glb.py`, Шаг 1.9, п. 2)."""
    gltf = GLTF2()
    gltf.asset.generator = "topology-geo tiling.gltf_mesh (Шаг 2.11)"
    binary_blob = bytearray()
    category_node_index: dict[str, int] = {}

    def category_node(category: str) -> int:
        if category not in category_node_index:
            index = len(gltf.nodes)
            gltf.nodes.append(Node(name=category, children=[]))
            category_node_index[category] = index
        return category_node_index[category]

    for part in parts:
        positions = np.asarray(part.vertices, dtype=np.float32).reshape(-1, 3)
        indices = np.asarray(part.faces, dtype=np.uint32).reshape(-1)
        if len(positions) == 0 or len(indices) == 0:
            continue

        pos_offset = len(binary_blob)
        binary_blob.extend(positions.tobytes())
        idx_offset = len(binary_blob)
        binary_blob.extend(indices.tobytes())

        pos_view = len(gltf.bufferViews)
        gltf.bufferViews.append(
            BufferView(buffer=0, byteOffset=pos_offset, byteLength=positions.nbytes, target=ARRAY_BUFFER)
        )
        idx_view = len(gltf.bufferViews)
        gltf.bufferViews.append(
            BufferView(buffer=0, byteOffset=idx_offset, byteLength=indices.nbytes, target=ELEMENT_ARRAY_BUFFER)
        )

        pos_accessor = len(gltf.accessors)
        gltf.accessors.append(
            Accessor(
                bufferView=pos_view, componentType=FLOAT, count=len(positions), type=VEC3,
                min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist(),
            )
        )
        idx_accessor = len(gltf.accessors)
        gltf.accessors.append(Accessor(bufferView=idx_view, componentType=UNSIGNED_INT, count=len(indices), type=SCALAR))

        mesh_index = len(gltf.meshes)
        gltf.meshes.append(Mesh(primitives=[Primitive(attributes={"POSITION": pos_accessor}, indices=idx_accessor)]))

        node_index = len(gltf.nodes)
        gltf.nodes.append(Node(name=part.name, mesh=mesh_index, extras=part.extras or None))
        gltf.nodes[category_node(part.category)].children.append(node_index)

    root_index = len(gltf.nodes)
    gltf.nodes.append(Node(name="Тайл", children=list(category_node_index.values())))
    gltf.scenes.append(Scene(nodes=[root_index]))
    gltf.scene = 0
    gltf.buffers.append(Buffer(byteLength=len(binary_blob)))
    gltf.set_binary_blob(bytes(binary_blob))
    return b"".join(gltf.save_to_bytes())
