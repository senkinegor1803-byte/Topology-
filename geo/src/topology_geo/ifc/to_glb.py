"""Конвертация IFC → GLB для веб-просмотра (Шаг 1.9, п. 1).

Источник геометрии — не повторная генерация из `SiteModel`, а сама IFC-модель
(`ifcopenshell.geom.iterator`, мировые координаты): GLB — производный формат
для отображения, а IFC остаётся единственным источником истины (можно
переоткрыть `site.ifc` в Renga, поправить и переконвертировать в GLB заново,
не трогая остальной конвейер).

Каждый объект IFC -> один узел glTF с мешем (позиции + индексы; нормали не
записываются — вьюер считает их сам, `computeVertexNormals`, чего достаточно
для процедурных объектов упрощённого уровня детализации Этапа 1) и `extras`
(GlobalId, класс IFC, наборы свойств по словарю данных) — для панели свойств
по клику (п. 2 шага). Узлы группируются по категориям (рельеф/здания/дороги/
вода/рельсы/деревья) в узлы-папки — это и есть «слои» и «дерево объектов»
из критерия шага, без отдельного файла метаданных: один самодостаточный .glb.

XKT (xeokit-convert) из плана не реализован: это отдельный Node.js-инструмент
конвертации из чужой закрытой SDK-экосистемы (xeokit) — GLB общедоступен,
открыт, штатно грузится three.js (см. `viewer/`) и достаточен для критерия
шага («модель открывается в браузере и на телефоне»); добавить конвертацию
XKT можно отдельным шагом, не меняя эту функцию.
"""

from __future__ import annotations

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
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

EXCLUDED_IFC_CLASSES = ("IfcSpace", "IfcOpeningElement")

CATEGORY_TERRAIN = "Рельеф"
CATEGORY_BUILDINGS = "Здания"
CATEGORY_ROADS = "Дороги"
CATEGORY_WATER = "Вода"
CATEGORY_RAIL = "Рельсы"
CATEGORY_TREES = "Деревья"
CATEGORY_OTHER = "Прочее"


def _categorize(element: ifcopenshell.entity_instance.entity_instance) -> str:
    """Слой объекта для группировки узлов glTF (папки во «дереве объектов»,
    переключаемые слои во вьюере) — по классу IFC, с уточнением по имени там,
    где один класс IFC используется для нескольких слоёв Шага 1.8 (прокси)."""
    ifc_class = element.is_a()
    name = element.Name or ""
    if ifc_class == "IfcRoad" or name.startswith("Дорога"):
        return CATEGORY_ROADS
    if ifc_class == "IfcGeographicElement":
        predefined = getattr(element, "PredefinedType", None)
        if predefined == "TERRAIN":
            return CATEGORY_TERRAIN
        if name.startswith("Дерево"):
            return CATEGORY_TREES
        if name.startswith(("Водоём", "Водоток")):
            return CATEGORY_WATER
        return CATEGORY_OTHER
    if name.startswith("Здание"):
        return CATEGORY_BUILDINGS
    if name.startswith("Ж/д"):
        return CATEGORY_RAIL
    return CATEGORY_OTHER


def _clean_psets(element: ifcopenshell.entity_instance.entity_instance) -> dict:
    psets = ifcopenshell.util.element.get_psets(element)
    for properties in psets.values():
        properties.pop("id", None)
    return psets


def convert_ifc_to_glb(model: ifcopenshell.file) -> bytes:
    """Собрать GLB из всех объектов с геометрией в `model` (мировые
    координаты). Возвращает бинарное содержимое `.glb`-файла (одним куском —
    JSON + бинарный буфер в одном контейнере, см. `GLTF2.save_to_bytes`)."""
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    iterator = ifcopenshell.geom.iterator(settings, model, exclude=list(EXCLUDED_IFC_CLASSES))

    gltf = GLTF2()
    gltf.asset.generator = "topology-geo ifc.to_glb (Шаг 1.9)"
    binary_blob = bytearray()
    category_node_index: dict[str, int] = {}

    def category_node(category: str) -> int:
        if category not in category_node_index:
            index = len(gltf.nodes)
            gltf.nodes.append(Node(name=category, children=[]))
            category_node_index[category] = index
        return category_node_index[category]

    if iterator.initialize():
        while True:
            shape = iterator.get()
            element = model.by_id(shape.id)
            verts = shape.geometry.verts
            faces = shape.geometry.faces
            if verts and faces:
                positions = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
                indices = np.asarray(faces, dtype=np.uint32)

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
                gltf.accessors.append(
                    Accessor(bufferView=idx_view, componentType=UNSIGNED_INT, count=len(indices), type=SCALAR)
                )

                mesh_index = len(gltf.meshes)
                gltf.meshes.append(Mesh(primitives=[Primitive(attributes={"POSITION": pos_accessor}, indices=idx_accessor)]))

                node_index = len(gltf.nodes)
                gltf.nodes.append(
                    Node(
                        name=element.Name or element.is_a(),
                        mesh=mesh_index,
                        extras={
                            "globalId": element.GlobalId,
                            "ifcClass": element.is_a(),
                            "psets": _clean_psets(element),
                        },
                    )
                )
                gltf.nodes[category_node(_categorize(element))].children.append(node_index)

            if not iterator.next():
                break

    root_index = len(gltf.nodes)
    gltf.nodes.append(Node(name="Участок", children=list(category_node_index.values())))
    gltf.scenes.append(Scene(nodes=[root_index]))
    gltf.scene = 0
    gltf.buffers.append(Buffer(byteLength=len(binary_blob)))
    gltf.set_binary_blob(bytes(binary_blob))

    return b"".join(gltf.save_to_bytes())
