"""Запекание освещения для веб-режима (Шаг 4.2).

«Запекание теней и окклюзии в текстуры ближнего кольца» (буквально по
плану) — здесь в вершинные цвета (`COLOR_0` в терминах glTF), не в
текстурный атлас: процедурная геометрия проекта (рельеф, здания-коробки,
дороги — Шаги 1.5-1.7/2.2/2.3) не имеет разверток UV (см. докстринг
`ifc/to_glb.py`: «нормали не записываются — вьюер считает их сам»), а
атлас без разметки UV пришлось бы либо разворачивать заново (лишний шаг,
искажения на процедурных мешах), либо строить руками. Вершинные цвета —
стандартная, реально применяемая на практике техника именно для такого
случая (крупная процедурная геометрия окружения, не единичный
детализированный объект под текстуру) — glTF 2.0 несёт их нативно как
атрибут примитива `COLOR_0`, вьюер (three.js) читает их без доп. кода.

Запекание — через `bpy` (Blender Cycles, CPU-бэкенд): реальный рендер
occlusion/shadow-проходов, не имитация. `bpy` — тяжёлая (374 МБ)
опциональная зависимость, не входит в `pyproject.toml` (см.
`docs/rendering.md`); импортируется только внутри функций этого модуля.

Итог — снова `.glb`: тот же формат, что уже используют тайлы 3D Tiles
проекта (`tiling/tile_content.py`, Шаг 2.1/2.11) — «экспорт в 3D
Tiles/GLB» по плану выполняется без отдельного шага конвертации формата,
запечённый `.glb` — само по себе валидное содержимое тайла.
"""

from __future__ import annotations

import math
from pathlib import Path

AO_LAYER = "AO"
SHADOW_LAYER = "Shadow"
BAKED_LAYER = "Baked"  # активный вершинный слой при экспорте — AO * Shadow

DEFAULT_AO_SAMPLES = 64
DEFAULT_SHADOW_SAMPLES = 64
DEFAULT_SUN_ENERGY = 3.0


def bake_lighting_to_vertex_colors(
    glb_path: Path, output_path: Path, *,
    sun_direction: tuple[float, float, float] | None = None,
    ao_samples: int = DEFAULT_AO_SAMPLES, shadow_samples: int = DEFAULT_SHADOW_SAMPLES,
) -> None:
    """Запечь окклюзию (всегда) и тень от солнца (если передано
    `sun_direction` — единичный вектор «от точки на сцене к солнцу», тот
    же формат, что возвращает `sun_position.sun_direction_vector`, Шаг
    4.3) в вершинные цвета всех мешей `.glb`-файла, записать результат в
    `output_path`. Без `sun_direction` — только окклюзия (нет
    привязанного к дате/времени освещения)."""
    try:
        import bpy
    except ImportError as exc:
        raise RuntimeError(
            "для запекания освещения нужен пакет bpy (Blender Python), "
            "не входит в основные зависимости — установите отдельно: pip install bpy"
        ) from exc

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "CPU"
    bpy.ops.import_scene.gltf(filepath=str(glb_path))

    # та же компенсация поворота, что в usd_scene.glb_layer_to_usd (Шаг
    # 4.1): glTF по спецификации Y-вверх, проект — Z-вверх.
    for obj in bpy.data.objects:
        if obj.parent is None:
            obj.rotation_mode = "XYZ"
            obj.rotation_euler.x -= math.radians(90.0)
    bpy.context.view_layer.update()

    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise ValueError("в GLB нет ни одного меша — запекать нечего")

    for obj in mesh_objects:
        if not obj.data.materials:
            obj.data.materials.append(bpy.data.materials.new(f"BakeMaterial_{obj.name}"))
        for material in obj.data.materials:
            material.use_nodes = True
        for layer_name in (AO_LAYER, SHADOW_LAYER, BAKED_LAYER):
            if layer_name not in obj.data.vertex_colors:
                obj.data.vertex_colors.new(name=layer_name)

    bpy.context.scene.cycles.samples = ao_samples
    _bake_pass(mesh_objects, "AO", AO_LAYER)

    has_shadow = sun_direction is not None
    if has_shadow:
        _add_sun_light(sun_direction)
        bpy.context.scene.cycles.samples = shadow_samples
        _bake_pass(mesh_objects, "SHADOW", SHADOW_LAYER)

    _combine_vertex_colors(mesh_objects, has_shadow=has_shadow)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.data.vertex_colors.active = obj.data.vertex_colors[BAKED_LAYER]
        obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_path), export_format="GLB", use_selection=True, export_vertex_color="ACTIVE",
    )


def _bake_pass(mesh_objects: list, bake_type: str, layer_name: str) -> None:
    import bpy

    for obj in mesh_objects:
        obj.data.vertex_colors.active = obj.data.vertex_colors[layer_name]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.context.scene.cycles.bake_type = bake_type
    bpy.ops.object.bake(type=bake_type, target="VERTEX_COLORS")


def _add_sun_light(sun_direction: tuple[float, float, float]):
    """Солнце Cycles светит вдоль локальной оси -Z объекта-лампы;
    направление `sun_direction` («к солнцу») даёт направление лучей
    «от солнца» — `-sun_direction` — которое и нужно совместить с -Z."""
    import bpy
    from mathutils import Vector

    light_data = bpy.data.lights.new("Солнце", type="SUN")
    light_data.energy = DEFAULT_SUN_ENERGY
    light_obj = bpy.data.objects.new("Солнце", light_data)
    bpy.context.collection.objects.link(light_obj)
    to_sun = Vector(sun_direction).normalized()
    light_obj.rotation_euler = (-to_sun).to_track_quat("-Z", "Y").to_euler()
    return light_obj


def _combine_vertex_colors(mesh_objects: list, *, has_shadow: bool) -> None:
    """`Baked` = AO (всегда) × Shadow (если запекалась) — произведение
    двух независимых коэффициентов затенения в [0,1], стандартный способ
    объединить окклюзию и прямую тень в одно значение освещённости."""
    for obj in mesh_objects:
        ao_data = obj.data.vertex_colors[AO_LAYER].data
        baked_data = obj.data.vertex_colors[BAKED_LAYER].data
        shadow_data = obj.data.vertex_colors[SHADOW_LAYER].data if has_shadow else None
        for i in range(len(baked_data)):
            ao_color = ao_data[i].color
            if shadow_data is not None:
                shadow_color = shadow_data[i].color
                baked_data[i].color = (
                    ao_color[0] * shadow_color[0],
                    ao_color[1] * shadow_color[1],
                    ao_color[2] * shadow_color[2],
                    1.0,
                )
            else:
                baked_data[i].color = ao_color
