"""Рендер-воркер, Blender Cycles (Шаг 4.5).

Реальный Cycles-рендер (CPU-бэкенд, `bpy.context.scene.cycles.device =
"CPU"`) одного кадра из готовой сцены (`.glb`, Шаг 4.1/4.2) по камере
(`rendering.camera_shots.Camera`, Шаг 4.4). Действие плана «Docker-образ
Blender с GPU» — по прежнему требует настоящего GPU-хоста для сборки и
проверки образа; здесь и в `infra/render-worker.Dockerfile` (обычный
CPU-рендер, ставится в очередь и выполняется реальным `bpy`) построена и
проверена вся логика ВОКРУГ рендера — качество, очередь, приоритеты,
формат вывода — честно без GPU-специфичной части (сборка образа с CUDA
и его запуск на реальной видеокарте не выполнялась и не проверялась в
этой среде, см. `docs/rendering.md`).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

from topology_geo.rendering.camera_shots import Camera

OUTPUT_FORMAT_PNG = "PNG"
OUTPUT_FORMAT_EXR = "OPEN_EXR"


@dataclass(frozen=True)
class QualityPreset:
    name: str
    samples: int
    resolution: tuple[int, int]
    use_denoising: bool


# «черновик (быстро), финал (4К, шумоподавление)» — буквально по плану, п. 2.
QUALITY_DRAFT = QualityPreset(name="draft", samples=32, resolution=(960, 540), use_denoising=False)
QUALITY_FINAL = QualityPreset(name="final", samples=128, resolution=(3840, 2160), use_denoising=True)
QUALITY_PRESETS = {QUALITY_DRAFT.name: QUALITY_DRAFT, QUALITY_FINAL.name: QUALITY_FINAL}


@dataclass(frozen=True)
class RenderResult:
    output_path: Path
    width: int
    height: int
    samples: int
    elapsed_s: float


def render_frame(
    glb_path: Path, camera: Camera, output_path: Path, *,
    quality: str = QUALITY_DRAFT.name, output_format: str = OUTPUT_FORMAT_PNG,
    sun_direction: tuple[float, float, float] | None = None,
    resolution_override: tuple[int, int] | None = None,
) -> RenderResult:
    """Отрендерить один кадр сцены `glb_path` камерой `camera` в
    `output_path`. `quality` — ключ `QUALITY_PRESETS` (`"draft"`/
    `"final"`); `resolution_override` меняет только разрешение, оставляя
    сэмплы/шумоподавление пресета (нужно для «превью в кабинете» — Шаг
    4.10 — которому не нужно полное 4K черновика)."""
    try:
        import bpy
    except ImportError as exc:
        raise RuntimeError(
            "для рендера кадра нужен пакет bpy (Blender Python), "
            "не входит в основные зависимости — установите отдельно: pip install bpy"
        ) from exc

    if quality not in QUALITY_PRESETS:
        raise ValueError(f"неизвестный пресет качества {quality!r}, ожидается один из {list(QUALITY_PRESETS)}")
    preset = QUALITY_PRESETS[quality]
    width, height = resolution_override or preset.resolution

    started_at = time.monotonic()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = preset.samples
    scene.cycles.use_denoising = preset.use_denoising
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = output_format

    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    # та же компенсация поворота, что в usd_scene.glb_layer_to_usd и
    # lightmap_bake.py (Шаги 4.1/4.2): glTF по спецификации Y-вверх,
    # проект - Z-вверх; после компенсации мировые координаты сцены
    # совпадают с локальными координатами проекта, в которых уже
    # посчитана камера (`camera_shots.py`, Шаг 4.4).
    for obj in bpy.data.objects:
        if obj.parent is None:
            obj.rotation_mode = "XYZ"
            obj.rotation_euler.x -= math.radians(90.0)
    bpy.context.view_layer.update()

    if sun_direction is not None:
        _add_sun_light(sun_direction)
    else:
        # без солнца сцена рендерится чёрной (Cycles - физический рендер,
        # без источника света объекты не освещены) - минимальный внешний
        # свет (мировой фон), чтобы кадр вообще был виден без явного солнца.
        world = bpy.data.worlds.new("Мир")
        world.use_nodes = True
        background = world.node_tree.nodes["Background"]
        background.inputs["Strength"].default_value = 1.0
        scene.world = world

    camera_object = _make_camera(camera)
    scene.camera = camera_object

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)

    elapsed_s = time.monotonic() - started_at
    return RenderResult(output_path=output_path, width=width, height=height, samples=preset.samples, elapsed_s=elapsed_s)


def _make_camera(camera: Camera):
    import bpy
    from mathutils import Vector

    camera_data = bpy.data.cameras.new("Камера")
    camera_data.angle = math.radians(camera.fov_deg)
    camera_object = bpy.data.objects.new("Камера", camera_data)
    bpy.context.collection.objects.link(camera_object)
    camera_object.location = camera.position

    direction = (Vector(camera.target) - Vector(camera.position))
    if direction.length > 0:
        camera_object.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return camera_object


def _add_sun_light(sun_direction: tuple[float, float, float]):
    """Та же ориентация лампы, что в `lightmap_bake._add_sun_light`
    (Шаг 4.2): солнце светит вдоль локальной -Z, направление на солнце
    задаёт противоположный вектор."""
    import bpy
    from mathutils import Vector

    light_data = bpy.data.lights.new("Солнце", type="SUN")
    light_data.energy = 3.0
    light_object = bpy.data.objects.new("Солнце", light_data)
    bpy.context.collection.objects.link(light_object)
    to_sun = Vector(sun_direction).normalized()
    light_object.rotation_euler = (-to_sun).to_track_quat("-Z", "Y").to_euler()
    return light_object
