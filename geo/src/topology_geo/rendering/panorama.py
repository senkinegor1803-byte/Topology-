"""Панорамы 360° и видео облёта (Шаг 4.7).

Сферический кадр (п. 1: «сферические кадры из точек пешехода и с
высоты») — настоящий Cycles-рендер панорамной камеры
(`camera.type = "PANO"`, `panorama_type = "EQUIRECTANGULAR"`), не склейка
нескольких обычных кадров кубической развёрткой (что потребовало бы
отдельного шва/бленда на стыках граней). Точки съёмки — те же, что уже
считает Шаг 4.4 (`camera_shots.pedestrian_view_points`/`bird_eye_cameras`),
здесь используется только их `position` (панорамная камера снимает во
всех направлениях сразу, `target`/`fov_deg` этих объектов не нужны).

Видео облёта (п. 3) — реальная кодировка H.264 через системный `ffmpeg`
(кадры даёт `camera_shots.flythrough_path`, Шаг 4.4), сцена и камера
рендерятся ОДИН раз на весь ролик (не через `render_frame` per-кадр — тот
заново переимпортирует GLB на каждый вызов, что для сотен кадров ролика
означало бы сотни лишних импортов одной и той же сцены).
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

from topology_geo.rendering.camera_shots import Camera
from topology_geo.rendering.render_frame import QUALITY_DRAFT, QUALITY_PRESETS

# Эквидистантная (equirectangular) проекция ТРЕБУЕТ соотношение сторон
# ровно 2:1 - иначе полюса/экватор сферы исказятся при разворачивании.
DEFAULT_PANORAMA_RESOLUTION = (4096, 2048)


def render_panorama(
    glb_path: Path, position: tuple[float, float, float], output_path: Path, *,
    quality: str = QUALITY_DRAFT.name, sun_direction: tuple[float, float, float] | None = None,
    resolution: tuple[int, int] = DEFAULT_PANORAMA_RESOLUTION,
) -> Path:
    """Отрендерить сферическую панораму 360° из точки `position`."""
    try:
        import bpy
    except ImportError as exc:
        raise RuntimeError(
            "для рендера панорамы нужен пакет bpy (Blender Python), "
            "не входит в основные зависимости — установите отдельно: pip install bpy"
        ) from exc

    width, height = resolution
    if width != 2 * height:
        raise ValueError(f"эквидистантная панорама требует соотношение сторон 2:1, получено {width}x{height}")
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"неизвестный пресет качества {quality!r}, ожидается один из {list(QUALITY_PRESETS)}")
    preset = QUALITY_PRESETS[quality]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = preset.samples
    scene.cycles.use_denoising = preset.use_denoising
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    _correct_gltf_up_axis()

    if sun_direction is not None:
        _add_sun_light(sun_direction)
    else:
        _add_world_background(scene)

    camera_data = bpy.data.cameras.new("Панорама")
    camera_data.type = "PANO"
    camera_data.panorama_type = "EQUIRECTANGULAR"
    camera_object = bpy.data.objects.new("Панорама", camera_data)
    bpy.context.collection.objects.link(camera_object)
    camera_object.location = position
    scene.camera = camera_object

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    return output_path


def render_flythrough_video(
    glb_path: Path, cameras: list[Camera], output_path: Path, *,
    quality: str = QUALITY_DRAFT.name, fps: int = 30,
    sun_direction: tuple[float, float, float] | None = None,
    resolution_override: tuple[int, int] | None = None,
) -> Path:
    """Видео облёта: кадры-камеры даёт `camera_shots.flythrough_path`
    (Шаг 4.4), каждый рендерится (сцена импортируется ОДИН раз на весь
    ролик), PNG-последовательность кодируется в MP4 системным `ffmpeg`."""
    try:
        import bpy
    except ImportError as exc:
        raise RuntimeError(
            "для рендера видео нужен пакет bpy (Blender Python), "
            "не входит в основные зависимости — установите отдельно: pip install bpy"
        ) from exc
    if not cameras:
        raise ValueError("список камер пуст — нечего рендерить")
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"неизвестный пресет качества {quality!r}, ожидается один из {list(QUALITY_PRESETS)}")

    from mathutils import Vector

    preset = QUALITY_PRESETS[quality]
    width, height = resolution_override or preset.resolution

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = preset.samples
    scene.cycles.use_denoising = preset.use_denoising
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    _correct_gltf_up_axis()

    if sun_direction is not None:
        _add_sun_light(sun_direction)
    else:
        _add_world_background(scene)

    camera_data = bpy.data.cameras.new("Облёт")
    camera_object = bpy.data.objects.new("Облёт", camera_data)
    bpy.context.collection.objects.link(camera_object)
    scene.camera = camera_object

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_dir = Path(tmp_dir)
        for i, camera in enumerate(cameras):
            camera_data.angle = math.radians(camera.fov_deg)
            camera_object.location = camera.position
            direction = Vector(camera.target) - Vector(camera.position)
            if direction.length > 0:
                camera_object.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = str(frame_dir / f"frame_{i:05d}.png")
            bpy.ops.render.render(write_still=True)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frame_dir / "frame_%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path),
            ],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg завершился с ошибкой (код {result.returncode}): {result.stderr}")

    return output_path


def _correct_gltf_up_axis() -> None:
    """Та же компенсация, что в `usd_scene.py`/`lightmap_bake.py`/
    `render_frame.py` (Шаги 4.1/4.2/4.5): glTF по спецификации Y-вверх,
    проект — Z-вверх."""
    import bpy

    for obj in bpy.data.objects:
        if obj.parent is None:
            obj.rotation_mode = "XYZ"
            obj.rotation_euler.x -= math.radians(90.0)
    bpy.context.view_layer.update()


def _add_sun_light(sun_direction: tuple[float, float, float]):
    import bpy
    from mathutils import Vector

    light_data = bpy.data.lights.new("Солнце", type="SUN")
    light_data.energy = 3.0
    light_object = bpy.data.objects.new("Солнце", light_data)
    bpy.context.collection.objects.link(light_object)
    to_sun = Vector(sun_direction).normalized()
    light_object.rotation_euler = (-to_sun).to_track_quat("-Z", "Y").to_euler()
    return light_object


def _add_world_background(scene) -> None:
    import bpy

    world = bpy.data.worlds.new("Мир")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    scene.world = world
