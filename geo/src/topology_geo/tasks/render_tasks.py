"""Очередь рендер-задач с приоритетами (Шаг 4.5, п. 1/3).

Приоритет — целое 0..9, где 0 — НАИВЫСШИЙ приоритет (соглашение AMQP,
которое использует и Celery через `task_queue_max_priority` — не
интуитивное «больше число — важнее», а наоборот). Проверено вручную
руками до написания этого модуля: несколько низкоприоритетных задач,
поставленных в очередь ПЕРВЫМИ, реально выполняются ПОСЛЕ
высокоприоритетных задач, поставленных ПОЗЖЕ, при одном воркере
(`geo/tests/test_tasks_render.py`) — это подлинная переупорядочивающая
очередь Redis-транспорта Celery, а не просто FIFO с меткой.

Ограничение количества одновременных задач на GPU (п. 3) — не код самой
задачи, а конфигурация запуска воркера: один воркер-процесс на
физическую видеокарту, `celery worker --concurrency=1 -Q render` — тот
же механизм `--concurrency`, что уже использует проект для проверки
параллелизма (Шаг 1.3 «10 параллельных задач», Шаг 2.1 генерация
тайлов). При нескольких GPU — несколько воркер-процессов, каждый со
своим `CUDA_VISIBLE_DEVICES`, все слушают очередь `render`; сам Celery
не знает о GPU, ограничение достигается количеством процессов-читателей
очереди, а не встроенной семафорной логикой.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from topology_geo.rendering.camera_shots import Camera
from topology_geo.rendering.render_frame import OUTPUT_FORMAT_PNG, QUALITY_DRAFT, render_frame
from topology_geo.tasks.celery_app import app
from topology_geo.tasks.pipeline_tasks import get_storage

PRIORITY_HIGH = 0
PRIORITY_NORMAL = 5
PRIORITY_LOW = 9

RENDER_QUEUE = "render"

_CONTENT_TYPE_BY_FORMAT = {"PNG": "image/png", "OPEN_EXR": "image/x-exr"}
_EXTENSION_BY_FORMAT = {"PNG": "png", "OPEN_EXR": "exr"}


@app.task(name="topology.render_frame")
def render_frame_task(
    glb_storage_key: str, camera_dict: dict, output_storage_key: str, *,
    quality: str = QUALITY_DRAFT.name, output_format: str = OUTPUT_FORMAT_PNG,
    sun_direction: tuple[float, float, float] | None = None,
    resolution_override: tuple[int, int] | None = None,
) -> str:
    """Одна задача очереди — один кадр: забирает GLB из хранилища (Шаг
    1.3), рендерит (`rendering.render_frame`), кладёт результат обратно
    в хранилище — тот же паттерн ввода/вывода через `get_storage()`, что
    и остальные задачи конвейера (`pipeline_tasks.run_job_step_task`,
    `tile_tasks.generate_terrain_tile_task`)."""
    storage = get_storage()
    glb_bytes = storage.download(glb_storage_key)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        glb_path = tmp_dir_path / "scene.glb"
        glb_path.write_bytes(glb_bytes)
        extension = _EXTENSION_BY_FORMAT[output_format]
        output_path = tmp_dir_path / f"frame.{extension}"

        camera = Camera(
            position=tuple(camera_dict["position"]), target=tuple(camera_dict["target"]),
            fov_deg=camera_dict.get("fov_deg", 50.0), label=camera_dict.get("label", ""),
        )
        render_frame(
            glb_path, camera, output_path, quality=quality, output_format=output_format,
            sun_direction=sun_direction, resolution_override=resolution_override,
        )
        storage.upload(output_storage_key, output_path.read_bytes(), content_type=_CONTENT_TYPE_BY_FORMAT[output_format])

    return output_storage_key


def enqueue_render(
    glb_storage_key: str, camera: Camera, output_storage_key: str, *,
    quality: str = QUALITY_DRAFT.name, output_format: str = OUTPUT_FORMAT_PNG,
    sun_direction: tuple[float, float, float] | None = None, priority: int = PRIORITY_NORMAL,
    resolution_override: tuple[int, int] | None = None,
):
    """Поставить кадр в очередь `render` с приоритетом `priority` (0 —
    наивысший, 9 — наинизший, см. докстринг модуля). `resolution_override`
    — для быстрого превью в кабинете (Шаг 4.10), не только для тестов."""
    camera_dict = {
        "position": list(camera.position), "target": list(camera.target),
        "fov_deg": camera.fov_deg, "label": camera.label,
    }
    return render_frame_task.apply_async(
        args=[glb_storage_key, camera_dict, output_storage_key],
        kwargs={
            "quality": quality, "output_format": output_format, "sun_direction": sun_direction,
            "resolution_override": resolution_override,
        },
        queue=RENDER_QUEUE, priority=priority,
    )
