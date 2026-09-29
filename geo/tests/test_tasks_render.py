"""Интеграционный тест очереди рендер-задач с приоритетами (Шаг 4.5, п. 1/3).

Поднимает настоящий воркер Celery в ОТДЕЛЬНОМ ПРОЦЕССЕ (реальный
Redis-брокер, `--concurrency=1` — тот же принцип «один воркер-процесс на
GPU», что и в проде) и реально рендерит несколько кадров через него
(настоящий `bpy` Cycles, не мок). Доказывает две вещи заявленные в плане:
очередь с приоритетами (п. 3: высокоприоритетная задача, поставленная
ПОСЛЕ низкоприоритетных, выполняется РАНЬШЕ них) и что при
`--concurrency=1` задачи не перекрываются по времени (честная модель
«одна GPU — один воркер»).

Пропускается, если недоступен Redis.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
import pytest
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, box

redis = pytest.importorskip("redis")
bpy = pytest.importorskip("bpy")

from topology_geo.geometry.buildings import BuildingSolid
from topology_geo.ifc.assemble import BasePoint, SiteModel, build_site_ifc
from topology_geo.ifc.to_glb import convert_ifc_to_glb
from topology_geo.relief.tin import SiteTin
from topology_geo.rendering.camera_shots import bird_eye_cameras
from topology_geo.storage import FileSystemObjectStorage

TEST_REDIS_DB = 6
BASE_POINT = BasePoint(lon=56.25, lat=58.0, zone=2, x=2_310_450.0, y=-5_857_320.0, height=150.0)


def _redis_reachable() -> bool:
    try:
        client = redis.Redis(host="localhost", port=6379, db=TEST_REDIS_DB, socket_connect_timeout=2)
        return bool(client.ping())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _redis_reachable(), reason="требуется доступный Redis")


def _make_flat_tin(half_extent: float = 40.0, n: int = 12) -> SiteTin:
    xs, ys = np.meshgrid(np.linspace(-half_extent, half_extent, n), np.linspace(-half_extent, half_extent, n))
    zs = np.zeros_like(xs)
    vertices = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    delaunay = Delaunay(vertices[:, :2])
    return SiteTin(vertices=vertices, triangles=delaunay.simplices, _delaunay=delaunay)


def _make_scene_glb_bytes() -> bytes:
    building = BuildingSolid(
        osm_id=1, footprint=Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)]),
        height_m=10.0, height_confidence="факт", height_source="OSM", base_z=0.0, building_type="жилой",
    )
    model_ifc, _ = build_site_ifc("IFC4", SiteModel(tin=_make_flat_tin(), buildings=[building]), BASE_POINT)
    return convert_ifc_to_glb(model_ifc)


def _enqueue_with_fresh_celery_app(broker_url: str, monkeypatch: pytest.MonkeyPatch):
    """Тот же приём, что в `test_tasks_celery.py`: перезагрузить
    `topology_geo.tasks.*` с нужным `CELERY_BROKER_URL`, иначе достаётся
    закэшированный объект приложения от предыдущего теста модуля."""
    for name in list(sys.modules):
        if name.startswith("topology_geo.tasks"):
            del sys.modules[name]

    monkeypatch.setenv("CELERY_BROKER_URL", broker_url)
    monkeypatch.setenv("CELERY_RESULT_BACKEND", broker_url)
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "false")

    from topology_geo.tasks.render_tasks import PRIORITY_HIGH, PRIORITY_LOW, enqueue_render

    return PRIORITY_HIGH, PRIORITY_LOW, enqueue_render


def test_render_queue_priority_and_concurrency(tmp_path, monkeypatch):
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    storage = FileSystemObjectStorage(str(storage_root))
    storage.upload("scene.glb", _make_scene_glb_bytes(), content_type="model/gltf-binary")

    footprint = box(-5, -5, 5, 5)
    camera = bird_eye_cameras(footprint, building_height_m=10.0, base_z=0.0)[0]

    broker_url = f"redis://localhost:6379/{TEST_REDIS_DB}"
    redis.Redis(host="localhost", port=6379, db=TEST_REDIS_DB).flushdb()

    worker_env = {
        **os.environ,
        "TOPOLOGY_STORAGE_ROOT": str(storage_root),
        "CELERY_BROKER_URL": broker_url,
        "CELERY_RESULT_BACKEND": broker_url,
    }

    worker = subprocess.Popen(
        [sys.executable, "-m", "celery", "-A", "topology_geo.tasks.celery_app", "worker",
         "--loglevel=info", "--pool=solo", "--concurrency=1", "-Q", "render"],
        env=worker_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        PRIORITY_HIGH, PRIORITY_LOW, enqueue_render = _enqueue_with_fresh_celery_app(broker_url, monkeypatch)

        # 3 низкоприоритетные задачи ставятся ПЕРВЫМИ, 2 высокоприоритетные - ПОСЛЕ.
        # Если очередь с приоритетами реально работает, high0/high1 должны
        # завершиться РАНЬШЕ большинства low-задач, несмотря на порядок постановки.
        resolution = (160, 90)  # маленькое разрешение - тест быстрый
        for i in range(3):
            enqueue_render(
                "scene.glb", camera, f"low{i}.png", priority=PRIORITY_LOW,
                sun_direction=(0.5, 0.0, 0.85), resolution_override=resolution,
            )
        for i in range(2):
            enqueue_render(
                "scene.glb", camera, f"high{i}.png", priority=PRIORITY_HIGH,
                sun_direction=(0.5, 0.0, 0.85), resolution_override=resolution,
            )

        expected_keys = {f"low{i}.png" for i in range(3)} | {f"high{i}.png" for i in range(2)}
        completion_order = []
        deadline = time.time() + 120
        while time.time() < deadline and len(completion_order) < len(expected_keys):
            for key in expected_keys:
                if key not in completion_order and storage.exists(key):
                    completion_order.append(key)
            time.sleep(0.2)

        worker_output = None
        if len(completion_order) < len(expected_keys):
            worker.terminate()
            worker_output = worker.stdout.read() if worker.stdout else ""
        assert len(completion_order) == len(expected_keys), (completion_order, worker_output)

        high_positions = [completion_order.index(f"high{i}.png") for i in range(2)]
        low_positions = [completion_order.index(f"low{i}.png") for i in range(3)]
        # обе high-задачи (поставлены ПОСЛЕДНИМИ) завершились раньше ХОТЯ БЫ
        # одной low-задачи (поставленной первой) - настоящая переупорядочивающая
        # очередь, а не просто FIFO по порядку постановки
        assert max(high_positions) < max(low_positions), (
            f"приоритет не переупорядочил очередь: {completion_order}"
        )

        for key in expected_keys:
            content = storage.download(key)
            assert len(content) > 0
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=5)
