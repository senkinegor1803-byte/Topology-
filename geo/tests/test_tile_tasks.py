"""Тест параллельной генерации тайлов через настоящую очередь Celery+Redis
(Шаг 2.1, п. 3). Поднимает реальный воркер в отдельном процессе с
`--concurrency=4` (несколько worker-процессов Celery, реальный параллелизм,
а не последовательная имитация) и проверяет: (а) все тайлы батча собраны
правильно и стыкуются по границам, (б) повторный запрос того же батча берёт
тайлы из кеша, не пересчитывая (created_at в PostGIS не меняется)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

redis = pytest.importorskip("redis")

from topology_geo.tasks.tile_tasks import deserialize_tin
from topology_geo.tiling.cache import ensure_schema, find_cached_tile
from topology_geo.tiling.grid import TileIndex

TEST_REDIS_DB = 3
DATA_VERSION = "dem-2026-01"
GENERATOR_VERSION = "gen-v1"


def _redis_reachable() -> bool:
    try:
        client = redis.Redis(host="localhost", port=6379, db=TEST_REDIS_DB, socket_connect_timeout=2)
        return bool(client.ping())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _redis_reachable(), reason="требуется доступный Redis")


def _seed_relief_geotiff(path: Path) -> None:
    """Плоский растр 0..600 x 0..600 м (МСК-59, условная зона) - покрывает
    тайлы (0,0),(1,0),(0,1),(1,1) целиком, с запасом на сэмплирование границ."""
    pixel = 5.0
    minx, miny, maxx, maxy = -20.0, -20.0, 620.0, 620.0
    width = int((maxx - minx) / pixel)
    height = int((maxy - miny) / pixel)
    transform = Affine(pixel, 0.0, minx, 0.0, -pixel, maxy)

    rows_idx, cols_idx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    xs = transform.c + (cols_idx + 0.5) * transform.a
    ys = transform.f + (rows_idx + 0.5) * transform.e
    values = (100.0 + 0.01 * xs + 0.02 * ys).astype("float32")

    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:3857", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(values, 1)


def _enqueue_env(dbname: str, storage_root: Path, broker_url: str) -> dict:
    return {
        **os.environ,
        "POSTGRES_HOST": os.environ.get("POSTGRES_HOST", "localhost"),
        "POSTGRES_PORT": os.environ.get("POSTGRES_PORT", "5432"),
        "POSTGRES_USER": os.environ.get("POSTGRES_USER", "topology"),
        "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD", "topology"),
        "POSTGRES_DB": dbname,
        "TOPOLOGY_STORAGE_ROOT": str(storage_root),
        "CELERY_BROKER_URL": broker_url,
        "CELERY_RESULT_BACKEND": broker_url,
    }


def _fresh_tile_tasks_module(broker_url: str, monkeypatch: pytest.MonkeyPatch):
    for name in list(sys.modules):
        if name.startswith("topology_geo.tasks"):
            del sys.modules[name]
    monkeypatch.setenv("CELERY_BROKER_URL", broker_url)
    monkeypatch.setenv("CELERY_RESULT_BACKEND", broker_url)
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "false")

    from topology_geo.tasks.tile_tasks import generate_terrain_tiles_parallel

    return generate_terrain_tiles_parallel


def test_parallel_tile_generation_with_real_worker_and_cache_reuse(pg_test_db, tmp_path, monkeypatch):
    dbname = pg_test_db.info.dbname
    ensure_schema(pg_test_db)

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    relief_path = storage_root / "relief.tif"
    _seed_relief_geotiff(relief_path)
    relief_storage_key = "relief.tif"

    broker_url = f"redis://localhost:6379/{TEST_REDIS_DB}"
    redis.Redis(host="localhost", port=6379, db=TEST_REDIS_DB).flushdb()

    worker = subprocess.Popen(
        [
            sys.executable, "-m", "celery", "-A", "topology_geo.tasks.celery_app", "worker",
            "--loglevel=info", "--pool=prefork", "--concurrency=4",
        ],
        env=_enqueue_env(dbname, storage_root, broker_url),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        generate_terrain_tiles_parallel = _fresh_tile_tasks_module(broker_url, monkeypatch)

        tiles = [TileIndex(zone=99, tx=tx, ty=ty) for tx in (0, 1) for ty in (0, 1)]

        deadline = time.time() + 40
        results = None
        last_error = None
        while time.time() < deadline and results is None:
            try:
                results = generate_terrain_tiles_parallel(tiles, DATA_VERSION, GENERATOR_VERSION, relief_storage_key)
            except Exception as exc:  # noqa: BLE001 - воркер мог не подняться ещё
                last_error = exc
                time.sleep(0.5)

        if results is None:
            worker.terminate()
            output = worker.stdout.read() if worker.stdout else ""
            pytest.fail(f"воркер не ответил вовремя: {last_error}\n{output}")

        assert len(results) == 4
        assert all(key.startswith("tiles/msk59-99/") for key in results.values())

        # тайлы реально стыкуются: общая граница (0,0)-(1,0) побитово совпадает.
        # Читаем напрямую с диска (тот же storage_root, что и у воркера) - не
        # через get_storage() этого процесса, у него своё пустое хранилище в памяти.
        from topology_geo.storage import FileSystemObjectStorage

        storage = FileSystemObjectStorage(storage_root)
        vertices_00, _ = deserialize_tin(storage.download(results[TileIndex(zone=99, tx=0, ty=0)]))
        vertices_10, _ = deserialize_tin(storage.download(results[TileIndex(zone=99, tx=1, ty=0)]))
        edge_00 = {(round(x, 6), round(y, 6)): z for x, y, z in vertices_00 if abs(x - 250.0) < 1e-9}
        edge_10 = {(round(x, 6), round(y, 6)): z for x, y, z in vertices_10 if abs(x - 250.0) < 1e-9}
        assert edge_00 and set(edge_00) == set(edge_10)
        assert all(edge_00[k] == edge_10[k] for k in edge_00)

        with pg_test_db.cursor() as cur:
            cur.execute("SELECT tile_key, created_at FROM tile_cache ORDER BY tile_key")
            first_run_rows = cur.fetchall()
        assert len(first_run_rows) == 4

        # повторный запрос того же батча - должен взять всё из кеша, без
        # повторной генерации (created_at не меняется, п. 3 Шага 2.1)
        results_again = generate_terrain_tiles_parallel(tiles, DATA_VERSION, GENERATOR_VERSION, relief_storage_key)
        assert results_again == results

        with pg_test_db.cursor() as cur:
            cur.execute("SELECT tile_key, created_at FROM tile_cache ORDER BY tile_key")
            second_run_rows = cur.fetchall()
        assert second_run_rows == first_run_rows
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=5)


def test_find_cached_tile_helper_still_works_after_parallel_run(pg_test_db):
    ensure_schema(pg_test_db)
    tile = TileIndex(zone=99, tx=0, ty=0)
    assert find_cached_tile(pg_test_db, tile.key("terrain", DATA_VERSION, GENERATOR_VERSION)) is None
