"""Тесты Шага 2.12, п. 5: монтирование `/citymap`/`/citymap-data` в FastAPI и
реальная нагрузка — 50 одновременных клиентов с Range-запросами к
`city.pmtiles` (критерий плана «выдерживает 50 одновременных пользователей»).

Как и `test_api.py`, окружение (`POSTGRES_DB`/`CITYMAP_DATA_DIR`/...)
читается один раз при первом импорте `topology_geo.api.app`, поэтому кэш
`sys.modules` чистится и приложение импортируется заново под нужное
окружение в каждой фикстуре/тесте (см. docstring `test_api.py`).
"""

from __future__ import annotations

import shutil
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

FIXTURE_PMTILES = Path(__file__).resolve().parent / "fixtures" / "citymap" / "perm_demo.pmtiles"


def _make_client(dbname: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, citymap_data_dir: Path | None):
    for name in list(sys.modules):
        if name.startswith("topology_geo.tasks") or name.startswith("topology_geo.api"):
            del sys.modules[name]

    monkeypatch.setenv("POSTGRES_DB", dbname)
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    monkeypatch.setenv("TOPOLOGY_STORAGE_ROOT", str(tmp_path / "storage"))
    if citymap_data_dir is not None:
        monkeypatch.setenv("CITYMAP_DATA_DIR", str(citymap_data_dir))
    else:
        monkeypatch.delenv("CITYMAP_DATA_DIR", raising=False)

    from fastapi.testclient import TestClient

    from topology_geo.api.app import app

    return TestClient(app)


@pytest.fixture()
def citymap_data_dir(tmp_path):
    data_dir = tmp_path / "citymap-data"
    data_dir.mkdir()
    shutil.copy(FIXTURE_PMTILES, data_dir / "city.pmtiles")
    return data_dir


def test_citymap_index_served_at_mount(pg_test_db, tmp_path, monkeypatch):
    with _make_client(pg_test_db.info.dbname, tmp_path, monkeypatch, citymap_data_dir=None) as client:
        resp = client.get("/citymap/index.html")
        assert resp.status_code == 200
        assert "Топология" in resp.text
        assert "pmtiles" in resp.text.lower()


def test_citymap_data_not_mounted_without_env_var(pg_test_db, tmp_path, monkeypatch):
    """Честное поведение при отсутствии `CITYMAP_DATA_DIR` (реальная
    городская выгрузка в этом окружении не строится, см. docs/citymap.md) -
    404, не падение при старте приложения."""
    with _make_client(pg_test_db.info.dbname, tmp_path, monkeypatch, citymap_data_dir=None) as client:
        resp = client.get("/citymap-data/city.pmtiles")
        assert resp.status_code == 404


def test_citymap_data_serves_real_pmtiles_with_range_support(pg_test_db, tmp_path, monkeypatch, citymap_data_dir):
    with _make_client(pg_test_db.info.dbname, tmp_path, monkeypatch, citymap_data_dir=citymap_data_dir) as client:
        full = client.get("/citymap-data/city.pmtiles")
        assert full.status_code == 200
        assert full.content[:7] == b"PMTiles"
        assert len(full.content) == FIXTURE_PMTILES.stat().st_size

        partial = client.get("/citymap-data/city.pmtiles", headers={"Range": "bytes=0-99"})
        assert partial.status_code == 206
        assert len(partial.content) == 100
        assert partial.content == FIXTURE_PMTILES.read_bytes()[:100]


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_citymap_data_withstands_50_concurrent_range_clients(pg_test_db, tmp_path, monkeypatch, citymap_data_dir):
    """Реальная HTTP-нагрузка (настоящий uvicorn на TCP-сокете, не
    in-process ASGI-транспорт) - критерий плана «выдерживает 50 одновременных
    пользователей» (Шаг 2.12): 50 потоков реально бьют Range-запросами по
    `city.pmtiles` одновременно (тот же паттерн доступа, что `pmtiles.js` в
    браузере - диапазонные запросы, не скачивание файла целиком)."""
    import uvicorn

    with _make_client(pg_test_db.info.dbname, tmp_path, monkeypatch, citymap_data_dir=citymap_data_dir):
        pass  # прогреть импорт/окружение тем же путём, что и остальные тесты модуля

    from topology_geo.api.app import app

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("uvicorn не запустился за 5 с")

        base_url = f"http://127.0.0.1:{port}"
        expected_first_100 = FIXTURE_PMTILES.read_bytes()[:100]

        def _fetch_range(i: int) -> tuple[int, bytes]:
            resp = httpx.get(f"{base_url}/citymap-data/city.pmtiles", headers={"Range": "bytes=0-99"}, timeout=10.0)
            return resp.status_code, resp.content

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=50) as pool:
            results = list(pool.map(_fetch_range, range(50)))
        elapsed = time.monotonic() - started

        assert len(results) == 50
        for status_code, content in results:
            assert status_code == 206
            assert content == expected_first_100

        # Честно: измерено только в этой среде (без реального интернета
        # между клиентом и сервером), не заявление о production-пропускной
        # способности - тот же компромисс, что и таймауты вьюера (Шаг 1.9/2.11).
        assert elapsed < 10.0
    finally:
        server.should_exit = True
        thread.join(timeout=5)
