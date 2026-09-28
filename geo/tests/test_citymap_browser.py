"""Тест публичной карты города (Шаг 2.12) в настоящем браузере: реальный
`city.pmtiles` (собран Planetiler'ом из реальных OSM-данных «первой точки»,
см. `tests/fixtures/citymap/perm_demo.pmtiles`) + реальная пирамида
terrain-RGB (`citymap.cli.build_terrain_tile_pyramid`, та же функция, что и
боевой CLI) рендерятся в MapLibre GL, клик по карте ставит задачу через
`POST /jobs`.

API `/jobs`/`/models/*/files` здесь замокан отдельным HTTP-сервером (через
`?jobsApi=`, см. `web/citymap/index.html`) - настоящий сквозной прогон с
реальным Postgres+пайплайном уже покрыт `test_api.py`
(`test_full_happy_path_creates_downloadable_files`), здесь цель - проверить
именно фронтенд (рендер карты, работу PMTiles по Range-запросам в браузере,
клик -> POST -> опрос статуса -> ссылка на результат), а не пересчитывать
пайплайн в каждом браузерном тесте."""

from __future__ import annotations

import functools
import glob
import http.server
import json
import re
import shutil
import threading
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from topology_geo.citymap.cli import build_terrain_tile_pyramid

CITYMAP_DIR = Path(__file__).resolve().parents[1] / "src" / "topology_geo" / "web" / "citymap"
FIXTURE_PMTILES = Path(__file__).resolve().parent / "fixtures" / "citymap" / "perm_demo.pmtiles"

# Bbox «первой точки» (см. summary сессии): реальный запрос к
# api.openstreetmap.org/api/0.6/map, из которого собран perm_demo.pmtiles.
DEMO_MIN_LON, DEMO_MIN_LAT, DEMO_MAX_LON, DEMO_MAX_LAT = 56.330348, 58.043698, 56.354114, 58.056274

playwright_sync_api = pytest.importorskip("playwright.sync_api")


def _launch_chromium(p):
    from playwright.sync_api import Error as PlaywrightError

    try:
        return p.chromium.launch()
    except PlaywrightError:
        pass
    sandbox_candidates = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))
    if not sandbox_candidates:
        pytest.skip("Chromium не найден (ни стандартный кэш Playwright, ни /opt/pw-browsers)")
    return p.chromium.launch(executable_path=sandbox_candidates[-1])


def _write_flat_dem(path: Path, elevation: float) -> None:
    half_extent = 0.02
    size = 60
    step = 2 * half_extent / size
    center_lon = (DEMO_MIN_LON + DEMO_MAX_LON) / 2
    center_lat = (DEMO_MIN_LAT + DEMO_MAX_LAT) / 2
    transform = from_origin(center_lon - half_extent, center_lat + half_extent, step, step)
    data = np.full((size, size), elevation, dtype="float64")
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1, dtype="float64",
        crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


FIXED_JOB_ID = "11111111-1111-1111-1111-111111111111"


class _CitymapMockHandler(http.server.BaseHTTPRequestHandler):
    """Раздаёт собранную страницу/vendor/citymap-data (с поддержкой
    Range-запросов - PMTiles их требует для случайного доступа к архиву,
    которую `http.server` не даёт из коробки) + имитирует `/jobs` (Шаг
    2.12, п. 4), не поднимая настоящий Postgres/Celery для браузерного
    теста фронтенда."""

    def __init__(self, *args, web_root: Path, poll_counts: dict, **kwargs):
        self.web_root = web_root
        self.poll_counts = poll_counts
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):  # noqa: A002 - сигнатура stdlib
        pass

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/jobs":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            assert "center" in body and "radius_m" in body  # реальная форма запроса API (schemas.JobCreateRequest)
            payload = json.dumps({"id": FIXED_JOB_ID, "status": "pending"}).encode("utf-8")
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == f"/jobs/{FIXED_JOB_ID}":
            self.poll_counts["n"] += 1
            status = "running" if self.poll_counts["n"] < 2 else "done"
            payload = json.dumps({
                "id": FIXED_JOB_ID, "status": status, "center": {"lon": 56.34, "lat": 58.05},
                "radius_m": 500.0, "layers": [], "detail": "LOD1", "error_message": None,
                "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z", "steps": [],
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == f"/models/{FIXED_JOB_ID}/files":
            payload = json.dumps({
                "job_id": FIXED_JOB_ID, "status": "done",
                "files": [{
                    "step_name": "convert_to_glb", "storage_key": f"jobs/{FIXED_JOB_ID}/model.glb",
                    "download_url": f"/models/{FIXED_JOB_ID}/download?key=model.glb",
                    "viewer_url": f"/viewer/index.html?model=%2Fmodels%2F{FIXED_JOB_ID}%2Fdownload&job={FIXED_JOB_ID}",
                }],
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._serve_static(parsed.path)

    def _serve_static(self, url_path: str) -> None:
        rel = url_path.lstrip("/")
        file_path = (self.web_root / rel).resolve()
        if self.web_root not in file_path.parents and file_path != self.web_root:
            self.send_error(404)
            return
        if not file_path.is_file():
            self.send_error(404)
            return

        size = file_path.stat().st_size
        range_header = self.headers.get("Range")
        content_type = "application/octet-stream"
        if file_path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif file_path.suffix == ".js":
            content_type = "text/javascript"
        elif file_path.suffix == ".css":
            content_type = "text/css"
        elif file_path.suffix == ".png":
            content_type = "image/png"

        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header)
            start_s, end_s = m.groups()
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
            end = min(end, size - 1)
            with open(file_path, "rb") as f:
                f.seek(start)
                chunk = f.read(end - start + 1)
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return

        with open(file_path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def citymap_server(tmp_path):
    web_root = tmp_path / "web"
    shutil.copytree(CITYMAP_DIR, web_root / "citymap")
    data_dir = web_root / "citymap-data"
    data_dir.mkdir(parents=True)
    shutil.copy(FIXTURE_PMTILES, data_dir / "city.pmtiles")

    dem_path = tmp_path / "dem.tif"
    _write_flat_dem(dem_path, elevation=120.0)
    build_terrain_tile_pyramid(
        dem_path, data_dir / "terrain",
        min_lon=DEMO_MIN_LON - 0.01, min_lat=DEMO_MIN_LAT - 0.01,
        max_lon=DEMO_MAX_LON + 0.01, max_lat=DEMO_MAX_LAT + 0.01,
        min_zoom=10, max_zoom=15,
    )

    poll_counts = {"n": 0}
    handler = functools.partial(_CitymapMockHandler, web_root=web_root, poll_counts=poll_counts)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def _page_url(server: str) -> str:
    lon = (DEMO_MIN_LON + DEMO_MAX_LON) / 2
    lat = (DEMO_MIN_LAT + DEMO_MAX_LAT) / 2
    return f"{server}/citymap/index.html?lon={lon}&lat={lat}&zoom=15"


def test_citymap_renders_real_pmtiles_and_terrain_no_errors(citymap_server):
    from playwright.sync_api import sync_playwright

    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = _launch_chromium(p)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))

            page.goto(_page_url(citymap_server))
            page.wait_for_function(
                "document.querySelector('#map-status').textContent.includes('загружена')", timeout=15_000
            )

            assert page.locator("#map canvas").count() >= 1
            attribution_text = page.locator("#attribution-text").inner_text()
            assert "OpenMapTiles" in attribution_text
            assert "OpenStreetMap" in attribution_text

            assert page_errors == []
        finally:
            browser.close()


def test_citymap_loads_within_three_seconds(citymap_server):
    """Критерий Шага 2.12: «карта открывается ≤ 3 с» - как и в Шагах
    1.9/2.11, измеряется появление первого готового состояния карты
    (событие MapLibre `load`, тайлы первого экрана запрошены), не кадр
    рендера на реальном устройстве/сети - тот же честный компромисс
    headless-браузера без GPU/реальной сети до клиента."""
    import time

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = _launch_chromium(p)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            started = time.monotonic()
            page.goto(_page_url(citymap_server))
            page.wait_for_function(
                "document.querySelector('#map-status').textContent.includes('загружена')", timeout=10_000
            )
            elapsed = time.monotonic() - started
            assert elapsed <= 3.0
        finally:
            browser.close()


def test_citymap_build_button_posts_job_and_shows_result(citymap_server):
    from playwright.sync_api import sync_playwright

    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = _launch_chromium(p)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))

            page.goto(_page_url(citymap_server))
            page.wait_for_function(
                "document.querySelector('#map-status').textContent.includes('загружена')", timeout=15_000
            )

            build_btn = page.locator("#build-btn")
            assert build_btn.is_disabled()

            map_box = page.locator("#map").bounding_box()
            cx = map_box["x"] + map_box["width"] / 2
            cy = map_box["y"] + map_box["height"] / 2
            page.mouse.click(cx, cy, button="right")

            page.wait_for_function(
                "document.querySelector('#picked-point').textContent !== 'не выбрана'", timeout=5_000
            )
            assert not build_btn.is_disabled()

            build_btn.click()
            page.wait_for_function(
                f"document.querySelector('#job-status').textContent.includes('{FIXED_JOB_ID}')", timeout=5_000
            )
            page.wait_for_function(
                "document.querySelector('#job-status').textContent.includes('готово')", timeout=10_000
            )

            assert page.locator("#job-status a").count() == 1
            assert page_errors == []
        finally:
            browser.close()


def test_citymap_2d_3d_toggle_switches_layers(citymap_server):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = _launch_chromium(p)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(_page_url(citymap_server))
            page.wait_for_function(
                "document.querySelector('#map-status').textContent.includes('загружена')", timeout=15_000
            )

            assert "active" in (page.locator("#btn-3d").get_attribute("class") or "")
            page.locator("#btn-2d").click()
            page.wait_for_timeout(200)
            assert "active" in (page.locator("#btn-2d").get_attribute("class") or "")
            assert "active" not in (page.locator("#btn-3d").get_attribute("class") or "")
        finally:
            browser.close()
