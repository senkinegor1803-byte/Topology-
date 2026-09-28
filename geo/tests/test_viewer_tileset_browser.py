"""Тест потокового режима вьюера (Шаг 2.11, п. 2-3) в настоящем браузере —
`?tileset=` загружает `tileset.json`, догружает тайлы по мере приближения
камеры к центру участка и строит то же дерево слоёв/панель свойств, что и
`?model=` (Шаг 1.9), как того требует действие плана «сохранение панели
свойств и слоёв из этапа 1»."""

from __future__ import annotations

import functools
import glob
import http.server
import json
import shutil
import threading
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from topology_geo.geometry.buildings import CONFIDENCE_FACT, SOURCE_OSM, BuildingSolid
from topology_geo.relief.service import Grid
from topology_geo.tiling.grid import LOD1, LOD2, TileIndex
from topology_geo.tiling.tile_content import build_tile_content
from topology_geo.tiling.tileset import TileContentEntry, build_tileset_json

VIEWER_DIR = Path(__file__).resolve().parents[1] / "src" / "topology_geo" / "web" / "viewer"

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


CENTER_X, CENTER_Y = 5_000.0, 8_000.0
PIXEL = 2.0


def _flat_relief_grid(half_extent: float) -> tuple[np.ndarray, Grid]:
    size = int(2 * half_extent / PIXEL)
    transform = Affine(PIXEL, 0.0, CENTER_X - half_extent, 0.0, -PIXEL, CENTER_Y + half_extent)
    grid = Grid(transform=transform, width=size, height=size, crs="EPSG:3857")
    values = np.full((size, size), 100.0, dtype="float64")
    return values, grid


def _building(local_x: float, local_y: float, osm_id: int) -> BuildingSolid:
    return BuildingSolid(
        osm_id=osm_id, footprint=box(local_x - 5, local_y - 5, local_x + 5, local_y + 5), height_m=9.0,
        height_confidence=CONFIDENCE_FACT, height_source=SOURCE_OSM, base_z=100.0, building_type="жилой",
    )


def _build_test_tileset(web_root: Path) -> None:
    """Реальные `tileset.json` + 2 тайла (LOD2 рядом с центром + LOD1
    дальше) - той же функцией (`tile_content.build_tile_content`), что и
    боевой шаг `jobs.steps.generate_tileset`."""
    values, grid = _flat_relief_grid(half_extent=600.0)
    tiles_dir = web_root / "tiles"
    tiles_dir.mkdir()

    entries = []
    specs = [
        (TileIndex(zone=2, tx=int(CENTER_X // 250), ty=int(CENTER_Y // 250)), LOD2, [_building(0.0, 0.0, 1)]),
        (TileIndex(zone=2, tx=int(CENTER_X // 250) + 2, ty=int(CENTER_Y // 250)), LOD1, []),
    ]
    for tile, lod, buildings in specs:
        content = build_tile_content(values, grid, tile, lod, buildings, center_x=CENTER_X, center_y=CENTER_Y)
        assert content is not None
        key = f"tiles/{lod}_{tile.tx}_{tile.ty}.glb"
        (web_root / key).write_bytes(content.glb_bytes)
        minx, miny, maxx, maxy = tile.bounds()
        entries.append(
            TileContentEntry(
                tile=tile, lod=lod, storage_key=key,
                local_minx=minx - CENTER_X, local_miny=miny - CENTER_Y,
                local_maxx=maxx - CENTER_X, local_maxy=maxy - CENTER_Y,
                z_min=content.z_min, z_max=content.z_max,
            )
        )

    tileset_doc = build_tileset_json(entries)
    (web_root / "tileset.json").write_text(json.dumps(tileset_doc, ensure_ascii=False), encoding="utf-8")


class _KeyQueryHandler(http.server.SimpleHTTPRequestHandler):
    """Тот же приём разрешения ключей, что и `/models/{job}/download?key=...`
    в реальном API (Шаг 1.3/2.11) — здесь без job/Storage, просто отдаёт
    файл по значению `key` из каталога вьюера."""

    def do_GET(self):  # noqa: N802 - имя метода задано http.server
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if "key" in qs:
            self.path = "/" + qs["key"][0]
        else:
            self.path = parsed.path
        return super().do_GET()


@pytest.fixture()
def viewer_tileset_server(tmp_path):
    web_root = tmp_path / "viewer"
    shutil.copytree(VIEWER_DIR, web_root)
    _build_test_tileset(web_root)

    handler = functools.partial(_KeyQueryHandler, directory=str(web_root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def test_tileset_viewer_loads_tiles_shows_tree_in_real_browser(viewer_tileset_server):
    from playwright.sync_api import sync_playwright

    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = _launch_chromium(p)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))

            page.goto(f"{viewer_tileset_server}/index.html?tileset={quote('serve?key=tileset.json', safe='')}&job=test")
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('Загружено')", timeout=10_000
            )

            status_text = page.locator("#status").inner_text()
            assert "из 2 тайлов" in status_text

            # Рельеф + здание в ближнем тайле (LOD2) уже должны быть в дереве
            assert page.locator("#tree-root .layer").count() >= 1
            layer_names = page.locator("#tree-root .layer label").all_inner_texts()
            assert any("Рельеф" in name for name in layer_names)
            assert any("Здания" in name for name in layer_names)

            leaf_items = page.locator("#tree-root li")
            assert leaf_items.count() >= 1
            leaf_items.first.click()
            page.wait_for_timeout(200)  # клик обрабатывается синхронно, только дать кадру перерисоваться

            assert page_errors == []
        finally:
            browser.close()


def test_tileset_viewer_loads_within_five_seconds(viewer_tileset_server):
    """Критерий Шага 2.11: «первый кадр ≤ 5 с» — здесь измеряется не кадр
    рендера (недоступен без реального GPU/устройства), а появление первой
    партии тайлов в статус-строке, тот же честный компромисс, что и в
    `test_viewer_loads_within_ten_seconds` (Шаг 1.9)."""
    import time

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = _launch_chromium(p)
        try:
            page = browser.new_page()
            started = time.monotonic()
            page.goto(f"{viewer_tileset_server}/index.html?tileset={quote('serve?key=tileset.json', safe='')}&job=test")
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('Загружено')", timeout=10_000
            )
            elapsed = time.monotonic() - started
            assert elapsed <= 5.0
        finally:
            browser.close()
