"""Тест Шага 4.7, п. 2 («вьюер панорам с переходами между точками») в
настоящем браузере (Chromium/Playwright) — реальная загрузка списка точек,
отрисовка сферы с текстурой панорамы, переключение между точками.

Панорамы для этого теста рендерятся по-настоящему через `bpy` (Шаг 4.7,
`rendering.panorama.render_panorama`), не заглушки — переключение реально
меняет отображаемую текстуру."""

from __future__ import annotations

import functools
import glob
import http.server
import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

bpy = pytest.importorskip("bpy")
playwright_sync_api = pytest.importorskip("playwright.sync_api")

from topology_geo.geometry.buildings import BuildingSolid
from topology_geo.ifc.assemble import BasePoint, SiteModel, build_site_ifc
from topology_geo.ifc.to_glb import convert_ifc_to_glb
from topology_geo.relief.tin import SiteTin
from topology_geo.rendering.panorama import render_panorama

WEB_DIR = Path(__file__).resolve().parents[1] / "src" / "topology_geo" / "web"
BASE_POINT = BasePoint(lon=56.25, lat=58.0, zone=2, x=2_310_450.0, y=-5_857_320.0, height=150.0)


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


def _make_flat_tin(half_extent: float = 40.0, n: int = 12) -> SiteTin:
    xs, ys = np.meshgrid(np.linspace(-half_extent, half_extent, n), np.linspace(-half_extent, half_extent, n))
    zs = np.zeros_like(xs)
    vertices = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    delaunay = Delaunay(vertices[:, :2])
    return SiteTin(vertices=vertices, triangles=delaunay.simplices, _delaunay=delaunay)


def _make_scene_glb(tmp_path: Path) -> Path:
    building = BuildingSolid(
        osm_id=1, footprint=Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)]),
        height_m=10.0, height_confidence="факт", height_source="OSM", base_z=0.0, building_type="жилой",
    )
    model_ifc, _ = build_site_ifc("IFC4", SiteModel(tin=_make_flat_tin(), buildings=[building]), BASE_POINT)
    glb_path = tmp_path / "scene.glb"
    glb_path.write_bytes(convert_ifc_to_glb(model_ifc))
    return glb_path


@pytest.fixture()
def panorama_server(tmp_path):
    web_root = tmp_path / "web"
    shutil.copytree(WEB_DIR / "panorama", web_root / "panorama")
    shutil.copytree(WEB_DIR / "viewer" / "vendor", web_root / "viewer" / "vendor")

    glb_path = _make_scene_glb(tmp_path)
    points = [
        {"label": "Точка А — вход", "position": (0.0, -20.0, 1.6)},
        {"label": "Точка Б — двор", "position": (15.0, 0.0, 1.6)},
    ]
    manifest = []
    for i, point in enumerate(points):
        filename = f"pano_{i}.png"
        render_panorama(
            glb_path, point["position"], web_root / "panorama" / filename,
            quality="draft", sun_direction=(0.5, 0.0, 0.85), resolution=(128, 64),
        )
        manifest.append({"label": point["label"], "url": filename})
    (web_root / "panorama" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(web_root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def test_panorama_viewer_loads_and_switches_between_points(panorama_server):
    from playwright.sync_api import sync_playwright

    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = _launch_chromium(p)
        try:
            page = browser.new_page(viewport={"width": 1024, "height": 768})
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))

            page.goto(f"{panorama_server}/panorama/index.html?manifest=manifest.json")
            page.wait_for_function("document.querySelectorAll('#points-list li').length === 2", timeout=10_000)

            page.wait_for_function("window.currentPanoramaUrl === 'pano_0.png'", timeout=10_000)
            assert page.locator("#points-list li").first.evaluate("el => el.classList.contains('active')")

            second_point = page.locator("#points-list li").nth(1)
            second_point.click()
            page.wait_for_function("window.currentPanoramaUrl === 'pano_1.png'", timeout=10_000)
            assert second_point.evaluate("el => el.classList.contains('active')")
            assert not page.locator("#points-list li").first.evaluate("el => el.classList.contains('active')")

            assert page.evaluate("document.querySelectorAll('canvas').length") == 1
            assert page_errors == []
        finally:
            browser.close()
