"""Тест Шага 4.2, п. «переключатель во вьюере технический/презентационный
вид» — в настоящем браузере (Chromium/Playwright), на настоящей запечённой
модели (реальный `bpy`-бейк AO×тень, Шаг 4.2). Проверяет, что клик по
кнопке реально переключает `material.vertexColors` у мешей с запечёнными
вершинными цветами (не просто меняет текст кнопки)."""

from __future__ import annotations

import functools
import glob
import http.server
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
from topology_geo.rendering.lightmap_bake import bake_lighting_to_vertex_colors

VIEWER_DIR = Path(__file__).resolve().parents[1] / "src" / "topology_geo" / "web" / "viewer"
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


def _make_baked_glb(tmp_path: Path) -> bytes:
    building = BuildingSolid(
        osm_id=1, footprint=Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)]),
        height_m=10.0, height_confidence="факт", height_source="OSM", base_z=0.0, building_type="жилой",
    )
    model_ifc, _ = build_site_ifc("IFC4", SiteModel(tin=_make_flat_tin(), buildings=[building]), BASE_POINT)
    raw_glb = tmp_path / "raw.glb"
    raw_glb.write_bytes(convert_ifc_to_glb(model_ifc))
    baked_glb = tmp_path / "baked.glb"
    bake_lighting_to_vertex_colors(raw_glb, baked_glb, sun_direction=(0.5, 0.0, 0.85), ao_samples=16, shadow_samples=16)
    return baked_glb.read_bytes()


@pytest.fixture()
def viewer_server_with_baked_model(tmp_path):
    web_root = tmp_path / "viewer"
    shutil.copytree(VIEWER_DIR, web_root)
    (web_root / "site.glb").write_bytes(_make_baked_glb(tmp_path))

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(web_root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def test_view_mode_toggle_switches_vertex_colors_on_baked_model(viewer_server_with_baked_model):
    from playwright.sync_api import sync_playwright

    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = _launch_chromium(p)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))

            page.goto(f"{viewer_server_with_baked_model}/index.html?model=site.glb&job=test")
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('Загружено')", timeout=10_000
            )

            toggle = page.locator("#view-mode-toggle")
            page.wait_for_function(
                "document.querySelector('#view-mode-toggle').style.display === 'block'", timeout=5_000
            )
            assert toggle.text_content() == "Презентационный вид"

            initial_state = page.evaluate("Array.from(bakedLitMaterials).map((m) => m.vertexColors)")
            assert initial_state, "не нашли ни одного материала с запечёнными вершинными цветами"
            assert all(initial_state), "презентационный вид должен включать vertexColors у всех запечённых материалов"

            toggle.click()
            assert toggle.text_content() == "Технический вид"
            technical_state = page.evaluate("Array.from(bakedLitMaterials).map((m) => m.vertexColors)")
            assert not any(technical_state), "технический вид должен отключать vertexColors"

            toggle.click()
            assert toggle.text_content() == "Презентационный вид"
            restored_state = page.evaluate("Array.from(bakedLitMaterials).map((m) => m.vertexColors)")
            assert all(restored_state)

            assert page_errors == []
        finally:
            browser.close()
