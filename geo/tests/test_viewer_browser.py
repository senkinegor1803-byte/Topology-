"""Тест вьюера (Шаг 1.9, п. 2) в настоящем браузере (Chromium/Playwright) —
не разбор HTML, а реальная загрузка GLB, построение сцены, дерево объектов
и панель свойств по клику, как того требует критерий шага («модель
открывается в браузере ... за ≤ 10 с»)."""

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
from shapely.geometry import LineString, Polygon

from topology_geo.geometry.buildings import BuildingSolid
from topology_geo.geometry.rail import RailRibbon
from topology_geo.geometry.roads import RoadRibbon
from topology_geo.geometry.vegetation import TreePoint
from topology_geo.geometry.water import WaterArea, WaterwayRibbon
from topology_geo.ifc.assemble import BasePoint, SiteModel, build_site_ifc
from topology_geo.ifc.to_glb import convert_ifc_to_glb
from topology_geo.relief.tin import SiteTin

VIEWER_DIR = Path(__file__).resolve().parents[1] / "src" / "topology_geo" / "web" / "viewer"

playwright_sync_api = pytest.importorskip("playwright.sync_api")


def _launch_chromium(p):
    """Запустить Chromium, преднастроенный `playwright install chromium` (в
    CI/обычной разработке — стандартный кэш, находится сам) или лежащий в
    нестандартном месте (эта песочница держит его в /opt/pw-browsers, см.
    `PLAYWRIGHT_BROWSERS_PATH`, из-за версии Playwright новее, чем сам
    браузер, — стандартный поиск его не находит). Пропускаем тест, если
    браузера нет вовсе, а не падаем."""
    from playwright.sync_api import Error as PlaywrightError

    try:
        return p.chromium.launch()
    except PlaywrightError:
        pass
    sandbox_candidates = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))
    if not sandbox_candidates:
        pytest.skip("Chromium не найден (ни стандартный кэш Playwright, ни /opt/pw-browsers)")
    return p.chromium.launch(executable_path=sandbox_candidates[-1])


def _make_flat_tin(half_extent: float = 50.0, n: int = 6) -> SiteTin:
    xs, ys = np.meshgrid(np.linspace(-half_extent, half_extent, n), np.linspace(-half_extent, half_extent, n))
    zs = 100.0 + 0.01 * xs + 0.02 * ys
    vertices = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    delaunay = Delaunay(vertices[:, :2])
    return SiteTin(vertices=vertices, triangles=delaunay.simplices, _delaunay=delaunay)


def _make_site_glb() -> bytes:
    building = BuildingSolid(
        osm_id=1, footprint=Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
        height_m=12.0, height_confidence="факт", height_source="OSM", base_z=100.0, building_type="жилой",
    )
    road = RoadRibbon(
        osm_id=2, ribbon=LineString([(-40, 0), (40, 0)]).buffer(3.0, cap_style="flat"),
        width_m=6.0, width_confidence="умолчание", surface="asphalt", highway_class="residential",
    )
    water = WaterArea(osm_id=3, polygon=Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]), level_z=99.5)
    waterway = WaterwayRibbon(osm_id=4, ribbon=LineString([(0, -40), (0, 40)]).buffer(1.5, cap_style="flat"), width_m=3.0)
    rail = RailRibbon(osm_id=5, ballast=LineString([(-40, -20), (40, -20)]).buffer(2.0, cap_style="flat"), rail_type="tram")
    tree = TreePoint(x=15.0, y=-15.0, species="Betula pendula", confidence="факт", source_osm_id=6)
    site_model = SiteModel(
        tin=_make_flat_tin(), buildings=[building], roads=[road],
        water_areas=[water], waterways=[waterway], rail=[rail], trees=[tree],
    )
    base_point = BasePoint(lon=56.25, lat=58.0, zone=2, x=2_310_450.0, y=-5_857_320.0, height=150.0)
    model, _ = build_site_ifc("IFC4X3", site_model, base_point)
    return convert_ifc_to_glb(model)


@pytest.fixture()
def viewer_server(tmp_path):
    """Копия каталога вьюера + сгенерированный site.glb, отданные по
    настоящему HTTP (не file://, чтобы ES-модули/fetch работали как в проде)."""
    web_root = tmp_path / "viewer"
    shutil.copytree(VIEWER_DIR, web_root)
    (web_root / "site.glb").write_bytes(_make_site_glb())

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(web_root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def test_viewer_loads_model_shows_tree_and_properties_in_real_browser(viewer_server):
    from playwright.sync_api import sync_playwright

    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = _launch_chromium(p)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))

            page.goto(f"{viewer_server}/index.html?model=site.glb&job=test")
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('Загружено')", timeout=10_000
            )

            assert page.locator("#tree-root .layer").count() == 6
            leaf_items = page.locator("#tree-root li")
            assert leaf_items.count() == 7  # 6 категорий, у "Вода" два объекта

            leaf_items.first.click()
            page.wait_for_function("document.querySelector('#props-body').textContent.includes('GlobalId')")
            props_text = page.locator("#props-body").inner_text()
            assert "IfcGeographicElement" in props_text or "IfcBuildingElementProxy" in props_text

            layer_checkbox = page.locator("#tree-root .layer label input").first
            layer_checkbox.uncheck()
            assert page.evaluate("document.querySelectorAll('canvas').length") == 1

            assert page_errors == []
        finally:
            browser.close()


def test_viewer_loads_within_ten_seconds(viewer_server):
    """Критерий Шага 1.9: «модель открывается за ≤ 10 с»."""
    import time

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = _launch_chromium(p)
        try:
            page = browser.new_page()
            started = time.monotonic()
            page.goto(f"{viewer_server}/index.html?model=site.glb&job=test")
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('Загружено')", timeout=10_000
            )
            elapsed = time.monotonic() - started
            assert elapsed <= 10.0
        finally:
            browser.close()
