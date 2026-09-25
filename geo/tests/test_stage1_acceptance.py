"""Шаг 1.10: сквозной тест пайплайна на 3 профилях участков + приёмка этапа 1.

Автоматизируемая часть плана (п. 1-2 Шага 1.10): реальный `osm2pgsql`-импорт
трёх разных профилей (центр, спальный район, частный сектор) -> реальный
конвейер до `site.ifc`/`site.glb` -> замер времени каждого шага. Плюс
геометрическая проверка «плановое расхождение с OSM ≤ 0,5 м» (раздел 14 ТЗ) —
не предположение, а сравнение экспортированной геометрии IFC с независимо
выбранными локальными координатами того же объекта (Шаг 1.4).

Проверка BIM-специалистом в Renga/Pilot-BIM (п. 3-4 Шага 1.10) и критерий
«модель ≤ 5 мин» на реальном пилоте (Шаг 0.1, реальные объёмы) — вне
возможностей AI-сессии, см. `docs/stage1-acceptance.md`.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from topology_geo.relief.coverage import CoverageEntry, register_coverage
from topology_geo.relief.coverage import ensure_schema as ensure_relief_schema

STYLE_LUA = Path(__file__).resolve().parents[1] / "osm2pgsql" / "style.lua"

CENTER_LON, CENTER_LAT = 56.243, 58.0105
_M_PER_DEG_LON = 111_320 * math.cos(math.radians(CENTER_LAT))
_M_PER_DEG_LAT = 111_320


def _osm2pgsql_available() -> bool:
    return shutil.which("osm2pgsql") is not None


def _lonlat(dx_m: float, dy_m: float) -> tuple[float, float]:
    return CENTER_LON + dx_m / _M_PER_DEG_LON, CENTER_LAT + dy_m / _M_PER_DEG_LAT


def _closed_ring(points_m: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return points_m if points_m[0] == points_m[-1] else [*points_m, points_m[0]]


class _NodeAllocator:
    def __init__(self) -> None:
        self._next_id = 1
        self.nodes_xml: list[str] = []

    def add_ring(self, points_m: list[tuple[float, float]]) -> list[int]:
        ids = []
        for dx, dy in _closed_ring(points_m)[:-1]:
            lon, lat = _lonlat(dx, dy)
            node_id = self._next_id
            self._next_id += 1
            self.nodes_xml.append(f'  <node id="{node_id}" lat="{lat:.7f}" lon="{lon:.7f}" version="1"/>')
            ids.append(node_id)
        ids.append(ids[0])  # замкнуть кольцо той же точкой (полигон)
        return ids

    def add_line(self, points_m: list[tuple[float, float]]) -> list[int]:
        ids = []
        for dx, dy in points_m:
            lon, lat = _lonlat(dx, dy)
            node_id = self._next_id
            self._next_id += 1
            self.nodes_xml.append(f'  <node id="{node_id}" lat="{lat:.7f}" lon="{lon:.7f}" version="1"/>')
            ids.append(node_id)
        return ids


def _way_xml(way_id: int, node_ids: list[int], tags: dict[str, str]) -> str:
    nds = "".join(f'<nd ref="{n}"/>' for n in node_ids)
    tag_xml = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items())
    return f'  <way id="{way_id}" version="1">{nds}{tag_xml}</way>'


def _build_osm_xml(polygons: list[tuple[list[tuple[float, float]], dict]], lines: list[tuple[list[tuple[float, float]], dict]]) -> str:
    alloc = _NodeAllocator()
    ways_xml = []
    way_id = 1000
    for points_m, tags in polygons:
        node_ids = alloc.add_ring(points_m)
        ways_xml.append(_way_xml(way_id, node_ids, tags))
        way_id += 1
    for points_m, tags in lines:
        node_ids = alloc.add_line(points_m)
        ways_xml.append(_way_xml(way_id, node_ids, tags))
        way_id += 1
    body = "\n".join(alloc.nodes_xml) + "\n" + "\n".join(ways_xml)
    return f'<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n<osm version="0.6" generator="topology-test">\n{body}\n</osm>\n'


def _square(cx: float, cy: float, half: float) -> list[tuple[float, float]]:
    return [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]


# --- три профиля участка (Шаг 1.10, п. 1) ------------------------------------


def profile_center() -> str:
    """Плотный центр: несколько многоэтажных зданий вплотную к магистрали."""
    buildings = [
        (_square(-40, 20, 12), {"building": "commercial", "building:levels": "6"}),
        (_square(-10, 25, 10), {"building": "apartments", "height": "22.5"}),
        (_square(20, 15, 14), {"building": "apartments", "building:levels": "8"}),
    ]
    lines = [
        ([(-150, 0), (150, 0)], {"highway": "primary", "lanes": "4", "surface": "asphalt"}),
        ([(0, -100), (0, 100)], {"highway": "secondary", "surface": "asphalt"}),
    ]
    return _build_osm_xml(buildings, lines)


def profile_residential() -> str:
    """Спальный район: крупные дома, дворовые проезды, зелень, пруд."""
    buildings = [
        (_square(-60, 40, 20), {"building": "apartments", "building:levels": "9"}),
        (_square(40, 40, 20), {"building": "apartments", "building:levels": "9"}),
    ]
    lines = [
        ([(-120, 0), (120, 0)], {"highway": "residential", "surface": "asphalt"}),
        ([(-60, 0), (-60, 60)], {"highway": "service"}),
    ]
    polygons_extra = [
        (_square(0, -60, 30), {"landuse": "grass"}),
        (_square(90, -40, 15), {"natural": "water"}),
    ]
    return _build_osm_xml(buildings + polygons_extra, lines)


def profile_private_sector() -> str:
    """Частный сектор: небольшие дома на отдельных участках, грунтовая дорога."""
    buildings = [
        (_square(-80, 30, 6), {"building": "house"}),
        (_square(-20, 35, 6), {"building": "house"}),
        (_square(40, 25, 6), {"building": "house"}),
    ]
    lines = [
        ([(-150, 0), (150, 0)], {"highway": "unclassified", "surface": "unpaved"}),
    ]
    return _build_osm_xml(buildings, lines)


PROFILES = {
    "центр": profile_center,
    "спальный_район": profile_residential,
    "частный_сектор": profile_private_sector,
}


def _seed_dem(pg_test_db, tmp_path: Path) -> None:
    ensure_relief_schema(pg_test_db)
    register_coverage(
        pg_test_db,
        CoverageEntry(
            source_name="TessaDEM (тест)", priority=1, storage_key="tessadem.tif",
            resolution_m=30.0, data_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            footprint_wkt="POLYGON((56.0 57.9, 56.5 57.9, 56.5 58.1, 56.0 58.1, 56.0 57.9))",
        ),
    )
    transform = from_origin(56.0, 58.1, 0.001, 0.001)
    data = np.full((200, 200), 155.0, dtype="float32")
    buf_path = tmp_path / "seed_dem.tif"
    with rasterio.open(
        buf_path, "w", driver="GTiff", height=200, width=200, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    from topology_geo.tasks.pipeline_tasks import get_storage

    get_storage().upload("tessadem.tif", buf_path.read_bytes())


def _import_osm(dbname: str, osm_xml: str, tmp_path: Path, name: str) -> None:
    osm_file = tmp_path / f"{name}.osm"
    osm_file.write_text(osm_xml, encoding="utf-8")
    env = {**os.environ, "PGPASSWORD": os.environ.get("POSTGRES_PASSWORD", "topology")}
    subprocess.run(
        [
            "osm2pgsql", "--output=flex", f"--style={STYLE_LUA}",
            f"--host={os.environ.get('POSTGRES_HOST', 'localhost')}",
            f"--port={os.environ.get('POSTGRES_PORT', '5432')}",
            f"--user={os.environ.get('POSTGRES_USER', 'topology')}",
            f"--database={dbname}", str(osm_file),
        ],
        check=True, capture_output=True, text=True, env=env,
    )


def _make_client(dbname: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for mod_name in list(sys.modules):
        if mod_name.startswith(("topology_geo.tasks", "topology_geo.api")):
            del sys.modules[mod_name]
    monkeypatch.setenv("POSTGRES_DB", dbname)
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    monkeypatch.setenv("TOPOLOGY_STORAGE_ROOT", str(tmp_path / "storage"))

    from fastapi.testclient import TestClient

    from topology_geo.api.app import app

    return TestClient(app)


@pytest.mark.skipif(not _osm2pgsql_available(), reason="требуется системный osm2pgsql")
@pytest.mark.parametrize("profile_name", list(PROFILES))
def test_pipeline_completes_for_each_site_profile(profile_name, pg_test_db, tmp_path, monkeypatch):
    """Шаг 1.10, п. 1-2: конвейер целиком (реальные OSM+рельеф+PostGIS) для
    каждого из 3 профилей + время каждого шага в журнале задачи."""
    dbname = pg_test_db.info.dbname
    _import_osm(dbname, PROFILES[profile_name](), tmp_path, profile_name)

    with _make_client(dbname, tmp_path, monkeypatch) as client:
        _seed_dem(pg_test_db, tmp_path)

        started = datetime.now(UTC)
        resp = client.post("/jobs", json={"center": {"lon": CENTER_LON, "lat": CENTER_LAT}, "radius_m": 500})
        assert resp.status_code == 201
        job_id = resp.json()["id"]

        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "done", job
        finished = datetime.now(UTC)

        total_elapsed = (finished - started).total_seconds()
        print(f"\n[{profile_name}] общее время конвейера: {total_elapsed:.2f} с")
        for step in job["steps"]:
            if step["started_at"] and step["finished_at"]:
                s = datetime.fromisoformat(step["started_at"])
                f = datetime.fromisoformat(step["finished_at"])
                print(f"  {step['step_name']}: {(f - s).total_seconds():.3f} с")

        # Критерий раздела 14 ТЗ — «модель ≤ 5 мин»; на синтетических данных
        # этого масштаба ожидается на порядки быстрее, щедрый запас — не для
        # проверки реального времени на пилоте (Шаг 0.1), а как регрессия
        # против зависания/деградации пайплайна.
        assert total_elapsed < 120.0

        files = client.get(f"/models/{job_id}/files").json()["files"]
        step_names = {f["step_name"] for f in files}
        assert "convert_to_glb" in step_names
        assert any(name.startswith("assemble_ifc:") for name in step_names)


@pytest.mark.skipif(not _osm2pgsql_available(), reason="требуется системный osm2pgsql")
def test_building_footprint_survives_ifc_export_within_half_a_meter(pg_test_db, tmp_path, monkeypatch):
    """Раздел 14 ТЗ: «плановое расхождение с OSM ≤ 0,5 м». Сравниваем не
    исходные координаты OSM напрямую (это уже 61 тест Шага 1.4), а то, что
    сборка IFC (Шаг 1.8) не искажает координаты, прошедшие нормализацию —
    именно здесь мог бы появиться сдвиг/перепутанная ось при мешировании."""
    dbname = pg_test_db.info.dbname
    _import_osm(dbname, profile_center(), tmp_path, "center-fidelity")

    with _make_client(dbname, tmp_path, monkeypatch) as client:
        _seed_dem(pg_test_db, tmp_path)

        resp = client.post("/jobs", json={"center": {"lon": CENTER_LON, "lat": CENTER_LAT}, "radius_m": 500})
        job_id = resp.json()["id"]
        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "done", job

        from topology_geo.selection.service import select_site_data

        dataset = select_site_data(pg_test_db, CENTER_LON, CENTER_LAT, 500.0)
        buildings = [f for f in dataset.features if f.layer == "osm_buildings"]
        assert buildings, "источник (Шаг 1.4) не вернул зданий для сравнения"
        reference_footprint: Polygon = buildings[0].geometry

        files = client.get(f"/models/{job_id}/files").json()["files"]
        ifc_file = next(f for f in files if f["step_name"] == "assemble_ifc:IFC4X3")
        ifc_bytes = client.get(ifc_file["download_url"]).content
        model = ifcopenshell.file.from_string(ifc_bytes.decode("utf-8"))

        settings = ifcopenshell.geom.settings()
        settings.set("use-world-coords", True)
        building_elem = next(e for e in model.by_type("IfcBuildingElementProxy") if (e.Name or "").startswith("Здание"))
        shape = ifcopenshell.geom.create_shape(settings, building_elem)
        verts = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
        min_z = verts[:, 2].min()
        bottom_xy = verts[np.isclose(verts[:, 2], min_z), :2]
        exported_footprint = Polygon(bottom_xy).convex_hull

        deviation = reference_footprint.hausdorff_distance(exported_footprint)
        print(f"\nотклонение контура здания IFC от нормализованного OSM: {deviation:.6f} м")
        assert deviation <= 0.5
