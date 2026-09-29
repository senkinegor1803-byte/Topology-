"""Тест Шага 4.1, п. 1 (конвертация GLB-слоя в USD) — единственная часть
модуля `usd_scene.py`, которой реально нужен `bpy` (Blender Python).
Собираем настоящий GLB тем же путём, что и остальной пайплайн
(`ifc.to_glb.convert_ifc_to_glb`, Шаг 1.9), конвертируем в USD через
`glb_layer_to_usd` и проверяем результат обратно через `pxr` (не через
сам `bpy` — так тест доказывает, что результат читается стандартным
USD-инструментарием, а не только тем же Blender, который его написал).

`bpy` — тяжёлая (374 МБ) опциональная зависимость, не входит в
`pyproject.toml` (см. `docs/rendering.md`); модуль целиком пропускается,
если пакет не установлен, вместо падения сборки."""

from __future__ import annotations

from pathlib import Path

import pytest
from shapely.geometry import Polygon

bpy = pytest.importorskip("bpy")

from pxr import Usd, UsdGeom

from topology_geo.geometry.buildings import BuildingSolid
from topology_geo.ifc.assemble import BasePoint, SiteModel, build_site_ifc
from topology_geo.ifc.to_glb import convert_ifc_to_glb
from topology_geo.rendering.usd_scene import glb_layer_to_usd

BASE_POINT = BasePoint(lon=56.25, lat=58.0, zone=2, x=2_310_450.0, y=-5_857_320.0, height=150.0)


def _make_single_building_glb() -> bytes:
    building = BuildingSolid(
        osm_id=1, footprint=Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
        height_m=12.0, height_confidence="факт", height_source="OSM", base_z=100.0, building_type="жилой",
    )
    model_ifc, _ = build_site_ifc("IFC4", SiteModel(buildings=[building]), BASE_POINT)
    return convert_ifc_to_glb(model_ifc)


def test_glb_layer_to_usd_produces_readable_usd_with_real_geometry(tmp_path: Path):
    glb_path = tmp_path / "building.glb"
    glb_path.write_bytes(_make_single_building_glb())
    usd_path = tmp_path / "building.usd"

    glb_layer_to_usd(glb_path, usd_path)

    assert usd_path.exists()
    assert usd_path.stat().st_size > 0

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    assert len(meshes) >= 1
    for prim in meshes:
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        assert points is not None and len(points) > 0

    # мировые координаты (не локальные — компенсирующий поворот в
    # glb_layer_to_usd хранится как xformOp на родительском Xform, не
    # запечён в points меша): BBoxCache честно проходит всю иерархию.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
    bbox = cache.ComputeWorldBound(stage.GetPrimAtPath("/"))
    bounds = bbox.ComputeAlignedRange()
    size = bounds.GetMax() - bounds.GetMin()

    # проверка масштаба и ориентации (Шаг 4.1, п. 4): здание 20x20 м в
    # плане, высота 12 м, база Z=100 - координаты после конвертации
    # должны остаться в метрах того же порядка и на той же оси (Z),
    # не «лечь на бок» из-за glTF-конвенции Y-вверх при импорте bpy
    assert size[0] == pytest.approx(20.0, abs=2.0)  # X — ширина по плану
    assert size[1] == pytest.approx(20.0, abs=2.0)  # Y — длина по плану
    assert size[2] == pytest.approx(12.0, abs=2.0)  # Z — высота, НЕ в Y
    assert bounds.GetMin()[2] == pytest.approx(100.0, abs=2.0)  # база Z=100
