"""Тесты Шага 4.1: сцена для рендера — композиция USD-слоёв, PBR-материалы,
подстановка библиотечных моделей, проверка сцены. Всё через настоящий
`pxr` (пакет `usd-core`), без заглушек: сцены реально пишутся на диск и
перечитываются обратно, композиция и подстановка проверяются по факту
прочитанной геометрии, а не по коду, который её должен был произвести."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdShade

from topology_geo.rendering.usd_scene import (
    CONTEXT_BUILDING_MATERIAL,
    apply_context_building_material,
    apply_library_substitutions,
    compose_scene,
    define_preview_surface_material,
    find_library_model_for_type,
    substitute_with_library_model,
    validate_scene,
)


def _write_mesh_layer(path: Path, prim_path: str, points: list[tuple[float, float, float]]) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points)
    n = len(points)
    mesh.CreateFaceVertexCountsAttr([n])
    mesh.CreateFaceVertexIndicesAttr(list(range(n)))
    stage.GetRootLayer().Save()


@pytest.fixture
def two_layers(tmp_path):
    building_layer = tmp_path / "buildings.usd"
    terrain_layer = tmp_path / "terrain.usd"
    _write_mesh_layer(building_layer, "/Layer/Building1", [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)])
    _write_mesh_layer(terrain_layer, "/Layer/Terrain", [(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)])
    return building_layer, terrain_layer


def test_compose_scene_combines_layers_via_sublayers(tmp_path, two_layers):
    building_layer, terrain_layer = two_layers
    output = tmp_path / "scene.usd"

    stage = compose_scene([building_layer, terrain_layer], output)

    assert stage.GetPrimAtPath("/Layer/Building1").IsValid()
    assert stage.GetPrimAtPath("/Layer/Terrain").IsValid()
    # реально записано на диск как composition arcs (subLayers), не скопированная геометрия:
    # у корневого файла своих примитивов нет, вся геометрия приходит из слоёв
    root_layer = Sdf.Layer.FindOrOpen(str(output))
    assert len(root_layer.subLayerPaths) == 2
    assert len(root_layer.rootPrims) == 0


def test_compose_scene_layer_order_determines_strength(tmp_path):
    layer_a = tmp_path / "a.usd"
    layer_b = tmp_path / "b.usd"
    _write_mesh_layer(layer_a, "/Shared", [(1, 1, 1)])
    _write_mesh_layer(layer_b, "/Shared", [(2, 2, 2)])
    output = tmp_path / "scene.usd"

    stage = compose_scene([layer_a, layer_b], output)  # a сильнее (первый в списке)

    points = stage.GetPrimAtPath("/Shared").GetAttribute("points").Get()
    assert points[0] == pytest.approx((1, 1, 1))


def test_compose_scene_sets_stage_metadata(tmp_path, two_layers):
    building_layer, terrain_layer = two_layers
    output = tmp_path / "scene.usd"

    stage = compose_scene([building_layer, terrain_layer], output)

    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0)


def test_define_preview_surface_material_has_usd_preview_surface_shader(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "mat.usd"))
    define_preview_surface_material(
        stage, "/Materials/Test", diffuse_color=(0.5, 0.5, 0.5), roughness=0.3, metallic=0.1,
    )
    shader = UsdShade.Shader(stage.GetPrimAtPath("/Materials/Test/PreviewSurface"))
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert shader.GetInput("diffuseColor").Get() == pytest.approx((0.5, 0.5, 0.5))
    assert shader.GetInput("roughness").Get() == pytest.approx(0.3)


def test_apply_context_building_material_binds_and_skips_missing(tmp_path, two_layers):
    building_layer, terrain_layer = two_layers
    stage = compose_scene([building_layer, terrain_layer], tmp_path / "scene.usd")

    bound = apply_context_building_material(stage, ["/Layer/Building1", "/Layer/DoesNotExist"])

    assert bound == ["/Layer/Building1"]
    binding = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/Layer/Building1"))
    material, _ = binding.ComputeBoundMaterial()
    shader = UsdShade.Shader(material.GetPrim().GetChild("PreviewSurface"))
    assert shader.GetInput("diffuseColor").Get() == pytest.approx(CONTEXT_BUILDING_MATERIAL["diffuse_color"])


def test_substitute_with_library_model_replaces_local_geometry(tmp_path):
    layer_path = tmp_path / "buildings.usd"
    _write_mesh_layer(layer_path, "/Layer/Tree1", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])

    library_path = tmp_path / "library_tree.usd"
    library_stage = Usd.Stage.CreateNew(str(library_path))
    tree_mesh = UsdGeom.Mesh.Define(library_stage, "/Model")
    tree_mesh.CreatePointsAttr([(0, 0, 0), (0, 0, 5), (0, 1, 5)])
    tree_mesh.CreateFaceVertexCountsAttr([3])
    tree_mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    library_stage.GetRootLayer().Save()

    applied = substitute_with_library_model(layer_path, "/Layer/Tree1", library_path, "/Model")

    assert applied is True
    reopened = Usd.Stage.Open(str(layer_path))
    points = reopened.GetPrimAtPath("/Layer/Tree1").GetAttribute("points").Get()
    assert list(points) == [(0, 0, 0), (0, 0, 5), (0, 1, 5)]  # геометрия дерева, не исходный квадрат


def test_substitute_with_library_model_returns_false_for_missing_prim(tmp_path, two_layers):
    building_layer, _ = two_layers
    applied = substitute_with_library_model(
        building_layer, "/Layer/DoesNotExist", tmp_path / "lib.usd", "/Model",
    )
    assert applied is False


def test_find_library_model_for_type_returns_none_for_current_manifest():
    """Честная проверка текущего состояния каталога (Шаг 2.8): записей
    вида kind="model" (детальные лицензированные модели деревьев/МАФ)
    в реальном manifest.json пока нет — подстановка не находит модель,
    это не ошибка, а честный результат отсутствия ассетов."""
    assert find_library_model_for_type("дерево") is None
    assert find_library_model_for_type("опора") is None  # это generator, не model


def test_find_library_model_for_type_finds_entry_in_synthetic_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    assets_dir = tmp_path / "assets"
    (assets_dir / "trees").mkdir(parents=True)
    model_file = assets_dir / "trees" / "oak.usd"
    lib_stage = Usd.Stage.CreateNew(str(model_file))
    UsdGeom.Mesh.Define(lib_stage, "/Model").CreatePointsAttr([(0, 0, 0), (0, 0, 5), (1, 0, 5)])
    lib_stage.GetRootLayer().Save()

    manifest.write_text(json.dumps([
        {
            "key": "tree_oak", "category": "дерево", "type": "дуб", "lod": "LOD2",
            "license": "CC-BY", "source": "тест", "version": "1.0", "kind": "model",
            "files": ["trees/oak.usd"], "description": "",
        }
    ]), encoding="utf-8")

    entry = find_library_model_for_type("дерево", path=manifest)
    assert entry is not None
    assert entry.key == "tree_oak"


def test_apply_library_substitutions_end_to_end_with_synthetic_library(tmp_path):
    manifest = tmp_path / "manifest.json"
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    model_file = assets_dir / "oak.usd"
    lib_stage = Usd.Stage.CreateNew(str(model_file))
    mesh = UsdGeom.Mesh.Define(lib_stage, "/Model")
    mesh.CreatePointsAttr([(0, 0, 0), (0, 0, 5), (1, 0, 5)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    lib_stage.GetRootLayer().Save()
    manifest.write_text(json.dumps([
        {
            "key": "tree_oak", "category": "дерево", "type": "дуб", "lod": "LOD2",
            "license": "CC-BY", "source": "тест", "version": "1.0", "kind": "model",
            "files": ["oak.usd"], "description": "",
        }
    ]), encoding="utf-8")

    layer_path = tmp_path / "trees_layer.usd"
    _write_mesh_layer(layer_path, "/Layer/Tree1", [(0, 0, 0), (1, 0, 0), (1, 1, 0)])

    substituted = apply_library_substitutions(
        layer_path, {"/Layer/Tree1": "дерево"}, manifest_path=manifest, assets_root=assets_dir,
    )

    assert substituted == ["/Layer/Tree1"]
    reopened = Usd.Stage.Open(str(layer_path))
    points = reopened.GetPrimAtPath("/Layer/Tree1").GetAttribute("points").Get()
    assert list(points) == [(0, 0, 0), (0, 0, 5), (1, 0, 5)]


def test_apply_library_substitutions_no_op_when_no_match(tmp_path, two_layers):
    building_layer, _ = two_layers
    substituted = apply_library_substitutions(building_layer, {"/Layer/Building1": "дерево"})
    assert substituted == []


def test_validate_scene_passes_clean_scene(tmp_path, two_layers):
    building_layer, terrain_layer = two_layers
    stage = compose_scene([building_layer, terrain_layer], tmp_path / "scene.usd")

    report = validate_scene(stage)

    assert report.mesh_count == 2
    scale_issues = [i for i in report.issues if "масштаб" in i.issue or "точек" in i.issue]
    assert scale_issues == []


def test_validate_scene_detects_out_of_range_scale(tmp_path):
    layer = tmp_path / "tiny.usd"
    # 0.5 мм в поперечнике - типичная ошибка "забыли перевести из мм в м"
    _write_mesh_layer(layer, "/Bad", [(0, 0, 0), (0.0005, 0, 0), (0.0005, 0.0005, 0)])
    stage = compose_scene([layer], tmp_path / "scene.usd")

    report = validate_scene(stage)

    assert not report.is_valid
    assert any("масштаба" in issue.issue for issue in report.issues)


def test_validate_scene_detects_out_of_bounds_face_indices(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "broken.usd"))
    mesh = UsdGeom.Mesh.Define(stage, "/Broken")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 5])  # 5 вне диапазона точек
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(tmp_path / "broken.usd"))
    report = validate_scene(reopened)

    assert not report.is_valid
    assert any("faceVertexIndices" in issue.issue for issue in report.issues)


def test_validate_scene_flags_missing_uv_and_normals_as_informational(tmp_path, two_layers):
    building_layer, terrain_layer = two_layers
    stage = compose_scene([building_layer, terrain_layer], tmp_path / "scene.usd")

    report = validate_scene(stage)

    # процедурные слои без UV/нормалей — это ожидаемо (Шаг 1.9), но должно попасть в отчёт
    assert any("UV" in issue.issue for issue in report.issues)


def test_validate_scene_empty_stage_has_zero_meshes(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "empty.usd"))
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(tmp_path / "empty.usd"))

    report = validate_scene(reopened)

    assert report.mesh_count == 0
    assert report.is_valid
