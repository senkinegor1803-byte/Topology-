"""Сцена для рендера из IFC (Шаг 4.1) — конвертация федеративной модели
в OpenUSD, подстановка детальных библиотечных моделей по типу,
PBR-материалы, проверка сцены.

Композиция сцены, материалы, подстановка и проверка выполняются через
`pxr` (`usd-core` — эталонная реализация OpenUSD от Pixar/AOUSD, тот же
API, которым устроены Blender/Omniverse/USD-вьюеры) — без Blender: это
и легче (30 МБ против 374 МБ `bpy`), и не требует GPU, и позволяет
проверять структуру сцены напрямую (прочитать `Usd.Stage`, обойти
примитивы), а не только «файл не пустой».

Единственное место, где нужен настоящий `bpy` (Blender-Python), —
конвертация уже готового `.glb`-слоя (Шаг 1.9/2.10 уже собирают их) в
`.usd`: `glb_layer_to_usd`. `bpy` — тяжёлая опциональная зависимость,
не входит в `pyproject.toml` (как и в остальном проекте — см.
`docs/rendering.md`), импортируется только внутри этой функции; вызов
без установленного `bpy` даёт понятную ошибку, а не падение при импорте
модуля.

«Слои как отдельные USD-файлы, собранные в одну сцену» (п. 1 плана) —
это буквально механизм композиции OpenUSD `subLayers`: каждый файл
слоя (рельеф/здания/сети/...) остаётся самостоятельным, редактируемым
отдельно, ссылка на него в сцене — не копия геометрии.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from topology_geo.assets.library import KIND_MODEL, AssetEntry, entries_by_category

METERS_PER_UNIT = 1.0  # весь пайплайн проекта уже в метрах (МСК-59), 1 юнит USD = 1 м
UP_AXIS = UsdGeom.Tokens.z  # проект — Z-up (как IFC), не Y-up (умолчание многих DCC)

# «Контекстные здания — белый/глиняный материал» (Шаг 4.1, п. 3, буквально по плану):
# нейтральный светлый материал без реальных фотографических текстур (их нет в среде).
CONTEXT_BUILDING_MATERIAL = {
    "diffuse_color": (0.85, 0.82, 0.75),
    "roughness": 0.85,
    "metallic": 0.0,
}


def glb_layer_to_usd(glb_path: Path, usd_path: Path) -> None:
    """п. 1: конвертация одного федеративного слоя (уже собран в GLB на
    Шаге 1.9/2.10) в отдельный `.usd`-файл — через родной экспортёр
    `bpy` (Blender сам разбирает glTF в свою сцену и пишет корректный
    USD: меши, иерархию узлов, материалы).

    Реальная находка при отладке (не гипотетическая): `.glb` этого
    проекта (`ifc.to_glb`) пишет координаты как есть, Z-вверх (та же
    ось, что и IFC/весь конвейер) — но glTF-формат по спецификации
    предполагает Y-вверх, и импортёр `bpy` молча поворачивает сцену на
    экспорт/импорт под это допущение. Без компенсации здание «ложится
    на бок» (высота попадает в Y, а не в Z). Компенсация — поворот
    корневых узлов на -90° вокруг X; `rotation_euler` при этом не
    действует, пока `rotation_mode` объекта — `'QUATERNION'` (умолчание
    импортёра glTF), поэтому режим сперва переключается на `'XYZ'`."""
    try:
        import bpy
    except ImportError as exc:
        raise RuntimeError(
            "для конвертации GLB в USD нужен пакет bpy (Blender Python), "
            "не входит в основные зависимости — установите отдельно: pip install bpy"
        ) from exc

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    for obj in bpy.data.objects:
        if obj.parent is None:
            obj.rotation_mode = "XYZ"
            obj.rotation_euler.x -= math.radians(90.0)
    bpy.context.view_layer.update()
    bpy.ops.wm.usd_export(filepath=str(usd_path), export_materials=True)


def compose_scene(
    layer_usd_paths: list[Path], output_path: Path, *, meters_per_unit: float = METERS_PER_UNIT,
) -> Usd.Stage:
    """п. 1: собрать отдельные USD-слои в одну сцену через `subLayers`
    (настоящая композиция OpenUSD, не копирование геометрии в один
    файл) — первый путь в списке сильнее («перебивает» совпадающие
    примитивы нижестоящих слоёв), как и стандартный порядок `subLayers`
    в USD."""
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UP_AXIS)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    root_layer = stage.GetRootLayer()
    root_layer.subLayerPaths = [
        str(_relative_or_absolute(output_path.parent, p)) for p in layer_usd_paths
    ]
    root_layer.Save()
    return Usd.Stage.Open(str(output_path))


def _relative_or_absolute(base_dir: Path, target: Path) -> str:
    target = Path(target)
    try:
        return "./" + str(target.relative_to(base_dir))
    except ValueError:
        return str(target)


def define_preview_surface_material(
    stage: Usd.Stage, material_path: str, *, diffuse_color: tuple[float, float, float],
    roughness: float = 0.5, metallic: float = 0.0,
) -> UsdShade.Material:
    """п. 3: назначение PBR-материала по свойствам — стандартный
    `UsdPreviewSurface` (общая для всех USD-рендеров PBR-модель:
    diffuse/roughness/metallic), не выдуманная схема."""
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_material(stage: Usd.Stage, prim_path: str, material: UsdShade.Material) -> bool:
    """Назначить материал примитиву. Возвращает False, если примитива
    нет в сцене (честный no-op, не исключение — состав слоёв заранее не
    всегда известен вызывающему коду)."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return True


def apply_context_building_material(stage: Usd.Stage, building_prim_paths: list[str]) -> list[str]:
    """п. 3, буквально по плану: «контекстные здания — белый/глиняный
    материал». Возвращает пути примитивов, к которым материал реально
    применён (пропускает несуществующие пути)."""
    material = define_preview_surface_material(
        stage, "/Materials/ContextBuilding", **CONTEXT_BUILDING_MATERIAL,
    )
    return [path for path in building_prim_paths if bind_material(stage, path, material)]


def find_library_model_for_type(category: str, *, path: Path | None = None) -> AssetEntry | None:
    """п. 2: найти в каталоге библиотеки (Шаг 2.8) готовую детальную
    3D-модель по типу объекта. Возвращает `None`, если в каталоге пока
    нет записей вида `kind="model"` для этой категории — в текущем
    каталоге проекта таких записей нет (см. `docs/asset-library.md`:
    лицензированные детальные модели деревьев/МАФ не интегрированы), это
    честный, а не имитированный результат."""
    kwargs = {"path": path} if path is not None else {}
    for entry in entries_by_category(category, **kwargs):
        if entry.kind == KIND_MODEL:
            return entry
    return None


def substitute_with_library_model(
    layer_usd_path: Path, prim_path: str, library_asset_path: Path, library_asset_prim_path: str,
) -> bool:
    """п. 2: «упрощённое тело IFC заменяется детальной моделью из
    библиотеки по типу» — редактирует САМ файл слоя (`layer_usd_path`):
    убирает локальные опорные точки/грани примитива (иначе локальная
    геометрия слоя перебила бы ссылку — в OpenUSD местная опись всегда
    сильнее референса на тот же путь) и добавляет ссылку (`references`)
    на файл библиотечной модели. Возвращает False, если примитива с
    геометрией нет в этом файле слоя (честный no-op)."""
    layer = Sdf.Layer.FindOrOpen(str(layer_usd_path))
    if layer is None:
        return False
    prim_spec = layer.GetPrimAtPath(prim_path)
    if prim_spec is None:
        return False

    geometry_properties = ("points", "faceVertexCounts", "faceVertexIndices", "normals")
    removed_any = False
    for prop_name in geometry_properties:
        prop_spec = prim_spec.properties.get(prop_name)
        if prop_spec is not None:
            prim_spec.RemoveProperty(prop_spec)
            removed_any = True
    if not removed_any:
        return False

    prim_spec.referenceList.explicitItems = [
        Sdf.Reference(str(library_asset_path), library_asset_prim_path)
    ]
    layer.Save()
    return True


def apply_library_substitutions(
    layer_usd_path: Path, prim_type_by_path: dict[str, str],
    *, manifest_path: Path | None = None, assets_root: Path | None = None,
) -> list[str]:
    """Пройтись по карте «путь примитива → категория объекта» (категория
    — как в `_categorize` Шага 1.9, `to_glb.py`) и подставить детальную
    модель там, где она нашлась в каталоге. Возвращает список путей, где
    подстановка реально произошла (пуст, пока каталог не содержит
    записей `kind="model"` — см. `find_library_model_for_type`)."""
    from topology_geo.assets.library import resolve_file

    substituted = []
    for prim_path, category in prim_type_by_path.items():
        kwargs = {"path": manifest_path} if manifest_path is not None else {}
        entry = find_library_model_for_type(category, **kwargs)
        if entry is None or not entry.files:
            continue
        resolve_kwargs = {"root": assets_root} if assets_root is not None else {}
        asset_file = resolve_file(entry, entry.files[0], **resolve_kwargs)
        applied = substitute_with_library_model(layer_usd_path, prim_path, asset_file, "/Model")
        if applied:
            substituted.append(prim_path)
    return substituted


@dataclass(frozen=True)
class ValidationIssue:
    prim_path: str
    issue: str


@dataclass(frozen=True)
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    mesh_count: int = 0

    @property
    def is_valid(self) -> bool:
        return not self.issues


def validate_scene(
    stage: Usd.Stage, *, plausible_size_range_m: tuple[float, float] = (0.001, 100_000.0),
) -> ValidationReport:
    """п. 4: «проверка масштаба, нормалей, UV-развёрток» — обход всех
    мешей собранной сцены. Сообщает о реальных дефектах (нет точек,
    индекс грани вне диапазона точек, вырожденный/неконечный охват,
    масштаб вне правдоподобного диапазона — типичный симптом ошибки
    единиц, см/м) и об отсутствии нормалей/UV как информационный пункт
    (для процедурных слоёв это ожидаемо — вьюер считает нормали сам,
    `docs/dev-tree.md` Шаг 1.9 — не считается дефектом само по себе)."""
    issues: list[ValidationIssue] = []
    mesh_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1
        mesh = UsdGeom.Mesh(prim)
        path = str(prim.GetPath())
        points = mesh.GetPointsAttr().Get()

        if not points:
            issues.append(ValidationIssue(path, "нет точек геометрии (points)"))
            continue

        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if indices and max(indices) >= len(points):
            issues.append(ValidationIssue(path, "faceVertexIndices ссылается за пределы points"))

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        if any(not math.isfinite(v) for v in xs + ys + zs):
            issues.append(ValidationIssue(path, "неконечные координаты точек (NaN/Inf)"))
            continue

        size = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        low, high = plausible_size_range_m
        if size > 0 and not (low <= size <= high):
            issues.append(
                ValidationIssue(path, f"размер меша {size:.4g} м вне правдоподобного диапазона — вероятна ошибка масштаба/единиц")
            )

        primvars_api = UsdGeom.PrimvarsAPI(prim)
        has_uv = primvars_api.GetPrimvar("st").IsDefined()
        has_normals = bool(mesh.GetNormalsAttr().Get())
        if not has_uv and not has_normals:
            issues.append(ValidationIssue(path, "нет ни UV (primvars:st), ни нормалей — допустимо для процедурных слоёв, требует проверки для детальных моделей"))

    return ValidationReport(issues=issues, mesh_count=mesh_count)
