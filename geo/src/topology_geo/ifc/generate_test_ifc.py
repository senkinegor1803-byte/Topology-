"""Генератор тестовых IFC-сцен для проверки совместимости (Шаг 0.2 плана).

Цель — не полноценная генерация модели окружения (это Этап 1+), а минимальная
сцена со всеми классами и приёмами, которые Шаг 0.2 требует проверить в Renga
и Pilot-BIM:

- рельеф как TIN (`IfcGeographicElement`, PredefinedType TERRAIN, тело —
  `IfcTriangulatedFaceSet`, разрешение сетки регулируется — им же управляется
  размер файла для нагрузочного теста);
- `IfcRoad`, `IfcBridge` — нативно в IFC4X3 (IFC 4.3); в IFC4 их нет в схеме,
  поэтому используется `IfcBuildingElementProxy` с пометкой в
  `Pset_Контекст.Заменяет_класс` — это и есть случай «как прокси» из таблицы
  совместимости, которую должен заполнить BIM-специалист вручную (шаг всё
  равно требует открытия в Renga/Pilot-BIM человеком);
- `IfcBuildingElementProxy`, `IfcGeographicElement` (объект окружения),
  `IfcSpatialZone`, `IfcPipeSegment`;
- пользовательские наборы свойств с кириллицей на каждом объекте
  (`Pset_Контекст`, `Pset_Источник`, `Pset_Ограничение`);
- геопривязка через `IfcMapConversion` в больших координатах (условные
  координаты МСК-59 — см. `topology_geo.coords`, реальная база точка
  появится после выбора пилота, Шаг 0.1).

Валидация — обёртка над `ifcopenshell.validate.validate`.

Использование как CLI:

    python -m topology_geo.ifc.generate_test_ifc --schema IFC4X3 --out out/test_ifc43.ifc
    python -m topology_geo.ifc.generate_test_ifc --schema IFC4 --out out/test_ifc4.ifc --validate

Нагрузочный тест (п. 5 шага 0.2, «IFC на 50/200/1000 МБ») запускается отдельно
через `--relief-grid N` (крупная TIN-сетка) и/или `--extra-buildings N`
(повтор простых прокси-зданий) — см. `docs/ifc-compatibility.md`.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.geometry
import ifcopenshell.api.georeference
import ifcopenshell.api.project
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.unit
import ifcopenshell.validate
import numpy as np

SUPPORTED_SCHEMAS = ("IFC4", "IFC4X3")

# Классы, которые IFC 4.3 (IFC4X3) знает нативно, а IFC4 — нет.
IFC43_ONLY_CLASSES = ("IfcRoad", "IfcBridge")

CYRILLIC_PSETS: dict[str, dict[str, Any]] = {
    "Pset_Контекст": {
        "Источник_создан": "Топология: генератор тестовых сцен (Шаг 0.2)",
        "Комментарий": "Тестовый объект для проверки совместимости IFC с Renga/Pilot-BIM",
    },
    "Pset_Источник": {
        "Организация": "ОАО «Пример РСО»",
        "Дата_получения": "2026-09-25",
        "Формат_исходника": "DWG",
    },
    "Pset_Ограничение": {
        "Вид_ограничения": "Охранная зона ЛЭП",
        "Статус": "расчётно",
        "Напряжение_кВ": 10.0,
    },
}


@dataclass(frozen=True)
class MapConversionParams:
    """Параметры геопривязки IfcMapConversion (условные большие координаты).

    По умолчанию — координаты, близкие к центру Перми в зоне 2 МСК-59
    (см. `topology_geo.coords`), условно, пока пилотный участок не выбран
    (Шаг 0.1). Заменить на реальную базовую точку после его выбора.
    """

    crs_name: str = "MSK-59 zone 2 (условно, см. docs/coordinate-systems.md)"
    eastings: float = 2_310_450.0
    northings: float = -5_857_320.0
    height: float = 150.0
    x_axis_abscissa: float = 1.0
    x_axis_ordinate: float = 0.0
    scale: float = 1.0


Vertex = tuple[float, float, float]
Face = tuple[int, ...]


def mesh_box(length: float, width: float, height: float) -> tuple[list[Vertex], list[Face]]:
    """Прямоугольный параллелепипед со стороной у пола (0,0,0)."""
    hl, hw = length / 2, width / 2
    verts: list[Vertex] = [
        (-hl, -hw, 0.0), (hl, -hw, 0.0), (hl, hw, 0.0), (-hl, hw, 0.0),
        (-hl, -hw, height), (hl, -hw, height), (hl, hw, height), (-hl, hw, height),
    ]
    faces: list[Face] = [
        (0, 1, 2), (0, 2, 3),  # низ
        (4, 6, 5), (4, 7, 6),  # верх
        (0, 5, 1), (0, 4, 5),  # y-
        (1, 6, 2), (1, 5, 6),  # x+
        (2, 7, 3), (2, 6, 7),  # y+
        (3, 4, 0), (3, 7, 4),  # x-
    ]
    return verts, faces


def mesh_cylinder(radius: float, length: float, segments: int = 12) -> tuple[list[Vertex], list[Face]]:
    """Цилиндр вдоль оси Z (упрощённая труба, `IfcPipeSegment`)."""
    if segments < 3:
        raise ValueError("segments must be >= 3")
    verts: list[Vertex] = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        verts.append((radius * math.cos(angle), radius * math.sin(angle), 0.0))
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        verts.append((radius * math.cos(angle), radius * math.sin(angle), length))

    faces: list[Face] = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((i, j, segments + j))
        faces.append((i, segments + j, segments + i))
    for i in range(1, segments - 1):
        faces.append((0, i + 1, i))  # нижняя крышка
    base = segments
    for i in range(1, segments - 1):
        faces.append((base, base + i, base + i + 1))  # верхняя крышка
    return verts, faces


def mesh_terrain_grid(
    nx: int, ny: int, cell_size: float = 10.0, amplitude: float = 3.0
) -> tuple[list[Vertex], list[Face]]:
    """Регулярная TIN-сетка рельефа (2 треугольника на ячейку).

    `nx`/`ny` управляют числом вершин/треугольников и, соответственно, размером
    файла — используется для нагрузочного теста (п. 5 Шага 0.2).
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")

    verts: list[Vertex] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y = i * cell_size, j * cell_size
            z = amplitude * math.sin(x / 40.0) * math.cos(y / 40.0)
            verts.append((x, y, z))

    def idx(i: int, j: int) -> int:
        return j * (nx + 1) + i

    faces: list[Face] = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))
    return verts, faces


def _translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def add_cyrillic_psets(
    f: ifcopenshell.file, product: ifcopenshell.entity_instance.entity_instance, extra: dict[str, Any] | None = None
) -> None:
    """Навесить на объект стандартный набор тестовых Pset с кириллицей."""
    psets = {k: dict(v) for k, v in CYRILLIC_PSETS.items()}
    if extra:
        psets.setdefault("Pset_Контекст", {}).update(extra)
    for name, properties in psets.items():
        pset = ifcopenshell.api.pset.add_pset(f, product=product, name=name)
        ifcopenshell.api.pset.edit_pset(f, pset=pset, properties=properties)


def _add_mesh_product(
    f: ifcopenshell.file,
    body_context: ifcopenshell.entity_instance.entity_instance,
    ifc_class: str,
    name: str,
    predefined_type: str | None,
    mesh: tuple[list[Vertex], list[Face]],
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    extra_pset: dict[str, Any] | None = None,
) -> ifcopenshell.entity_instance.entity_instance:
    verts, faces = mesh
    product = ifcopenshell.api.root.create_entity(
        f, ifc_class=ifc_class, name=name, predefined_type=predefined_type
    )
    rep = ifcopenshell.api.geometry.add_mesh_representation(
        f, context=body_context, vertices=[verts], faces=[faces]
    )
    ifcopenshell.api.geometry.assign_representation(f, product=product, representation=rep)
    ifcopenshell.api.geometry.edit_object_placement(f, product=product, matrix=_translation_matrix(*location))
    add_cyrillic_psets(f, product, extra_pset)
    return product


def build_test_model(
    schema: str,
    *,
    map_conversion: MapConversionParams | None = None,
    relief_grid: int = 8,
    extra_buildings: int = 0,
) -> ifcopenshell.file:
    """Собрать тестовую сцену со всеми классами Шага 0.2 в заданной схеме."""
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"Неподдерживаемая схема {schema!r}; ожидается одна из {SUPPORTED_SCHEMAS}")
    map_conversion = map_conversion or MapConversionParams()

    f = ifcopenshell.api.project.create_file(version=schema)
    project = ifcopenshell.api.root.create_entity(f, ifc_class="IfcProject", name="Топология — тест IFC (Шаг 0.2)")
    ifcopenshell.api.unit.assign_unit(f)

    model_context = ifcopenshell.api.context.add_context(f, context_type="Model")
    body_context = ifcopenshell.api.context.add_context(
        f, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=model_context
    )

    site = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSite", name="Пилотный участок (условно)")
    ifcopenshell.api.aggregate.assign_object(f, products=[site], relating_object=project)

    # Геопривязка в больших координатах (IfcMapConversion + IfcProjectedCRS).
    ifcopenshell.api.georeference.add_georeferencing(f)
    ifcopenshell.api.georeference.edit_georeferencing(
        f,
        projected_crs={"Name": map_conversion.crs_name},
        coordinate_operation={
            "Eastings": map_conversion.eastings,
            "Northings": map_conversion.northings,
            "OrthogonalHeight": map_conversion.height,
            "XAxisAbscissa": map_conversion.x_axis_abscissa,
            "XAxisOrdinate": map_conversion.x_axis_ordinate,
            "Scale": map_conversion.scale,
        },
    )

    # IfcRoad/IfcBridge (IFC4X3) и IfcSpatialZone — это IfcSpatialElement
    # (как IfcSite), а не обычный IfcElement: их роднит с сайтом
    # aggregate.assign_object, а не spatial.assign_container.
    products = []
    spatial_children = []

    relief = _add_mesh_product(
        f, body_context, "IfcGeographicElement", "Рельеф (TIN, тест)", "TERRAIN",
        mesh_terrain_grid(relief_grid, relief_grid),
    )
    products.append(relief)

    if schema == "IFC4X3":
        road = _add_mesh_product(
            f, body_context, "IfcRoad", "Дорога (тест)", None,
            mesh_box(40.0, 6.0, 0.2), location=(0.0, 20.0, 0.0),
        )
        spatial_children.append(road)
    else:
        road = _add_mesh_product(
            f, body_context, "IfcBuildingElementProxy", "Дорога (тест, прокси IFC4)", "USERDEFINED",
            mesh_box(40.0, 6.0, 0.2), location=(0.0, 20.0, 0.0),
            extra_pset={"Заменяет_класс": "IfcRoad (недоступен в IFC4)"},
        )
        products.append(road)

    if schema == "IFC4X3":
        bridge = _add_mesh_product(
            f, body_context, "IfcBridge", "Мост (тест)", None,
            mesh_box(30.0, 8.0, 1.0), location=(0.0, -30.0, 5.0),
        )
        spatial_children.append(bridge)
    else:
        bridge = _add_mesh_product(
            f, body_context, "IfcBuildingElementProxy", "Мост (тест, прокси IFC4)", "USERDEFINED",
            mesh_box(30.0, 8.0, 1.0), location=(0.0, -30.0, 5.0),
            extra_pset={"Заменяет_класс": "IfcBridge (недоступен в IFC4)"},
        )
        products.append(bridge)

    building = _add_mesh_product(
        f, body_context, "IfcBuildingElementProxy", "Здание-прокси (тест)", "USERDEFINED",
        mesh_box(20.0, 15.0, 9.0), location=(50.0, 0.0, 0.0),
    )
    products.append(building)

    for i in range(extra_buildings):
        products.append(
            _add_mesh_product(
                f, body_context, "IfcBuildingElementProxy", f"Здание-прокси (стресс {i})", "USERDEFINED",
                mesh_box(12.0, 10.0, 6.0), location=(80.0 + 20.0 * (i % 30), 20.0 * (i // 30), 0.0),
            )
        )

    tree = _add_mesh_product(
        f, body_context, "IfcGeographicElement", "Дерево (тест)", "USERDEFINED",
        mesh_cylinder(0.3, 6.0), location=(-30.0, 10.0, 0.0),
    )
    products.append(tree)

    zone = _add_mesh_product(
        f, body_context, "IfcSpatialZone", "ЗОУИТ (тест)", "USERDEFINED",
        mesh_box(60.0, 60.0, 0.1), location=(0.0, 0.0, -0.1),
    )
    spatial_children.append(zone)

    pipe = _add_mesh_product(
        f, body_context, "IfcPipeSegment", "Труба (тест)", "CULVERT",
        mesh_cylinder(0.15, 25.0, segments=10), location=(-10.0, -20.0, -1.5),
    )
    products.append(pipe)

    ifcopenshell.api.spatial.assign_container(f, products=products, relating_structure=site)
    ifcopenshell.api.aggregate.assign_object(f, products=spatial_children, relating_object=site)

    return f


def validate_model(f: ifcopenshell.file) -> list[dict[str, Any]]:
    """Прогнать ifcopenshell.validate и вернуть список проблем (пусто = ок)."""
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(f, logger)
    return logger.statements


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", choices=[*SUPPORTED_SCHEMAS, "both"], default="both")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--relief-grid", type=int, default=8, help="разрешение TIN-сетки рельефа (NxN ячеек)")
    parser.add_argument("--extra-buildings", type=int, default=0, help="доп. здания-прокси для нагрузочного теста")
    parser.add_argument("--validate", action="store_true", help="прогнать ifcopenshell.validate после генерации")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    schemas = SUPPORTED_SCHEMAS if args.schema == "both" else (args.schema,)
    args.out.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for schema in schemas:
        model = build_test_model(schema, relief_grid=args.relief_grid, extra_buildings=args.extra_buildings)
        out_path = args.out / f"test_{schema.lower()}.ifc"
        model.write(str(out_path))
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"{schema}: {out_path} ({size_mb:.2f} МБ)")

        if args.validate:
            issues = validate_model(model)
            if issues:
                exit_code = 1
                print(f"  Валидация: {len(issues)} замечаний")
                for issue in issues[:20]:
                    print(f"    - [{issue.get('level')}] {issue.get('message')}")
            else:
                print("  Валидация: без замечаний")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
