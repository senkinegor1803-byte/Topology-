"""CityJSON экспорт (Шаг 2.10, п. 2: «CityJSON (здания)»).

CityJSON 1.1, пишется напрямую как `dict` (JSON-совместимая структура,
`json.dumps` снаружи) — формат сам по себе просто JSON, специальная
библиотека не нужна.

Геометрия зданий берётся ТЕМ ЖЕ кодом, что и в `ifc/assemble.py` (строки
~450-464): плоская крыша -> `extrude_polygon_mesh` по исходному контуру;
скатная -> `oriented_bounding_box` + `build_pitched_building_mesh`. Так
исключено расхождение между IFC- и CityJSON-геометрией одного здания.

Три уровня детализации на здание (план требует явно «CityJSON (LOD0-2)»):
- LOD 0 — плоский контур пятна застройки на отметке `base_z`
  (`MultiSurface`, БЕЗ экструзии — по определению LOD0: только «footprint»/
  «roof edge», высота не участвует);
- LOD 1 — «блок»: `extrude_polygon_mesh(footprint, base_z, base_z + height_m)`
  (простая призма на всю высоту, форма крыши не учитывается — по определению
  LOD1 в CityGML/CityJSON: обобщённый параллелепипед/призма);
- LOD 2 — «с крышей»: та же логика, что и IFC (для плоской крыши совпадает
  с LOD1 геометрически — это честно, не искусственно различается).

Упрощения (сознательно, не скрыто):
- Каждая треугольная грань меша записывается как ОТДЕЛЬНАЯ поверхность
  `Solid`-границы (`[[i, j, k]]`), а не объединённые плоские полигоны стен/
  ската — валидно по спецификации CityJSON, но более многословно, чем
  вручную объединённые грани. Упрощает код, не меняет геометрию.
- Список вершин ГЛОБАЛЬНЫЙ, БЕЗ дедупликации между зданиями (одна и та же
  точка на границе двух объектов будет продублирована) — не проблема
  корректности (каждое здание геометрически замкнуто само по себе), только
  не самый компактный файл.
- `transform` — тождественный (`scale=[1,1,1]`, `translate=[0,0,0]`),
  координаты пишутся как есть (локальные метры участка), БЕЗ целочисленного
  квантования (упрощение точности, применяемое в реальных CityJSON-экспортёрах
  для сжатия, здесь не делается).
- `metadata.referenceSystem` (EPSG-код) НЕ указывается: у МСК-59 в этом
  проекте нет декларированного глобального EPSG-кода (см. `coords.py`) —
  честнее не иметь поля, чем указать неверный/придуманный код.
"""

from __future__ import annotations

from typing import Protocol

from topology_geo.geometry.roofs import ROOF_FLAT, RoofParams, build_pitched_building_mesh, oriented_bounding_box
from topology_geo.ifc.assemble import extrude_polygon_mesh

CITYJSON_VERSION = "1.1"


class _BuildingLike(Protocol):
    osm_id: int
    footprint: object  # shapely Polygon
    base_z: float
    height_m: float
    height_confidence: str
    building_type: str
    roof_shape: str
    roof_height_m: float
    roof_ridge_along_long_axis: bool
    roof_direction: tuple[float, float] | None


def _lod0_footprint_boundaries(vertices: list[list[float]], footprint, z: float) -> list[list[list[int]]]:
    """LOD0: плоский контур пятна застройки (`MultiSurface`, boundaries -
    массив поверхностей, БЕЗ обёртки оболочкой, в отличие от `Solid`), одна
    поверхность на кольцо (внешний контур + отверстия дворов как есть, БЕЗ
    триангуляции — полигон плоский, ring с отверстиями допустим напрямую)."""
    polygons = footprint.geoms if footprint.geom_type.startswith("Multi") else [footprint]
    surfaces: list[list[list[int]]] = []
    for polygon in polygons:
        rings = [polygon.exterior, *polygon.interiors]
        surface: list[list[int]] = []
        for ring in rings:
            offset = len(vertices)
            vertices.extend([float(x), float(y), float(z)] for x, y in list(ring.coords)[:-1])
            surface.append(list(range(offset, len(vertices))))
        surfaces.append(surface)
    return surfaces


def _append_mesh(vertices: list[list[float]], mesh: tuple[list, list]) -> list[list[list[int]]]:
    """Добавить меш (verts, faces) в общий список вершин, вернуть boundaries
    Solid-геометрии CityJSON (одна оболочка, по одной треугольной грани на
    поверхность)."""
    mesh_verts, mesh_faces = mesh
    offset = len(vertices)
    vertices.extend([float(c) for c in v] for v in mesh_verts)
    surfaces = [[[int(idx) + offset for idx in face]] for face in mesh_faces]
    return [surfaces]  # одна оболочка (без внутренних полостей)


def _building_lod2_mesh(building: _BuildingLike) -> tuple[list, list]:
    if building.roof_shape == ROOF_FLAT:
        return extrude_polygon_mesh(building.footprint, building.base_z, building.base_z + building.height_m)
    obb = oriented_bounding_box(building.footprint)
    roof_params = RoofParams(
        shape=building.roof_shape,
        shape_confidence=building.roof_height_confidence,
        height_m=building.roof_height_m,
        height_confidence=building.roof_height_confidence,
        ridge_along_long_axis=building.roof_ridge_along_long_axis,
        direction=building.roof_direction,
    )
    eave_z = building.base_z + building.height_m - building.roof_height_m
    return build_pitched_building_mesh(obb, roof_params, building.base_z, eave_z)


def _geographical_extent(vertices: list[list[float]]) -> list[float] | None:
    if not vertices:
        return None
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    return [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)]


def build_cityjson(buildings: list[_BuildingLike]) -> dict:
    """Собрать CityJSON-документ: по одному `Building` CityObject на каждое
    здание участка, с геометрией LOD0 (контур) + LOD1 (блок) + LOD2 (форма
    крыши) — план явно требует «CityJSON (LOD0-2)»."""
    vertices: list[list[float]] = []
    city_objects: dict[str, dict] = {}

    for building in buildings:
        lod1_mesh = extrude_polygon_mesh(building.footprint, building.base_z, building.base_z + building.height_m)
        lod2_mesh = _building_lod2_mesh(building)

        # LOD1/LOD2 добавляются всегда, даже когда меш LOD2 совпадает с LOD1
        # (плоская крыша) - чтобы у всех зданий был одинаковый набор LOD
        # (проще для потребителя CityJSON), а не условно, по форме крыши.
        # Небольшое дублирование вершин, не проблема корректности.
        geometries = [
            {
                "type": "MultiSurface", "lod": "0",
                "boundaries": _lod0_footprint_boundaries(vertices, building.footprint, building.base_z),
            },
            {"type": "Solid", "lod": "1", "boundaries": _append_mesh(vertices, lod1_mesh)},
            {"type": "Solid", "lod": "2", "boundaries": _append_mesh(vertices, lod2_mesh)},
        ]

        city_objects[f"building-{building.osm_id}"] = {
            "type": "Building",
            "attributes": {
                "высота_м": building.height_m,
                "источник_высоты": building.height_confidence,
                "тип": building.building_type,
                "форма_крыши": building.roof_shape,
            },
            "geometry": geometries,
        }

    doc: dict = {
        "type": "CityJSON",
        "version": CITYJSON_VERSION,
        "transform": {"scale": [1.0, 1.0, 1.0], "translate": [0.0, 0.0, 0.0]},
        "CityObjects": city_objects,
        "vertices": vertices,
    }
    extent = _geographical_extent(vertices)
    if extent is not None:
        doc["metadata"] = {"geographicalExtent": extent}
    return doc
