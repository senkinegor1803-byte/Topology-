"""Реализация шагов пайплайна задачи (Шаг 1.3).

`select_osm`, `prepare_relief`, `select_and_normalize`, `assemble_ifc`,
`convert_to_glb`, `generate_tileset` и `package_outputs` — реальные шаги,
использующие уже реализованные Шаги 1.1, 1.2, 1.4-1.9, 2.1, 2.10, 2.11.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ifcopenshell
import numpy as np
from affine import Affine

from topology_geo import __version__
from topology_geo.coords import MSK59_ZONES, pick_msk59_zone, wgs84_to_msk59
from topology_geo.export.cityjson import build_cityjson
from topology_geo.export.dxf import build_dxf
from topology_geo.export.landxml import build_landxml
from topology_geo.geometry.bridges import build_bridge_ribbons
from topology_geo.geometry.buildings import NullOvertureSource, extrude_buildings
from topology_geo.geometry.landscaping import build_benches, build_fences, build_streetlamps
from topology_geo.geometry.power import (
    build_poles,
    build_power_safety_zones,
    build_substations,
    build_wire_spans,
    place_calculated_poles,
)
from topology_geo.geometry.rail import (
    build_level_crossings,
    build_platform_areas,
    build_rail_ribbons,
    place_catenary_poles,
)
from topology_geo.geometry.road_network import NETWORK_BACKBONE, NETWORK_INTERNAL
from topology_geo.geometry.roads import build_road_ribbons
from topology_geo.geometry.streets import StreetNetwork, build_lane_network, is_osm2streets_available
from topology_geo.geometry.vegetation import (
    build_individual_trees,
    build_lawns,
    scatter_forest_trees,
    scatter_shrubs,
)
from topology_geo.geometry.water import build_water_areas, build_waterway_ribbons
from topology_geo.ifc.assemble import BasePoint, SiteModel, build_site_ifc
from topology_geo.ifc.generate_test_ifc import validate_model
from topology_geo.ifc.registry import ensure_schema as ensure_ifc_registry_schema
from topology_geo.ifc.registry import register_global_ids
from topology_geo.ifc.to_glb import convert_ifc_to_glb
from topology_geo.jobs import store
from topology_geo.osm.import_log import all_latest_sources
from topology_geo.osm.import_log import ensure_schema as ensure_import_log_schema
from topology_geo.osm.queries import count_within_radius
from topology_geo.osm.raw_roads import fetch_raw_roads_in_buffer
from topology_geo.relief.cog import to_cog
from topology_geo.relief.coverage import find_coverage
from topology_geo.relief.coverage import ensure_schema as ensure_dem_coverage_schema
from topology_geo.relief.service import Grid, get_dem, read_relief_from_storage
from topology_geo.relief.tin import build_site_tin, sample_bilinear
from topology_geo.selection.geopackage import dataset_to_geopackage_bytes
from topology_geo.selection.service import select_site_data
from topology_geo.storage import ObjectStorage
from topology_geo.tiling.grid import (
    LOD0,
    LOD0_MAX_M,
    LOD1,
    LOD2,
    TileIndex,
    classify_lod_ring,
    tiles_covering_circle,
    world_to_tile_index,
)
from topology_geo.tiling.tile_content import build_tile_content
from topology_geo.tiling.tileset import TileContentEntry, build_tileset_json
from topology_geo.cadastre import search_cadastre_by_coords

RELIEF_PIXEL_SIZE_M = 1.0
RELIEF_MARGIN_M = 200.0
IFC_SCHEMAS = ("IFC4", "IFC4X3")


def select_osm(conn: Any, storage: ObjectStorage, job: store.Job) -> dict:
    """Шаг 1: посчитать объекты OSM в радиусе задачи (Шаг 1.1) и сохранить сводку."""
    try:
        counts = count_within_radius(conn, lon=job.center_lon, lat=job.center_lat, radius_m=job.radius_m)
    except Exception as exc:  # noqa: BLE001 - таблиц может не быть, если импорт (Шаг 1.1) не запускался
        raise RuntimeError(
            "нет данных OSM для этой области (проверьте, что выполнен импорт по Шагу 1.1)"
        ) from exc

    key = f"jobs/{job.id}/osm_selection.json"
    storage.upload(key, json.dumps(counts, ensure_ascii=False).encode("utf-8"), content_type="application/json")
    return {"storage_key": key, "counts": counts}


def _wgs84_bbox_for_radius(lon: float, lat: float, radius_m: float, zone: int) -> tuple[float, float, float, float]:
    """Bbox в WGS-84, покрывающий круг `radius_m` вокруг (lon, lat), считая
    через МСК-59 (метры) — точнее, чем наивный градусный отступ, который
    искажается с широтой (переиспользует Шаг 0.4)."""
    x, y, _ = wgs84_to_msk59(lon, lat, zone=zone)
    corners = [(x - radius_m, y - radius_m), (x + radius_m, y - radius_m), (x - radius_m, y + radius_m), (x + radius_m, y + radius_m)]
    lons, lats = [], []
    for cx, cy in corners:
        clon, clat = _msk59_to_wgs84_local(cx, cy, zone)
        lons.append(clon)
        lats.append(clat)
    return (min(lons), min(lats), max(lons), max(lats))


def _msk59_to_wgs84_local(x: float, y: float, zone: int):
    from topology_geo.coords import msk59_to_wgs84

    return msk59_to_wgs84(x, y, zone=zone)


def _array_to_geotiff_bytes(values: np.ndarray, grid: Grid, nodata: float = -9999.0) -> bytes:
    from rasterio.io import MemoryFile

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=grid.height, width=grid.width, count=1, dtype="float64",
            crs=grid.crs, transform=grid.transform, nodata=nodata,
        ) as dst:
            dst.write(values, 1)
        return memfile.read()


def prepare_relief(conn: Any, storage: ObjectStorage, job: store.Job) -> dict:
    """Шаг 2: собрать лучший доступный рельеф по радиусу задачи (Шаг 1.2) и
    сохранить как COG."""
    zone = pick_msk59_zone(job.center_lon)
    bbox = _wgs84_bbox_for_radius(job.center_lon, job.center_lat, job.radius_m, zone)

    half_extent = job.radius_m + RELIEF_MARGIN_M
    cx, cy, _ = wgs84_to_msk59(job.center_lon, job.center_lat, zone=zone)
    width = height = max(int(2 * half_extent / RELIEF_PIXEL_SIZE_M), 2)
    transform = Affine(RELIEF_PIXEL_SIZE_M, 0.0, cx - half_extent, 0.0, -RELIEF_PIXEL_SIZE_M, cy + half_extent)
    grid = Grid(transform=transform, width=width, height=height, crs=MSK59_ZONES[zone].to_proj4())

    try:
        values, coverage = get_dem(conn, storage, bbox, grid)
    except LookupError as exc:
        raise RuntimeError("нет данных рельефа в этой области") from exc

    raw_bytes = _array_to_geotiff_bytes(values, grid)
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "relief_raw.tif"
        dst_path = Path(tmp) / "relief.tif"
        src_path.write_bytes(raw_bytes)
        to_cog(src_path, dst_path)
        cog_bytes = dst_path.read_bytes()

    key = f"jobs/{job.id}/relief.tif"
    storage.upload(key, cog_bytes, content_type="image/tiff")
    return {
        "storage_key": key,
        "coverage_fraction": float(coverage.mean()),
        "width": width,
        "height": height,
        "pixel_size_m": RELIEF_PIXEL_SIZE_M,
    }


def select_and_normalize(conn: Any, storage: ObjectStorage, job: store.Job) -> dict:
    """Шаг 3: выборка, обрезка кругом и нормализация атрибутов данных участка
    (Шаг 1.4) — «чистый набор данных участка, одинаковый для всех дальнейших
    генераторов», сохранён как GeoPackage."""
    try:
        dataset = select_site_data(conn, job.center_lon, job.center_lat, job.radius_m)
    except Exception as exc:  # noqa: BLE001 - таблиц может не быть, если импорт (Шаг 1.1) не запускался
        raise RuntimeError(
            "нет данных OSM для этой области (проверьте, что выполнен импорт по Шагу 1.1)"
        ) from exc

    gpkg_bytes = dataset_to_geopackage_bytes(dataset)

    key = f"jobs/{job.id}/site.gpkg"
    storage.upload(key, gpkg_bytes, content_type="application/geopackage+sqlite3")
    return {
        "storage_key": key,
        "zone": dataset.zone,
        "feature_count": len(dataset.features),
        "layers": {layer: len(features) for layer, features in dataset.by_layer().items()},
    }


def assemble_ifc(conn: Any, storage: ObjectStorage, job: store.Job) -> dict:
    """Шаги 4-5 (Шаги 1.5-1.8): TIN участка, здания/дороги/мосты/вода/рельсы/
    деревья и сборка `site.ifc` в обеих схемах (IFC4, IFC4X3), с реестром
    GlobalId в PostGIS (Шаг 1.8, п. 2). Плюс (Шаг 2.4, п. 4) два
    дополнительных файла на каждую схему — `roads_backbone_*.ifc`/
    `roads_internal_*.ifc`, только дорожная сеть соответствующей
    классификации (`build_site_ifc`, `road_network_filter`), каркасная
    помечена нередактируемой (`Pset_Дорога/Полоса.Редактируемый=false`).

    Ж/д и трамвай (Шаг 2.6, `geometry.rail`): ширина насыпи `rail` теперь
    считается по числу путей и колее (`build_rail_ribbons`), плюс платформы
    (`build_platform_areas`), переезды (`build_level_crossings` — по осям
    уже построенных путей `rail`) и упрощённые опоры контактной сети
    (`place_catenary_poles`). Шпалы/рельсы индивидуальной геометрией в
    ближнем кольце в этом проходе не строятся, см. `docs/rail.md`.

    Электросети (Шаг 2.7, `geometry.power`): опоры/башни по точкам OSM
    (`build_poles`) + расчётная расстановка для линий без реальных опор
    (`place_calculated_poles`); провода — цепная линия между опорами
    (`build_wire_spans`); подстанции/ТП — упрощённый объём
    (`build_substations`); охранная зона по классу напряжения
    (`build_power_safety_zones`). Подробности и упрощения — `docs/power.md`.

    Ограждения/стены и фонари (Шаг 2.8, п. 2, `geometry.landscaping`) —
    оставшиеся из действия «опоры, столбы, бордюры, ограждения, пролёты,
    фонари» параметрические генераторы (опоры/столбы и бордюр — Шаги
    2.6/2.7/2.3). Библиотека элементов (п. 1, каталог + PBR-текстуры Poly
    Haven, п. 4) — `topology_geo.assets`, подробности `docs/asset-library.md`.

    Растительность и благоустройство (Шаг 2.9, `geometry.vegetation`/
    `geometry.landscaping`): высота дерева из `height`, плотность массива
    по типу леса (`leaf_type`), газон (`landuse=grass`) выделен из леса в
    плоскую поверхность без рассеивания деревьев, кустарник (`natural=
    scrub`) — отдельная более низкая/частая расстановка, скамейки
    (`amenity=bench`). Подробности — `docs/vegetation.md`.

    Мосты (Шаг 2.5, п. 1-3, `geometry.bridges.build_bridge_ribbons`) строятся
    ПОСЛЕ `roads`/`rail` — габарит проверяется по их осям (`RoadRibbon.axis`/
    `RailRibbon.axis`).

    Выходные форматы (Шаг 2.10): на каждую IFC-схему — три федеративных
    файла по слоям (п. 1, `relief_*.ifc`/`buildings_*.ifc`/`power_*.ifc`,
    тот же приём, что `roads_backbone/internal`, Шаг 2.4, п. 4), плюс ОДИН
    раз (не зависят от IFC-схемы) — LandXML (`topology_geo.export.landxml`:
    поверхность TIN + оси дорог/путей), CityJSON (`export.cityjson`: здания
    LOD0/LOD1/LOD2, та же логика меша, что и IFC-рендер зданий выше) и DXF
    (`export.dxf`: 2D-контуры зданий/дорог/воды по слоям). Подробности и
    упрощения каждого формата — `docs/output-formats.md`."""
    zone = pick_msk59_zone(job.center_lon)
    center_x, center_y, _ = wgs84_to_msk59(job.center_lon, job.center_lat, zone=zone)

    try:
        dataset = select_site_data(conn, job.center_lon, job.center_lat, job.radius_m)
    except Exception as exc:
        raise RuntimeError(
            "нет данных OSM для этой области (проверьте, что выполнен импорт по Шагу 1.1)"
        ) from exc

    relief_values, relief_grid = read_relief_from_storage(storage, f"jobs/{job.id}/relief.tif")
    tin = build_site_tin(relief_values, relief_grid, center_x, center_y, job.radius_m, dataset.features)

    # Overture Buildings как второй уровень водопада высоты (Шаг 2.2, п. 2) -
    # реального доступа к датасету в этой среде нет (см. NullOvertureSource);
    # подключение реального источника не потребует изменений здесь.
    buildings = extrude_buildings(dataset.features, tin.interpolate_z, NullOvertureSource())
    roads = build_road_ribbons(dataset.features)
    water_areas = build_water_areas(dataset.features, tin.interpolate_z)
    waterways = build_waterway_ribbons(dataset.features, tin.interpolate_z)
    rail = build_rail_ribbons(dataset.features)
    platforms = build_platform_areas(dataset.features)
    level_crossings = build_level_crossings(dataset.features, rail)
    catenary_poles = place_catenary_poles(dataset.features)
    # Растительность (Шаг 2.9): высота дерева из `height` и плотность
    # массива по типу леса теперь внутри `build_individual_trees`/
    # `scatter_forest_trees` самих (см. модуль); газон (`landuse=grass`)
    # выделен из леса в свою плоскую поверхность, кустарник
    # (`natural=scrub`) - отдельная более низкая/частая расстановка.
    trees = build_individual_trees(dataset.features) + scatter_forest_trees(dataset.features)
    shrubs = scatter_shrubs(dataset.features)
    lawns = build_lawns(dataset.features)

    # Электросети с опорами (Шаг 2.7) - реальные опоры/башни по точкам OSM,
    # расчётная расстановка только для линий без единой реальной опоры
    # (`place_calculated_poles`, п. 2); провода строятся по ОБЪЕДИНЁННОМУ
    # списку реальных+расчётных опор (`build_wire_spans` сам находит, какие
    # из них лежат на конкретной линии, по расстоянию до её оси).
    power_real_poles = build_poles(dataset.features)
    power_calculated_poles = place_calculated_poles(dataset.features, power_real_poles)
    power_poles = power_real_poles + power_calculated_poles
    power_wires = build_wire_spans(dataset.features, power_poles, tin.interpolate_z)
    substations = build_substations(dataset.features, tin.interpolate_z)
    power_safety_zones = build_power_safety_zones(dataset.features)

    # Ограждения/стены и фонари (Шаг 2.8, п. 2) - опоры/столбы (рельс/ЛЭП) и
    # бордюр уже параметрические с более ранних шагов.
    fences = build_fences(dataset.features)
    streetlamps = build_streetlamps(dataset.features)
    benches = build_benches(dataset.features)

    # Мосты, путепроводы (Шаг 2.5, п. 1-3) - габарит проверяется по осям уже
    # построенных немостовых дорог/путей (`RoadRibbon.axis`/`RailRibbon.axis`,
    # Шаг 2.4/2.5); дорога-мост сама не входит в `roads` (`is_bridge` в
    # `build_road_ribbons`, `geometry.roads`), поэтому мост не проверяется
    # сам на себя.
    bridges = build_bridge_ribbons(
        dataset.features, tin.interpolate_z,
        crossing_road_axes=[r.axis for r in roads if r.axis is not None],
        crossing_rail_axes=[r.axis for r in rail if r.axis is not None],
    )

    # Полосы через osm2streets (Шаг 2.3, п. 1 и 3) - честный водопад: если
    # инструмента нет в окружении (см. `is_osm2streets_available`), участок
    # остаётся с одной лентой на дорогу (Шаг 1.7, `roads` выше), не падает -
    # тот же приём, что `NullOvertureSource` для водопада высоты (Шаг 2.2).
    street_network = StreetNetwork(lanes=[], intersections=[], markings=[])
    if is_osm2streets_available():
        raw_roads = fetch_raw_roads_in_buffer(conn, job.center_lon, job.center_lat, job.radius_m)
        street_network = build_lane_network(raw_roads, job.center_lon, job.center_lat, job.radius_m, zone)

    site_model = SiteModel(
        tin=tin, buildings=buildings, roads=roads, bridges=bridges,
        water_areas=water_areas, waterways=waterways, rail=rail,
        platforms=platforms, level_crossings=level_crossings, catenary_poles=catenary_poles,
        power_poles=power_poles, power_wires=power_wires, substations=substations,
        power_safety_zones=power_safety_zones,
        fences=fences, streetlamps=streetlamps, benches=benches,
        trees=trees, shrubs=shrubs, lawns=lawns,
        lanes=street_network.lanes, intersections=street_network.intersections, markings=street_network.markings,
    )
    base_point = BasePoint(
        lon=job.center_lon, lat=job.center_lat, zone=zone, x=center_x, y=center_y,
        height=tin.interpolate_z(0.0, 0.0),
    )

    ensure_ifc_registry_schema(conn)
    model_id = str(job.id)
    schemas: dict[str, dict] = {}
    for schema in IFC_SCHEMAS:
        model, registry = build_site_ifc(schema, site_model, base_point)
        issues = validate_model(model)
        if issues:
            raise RuntimeError(f"site.ifc ({schema}) не прошёл ifcopenshell.validate: {issues[:3]!r}")
        register_global_ids(conn, model_id, registry)

        key = f"jobs/{job.id}/site_{schema.lower()}.ifc"
        storage.upload(key, model.to_string().encode("utf-8"), content_type="application/x-step")
        schemas[schema] = {"storage_key": key, "product_count": len(model.by_type("IfcProduct"))}

        # Каркасная и внутриквартальная сеть в отдельных файлах (Шаг 2.4,
        # п. 4: «два независимых файла дорог», каркасная — нередактируемая,
        # `Pset_Дорога/Полоса.Редактируемый`). Реестр GlobalId — под своим
        # namespace `model_id`, не под общим `str(job.id)`: иначе `ON
        # CONFLICT (model_id, layer, osm_id)` в `register_global_ids`
        # затёр бы GlobalId комбинированного `site.ifc` выше своим (та же
        # запись `(layer, osm_id)` встречается в обоих файлах с РАЗНЫМ
        # GlobalId — это два разных IFC-объекта на один исходный OSM-way).
        for network, suffix in ((NETWORK_BACKBONE, "backbone"), (NETWORK_INTERNAL, "internal")):
            network_model, network_registry = build_site_ifc(
                schema, site_model, base_point, road_network_filter=network
            )
            issues = validate_model(network_model)
            if issues:
                raise RuntimeError(
                    f"roads_{suffix} ({schema}) не прошёл ifcopenshell.validate: {issues[:3]!r}"
                )
            register_global_ids(conn, f"{model_id}:roads_{suffix}", network_registry)

            network_key = f"jobs/{job.id}/roads_{suffix}_{schema.lower()}.ifc"
            storage.upload(
                network_key, network_model.to_string().encode("utf-8"), content_type="application/x-step"
            )
            schemas[schema][f"roads_{suffix}_storage_key"] = network_key

        # Федеративные файлы по слоям (Шаг 2.10, п. 1: «набор IFC-файлов по
        # слоям (federated model): relief.ifc, buildings.ifc, power.ifc...»).
        # Каждый — независимый `SiteModel` только со своим слоем (те же уже
        # посчитанные объекты, что и в комбинированном `site.ifc` выше, без
        # пересборки геометрии), тот же приём, что и у `roads_backbone/
        # internal` (свой namespace реестра GlobalId, иначе `ON CONFLICT`
        # затёр бы GlobalId комбинированного файла). У `buildings`/`power`
        # НЕТ своего TIN (не дублируем самый тяжёлый по памяти меш, та же
        # причина, что и у `roads_backbone/internal`) — их объекты несут
        # свою абсолютную высоту (`base_z`/расчётная опора уже над рельефом
        # в комбинированном `site.ifc`), кроме `power`: без TIN его точки
        # интерполяции рельефа падают на плоскость z=0 (см.
        # `_terrain_elevation` в `ifc/assemble.py`) — то есть провода/опоры в
        # ЭТОМ отдельном файле привязаны к z=0, не к реальному рельефу
        # (известное упрощение, только для `power.ifc`; в комбинированном
        # `site.ifc` рельеф есть и высоты верны).
        layer_models = {
            "relief": SiteModel(tin=tin),
            "buildings": SiteModel(buildings=buildings),
            "power": SiteModel(
                power_poles=power_poles, power_wires=power_wires,
                substations=substations, power_safety_zones=power_safety_zones,
            ),
        }
        for layer_name, layer_model in layer_models.items():
            layer_model_ifc, layer_registry = build_site_ifc(schema, layer_model, base_point)
            issues = validate_model(layer_model_ifc)
            if issues:
                raise RuntimeError(f"{layer_name} ({schema}) не прошёл ifcopenshell.validate: {issues[:3]!r}")
            register_global_ids(conn, f"{model_id}:{layer_name}", layer_registry)

            layer_key = f"jobs/{job.id}/{layer_name}_{schema.lower()}.ifc"
            storage.upload(
                layer_key, layer_model_ifc.to_string().encode("utf-8"), content_type="application/x-step"
            )
            schemas[schema][f"{layer_name}_storage_key"] = layer_key

    # Форматы, не зависящие от IFC-схемы (Шаг 2.10, п. 2) — строятся ОДИН
    # раз (не в цикле по `IFC_SCHEMAS` выше), из уже готовых в памяти
    # структур (`tin`, `roads`, `rail`, `buildings`, `water_areas`,
    # `waterways`) — та же экономия, что и у федеративных IFC-слоёв: не
    # пересчитывать TIN/геометрию по новой.
    landxml_bytes = build_landxml(tin, roads=roads, rail=rail)
    landxml_key = f"jobs/{job.id}/site.landxml"
    storage.upload(landxml_key, landxml_bytes, content_type="application/xml")

    cityjson_doc = build_cityjson(buildings)
    cityjson_key = f"jobs/{job.id}/site.cityjson"
    storage.upload(
        cityjson_key, json.dumps(cityjson_doc, ensure_ascii=False).encode("utf-8"), content_type="application/json"
    )

    dxf_bytes = build_dxf(buildings=buildings, roads=roads, water_areas=water_areas, waterways=waterways)
    dxf_key = f"jobs/{job.id}/site.dxf"
    storage.upload(dxf_key, dxf_bytes, content_type="application/dxf")

    return {
        "schemas": schemas,
        "landxml_storage_key": landxml_key,
        "cityjson_storage_key": cityjson_key,
        "dxf_storage_key": dxf_key,
        # Сохраняем базовую точку здесь, чтобы `package_outputs` (Шаг 2.10,
        # п. 3) не пересчитывал высоту через TIN заново - интерполяция
        # требует загрузки/перестройки TIN, самой тяжёлой по памяти части
        # сборки (см. комментарий у федеративных IFC-слоёв выше).
        "base_point": {
            "x": base_point.x, "y": base_point.y, "height": base_point.height, "zone": base_point.zone,
        },
        "tin_vertices": int(tin.vertices.shape[0]),
        "buildings": len(buildings),
        "roads": len(roads),
        "bridges": len(bridges),
        "water_areas": len(water_areas),
        "waterways": len(waterways),
        "rail": len(rail),
        "platforms": len(platforms),
        "level_crossings": len(level_crossings),
        "catenary_poles": len(catenary_poles),
        "power_poles": len(power_poles),
        "power_wires": len(power_wires),
        "substations": len(substations),
        "power_safety_zones": len(power_safety_zones),
        "fences": len(fences),
        "streetlamps": len(streetlamps),
        "benches": len(benches),
        "trees": len(trees),
        "shrubs": len(shrubs),
        "lawns": len(lawns),
        "lanes": len(street_network.lanes),
        "intersections": len(street_network.intersections),
        "markings": len(street_network.markings),
    }


def convert_to_glb(conn: Any, storage: ObjectStorage, job: store.Job) -> dict:
    """Шаг 1.9, п. 1: IFC -> GLB для веб-просмотра. Источник — уже собранный
    `site_ifc4x3.ifc` (Шаг 1.8, схема с нативными классами) из хранилища, не
    пересборка из геометрии заново — GLB остаётся производным от IFC."""
    ifc_key = f"jobs/{job.id}/site_ifc4x3.ifc"
    ifc_text = storage.download(ifc_key).decode("utf-8")
    model = ifcopenshell.file.from_string(ifc_text)

    glb_bytes = convert_ifc_to_glb(model)
    key = f"jobs/{job.id}/site.glb"
    storage.upload(key, glb_bytes, content_type="model/gltf-binary")
    return {"storage_key": key, "size_bytes": len(glb_bytes)}


def generate_tileset(conn: Any, storage: ObjectStorage, job: store.Job) -> dict:
    """Шаг 2.11, п. 1: 3D Tiles (`tileset.json` + GLB по тайлам 250×250 м,
    Шаг 2.1) с иерархией детализации по кольцам LOD (Шаг 2.1, п. 1:
    `classify_lod_ring`) — для потокового вьюера (п. 2-3, `docs/streaming-
    viewer.md`): рельеф тайла грубее с расстоянием (`tiling.tileset.
    RING_BACKGROUND_STEP_M`), здания — без крыши/блоком/с крышей по кольцу
    (та же меш-логика, что CityJSON, Шаг 2.10, `tiling.tile_content`).

    Лёгкий шаг: НЕ строит общий TIN участка (самая тяжёлая по памяти часть
    `assemble_ifc`) — рельеф тайла берётся напрямую из уже сохранённого
    растра (`prepare_relief`, тот же приём, что параллельный генератор
    тайлов Шага 2.1, `tasks.tile_tasks`), высота площадки здания — билинейно
    прямо из растра (`relief.tin.sample_bilinear`), не через `SiteTin.
    interpolate_z`.

    Честно не входит в этот проход: дороги/вода/рельсы/растительность/ЛЭП в
    тайлах (только рельеф и здания); выгрузка тайлов за пределы актуальных в
    кэше (Шаг 2.1) не переиспользуется — каждый вызов строит тайлы заново
    (кэш `tiling.cache` используется только параллельным генератором тайлов
    рельефа Шага 2.1, здесь не подключён, см. `docs/streaming-viewer.md`)."""
    zone = pick_msk59_zone(job.center_lon)
    center_x, center_y, _ = wgs84_to_msk59(job.center_lon, job.center_lat, zone=zone)

    try:
        dataset = select_site_data(conn, job.center_lon, job.center_lat, job.radius_m)
    except Exception as exc:
        raise RuntimeError(
            "нет данных OSM для этой области (проверьте, что выполнен импорт по Шагу 1.1)"
        ) from exc

    relief_values, relief_grid = read_relief_from_storage(storage, f"jobs/{job.id}/relief.tif")

    buildings = extrude_buildings(
        dataset.features,
        lambda x, y: sample_bilinear(relief_values, relief_grid, center_x + x, center_y + y),
        NullOvertureSource(),
    )

    buildings_by_tile: dict[TileIndex, list] = {}
    for building in buildings:
        centroid = building.footprint.centroid
        tile = world_to_tile_index(zone, center_x + centroid.x, center_y + centroid.y)
        buildings_by_tile.setdefault(tile, []).append(building)

    entries: list[TileContentEntry] = []
    tile_counts = {LOD2: 0, LOD1: 0, LOD0: 0}
    for tile in tiles_covering_circle(zone, center_x, center_y, job.radius_m):
        minx, miny, maxx, maxy = tile.bounds()
        tile_center_x, tile_center_y = (minx + maxx) / 2, (miny + maxy) / 2
        distance = math.hypot(tile_center_x - center_x, tile_center_y - center_y)
        if distance > LOD0_MAX_M:
            continue  # угол тайла зацепил круг задачи, но центр тайла уже вне зоны Этапа 2
        lod = classify_lod_ring(distance)

        content = build_tile_content(
            relief_values, relief_grid, tile, lod, buildings_by_tile.get(tile, []),
            center_x=center_x, center_y=center_y,
        )
        if content is None:
            continue

        tile_key = f"jobs/{job.id}/tiles/{lod}/{tile.zone}_{tile.tx}_{tile.ty}.glb"
        storage.upload(tile_key, content.glb_bytes, content_type="model/gltf-binary")
        entries.append(
            TileContentEntry(
                tile=tile, lod=lod, storage_key=tile_key,
                local_minx=minx - center_x, local_miny=miny - center_y,
                local_maxx=maxx - center_x, local_maxy=maxy - center_y,
                z_min=content.z_min, z_max=content.z_max,
            )
        )
        tile_counts[lod] += 1

    tileset_doc = build_tileset_json(entries)
    tileset_key = f"jobs/{job.id}/tileset.json"
    storage.upload(
        tileset_key, json.dumps(tileset_doc, ensure_ascii=False).encode("utf-8"), content_type="application/json"
    )

    return {"tileset_storage_key": tileset_key, "tile_count": len(entries), "tiles_by_lod": tile_counts}


# Слои пакета (Шаг 2.10, п. 3, `layers` в `meta.json`) - "networks"/
# "constraints" (Этап 3, Шаги 3.1/3.4) сюда не входят, их ещё нет.
# "status" - ни один слой не официальные регуляторные данные (те появятся
# только в Этапе 3), у всех статус "расчётно" (посчитаны из тегов OSM +
# эвристик, не заявлены как официальный документ) - честно, не завышаем.
_PACKAGE_LAYERS_META: list[dict[str, str]] = [
    {"name": "relief", "lod": "LOD1", "status": "расчётно"},
    {"name": "roads_backbone", "lod": "LOD1", "status": "расчётно"},
    {"name": "roads_local", "lod": "LOD1", "status": "расчётно"},
    {"name": "buildings", "lod": "LOD2", "status": "расчётно"},
    {"name": "power", "lod": "LOD1", "status": "расчётно"},
    {"name": "vegetation", "lod": "LOD1", "status": "расчётно"},
    {"name": "site", "lod": "LOD1", "status": "расчётно"},
]


def _collect_sources(conn: Any, job: store.Job, zone: int) -> list[dict[str, str]]:
    """Журнал источников (Шаг 2.10, п. 3, `sources` в `meta.json`) — реальные
    записи `osm_import_log` (Шаг 1.1, п. 4) и `dem_coverage` (Шаг 1.2, п. 4,
    покрытия, пересекающие bbox задачи), не выдуманные значения.

    Лицензия OSM (ODbL) — известная константа для ЛЮБОЙ записи этого
    журнала (он ведётся только для OSM-загрузок, см. `import_log.py`).
    Лицензия рельефа НЕ хранится в `dem_coverage` (поле отсутствует в схеме
    таблицы, Шаг 1.2) — честно помечена как не зафиксированная, а не
    придумана (см. `docs/output-formats.md`)."""
    # На случай, если ни один импорт/покрытие не были зарегистрированы через
    # CLI (`osm/cli.py`) в этом окружении — таблиц может не быть вовсе,
    # `ensure_schema` идемпотентен (тот же приём, что и `ensure_ifc_registry_schema` выше).
    ensure_import_log_schema(conn)
    ensure_dem_coverage_schema(conn)

    sources: list[dict[str, str]] = []
    for entry in all_latest_sources(conn):
        sources.append(
            {
                "name": entry.source_name,
                "retrieved_at": entry.data_timestamp.date().isoformat(),
                "license": "ODbL",
            }
        )

    bbox = _wgs84_bbox_for_radius(job.center_lon, job.center_lat, job.radius_m, zone)
    seen_relief_sources: set[str] = set()
    for coverage in find_coverage(conn, bbox):
        if coverage.source_name in seen_relief_sources:
            continue
        seen_relief_sources.add(coverage.source_name)
        sources.append(
            {
                "name": coverage.source_name,
                "retrieved_at": coverage.data_timestamp.date().isoformat(),
                "license": "не зафиксирована в реестре покрытия (dem_coverage, Шаг 1.2, п. 4)",
            }
        )
    return sources


def package_outputs(conn: Any, storage: ObjectStorage, job: store.Job) -> dict:
    """Шаг 2.10, п. 3: `meta.json` (схема `schemas/meta.schema.json`, Шаг
    0.8) + zip-архив всех файлов задачи. Последний, лёгкий шаг конвейера —
    только скачивает уже загруженные в Storage байты предыдущих шагов и
    упаковывает их, БЕЗ пересборки TIN/геометрии (та же экономия памяти,
    что и у федеративных IFC-слоёв в `assemble_ifc`).

    Имя `roads_internal` (внутренний ключ Storage с Шага 2.4) здесь
    переименовывается в архиве в `roads_local` — так называет итоговый файл
    `docs/data-dictionary.md` §4, отдельно от промежуточного имени шага
    конвейера, как и предусмотрено в этой таблице (`roads_backbone.ifc`/
    `roads_local.ifc`)."""
    zone = pick_msk59_zone(job.center_lon)
    steps_by_name = {s.step_name: s for s in job.steps}
    normalize_result = steps_by_name["select_and_normalize"].result or {}
    assemble_result = steps_by_name["assemble_ifc"].result or {}
    glb_result = steps_by_name["convert_to_glb"].result or {}
    base_point = assemble_result["base_point"]

    # (архивный путь, storage_key, format, layer) - format/layer как в
    # `schemas/meta.schema.json` (`files[].format` enum, `files[].layer` -
    # свободная строка, не обязана совпадать с `layers[].name`).
    entries: list[tuple[str, str, str, str]] = [
        ("site.gpkg", normalize_result["storage_key"], "GeoPackage", "site"),
    ]
    for schema in IFC_SCHEMAS:
        schema_result = assemble_result["schemas"][schema]
        suffix = schema.lower()
        entries.append((f"site_{suffix}.ifc", schema_result["storage_key"], schema, "site"))
        entries.append(
            (f"roads_backbone_{suffix}.ifc", schema_result["roads_backbone_storage_key"], schema, "roads_backbone")
        )
        entries.append(
            (f"roads_local_{suffix}.ifc", schema_result["roads_internal_storage_key"], schema, "roads_local")
        )
        entries.append((f"relief_{suffix}.ifc", schema_result["relief_storage_key"], schema, "relief"))
        entries.append((f"buildings_{suffix}.ifc", schema_result["buildings_storage_key"], schema, "buildings"))
        entries.append((f"power_{suffix}.ifc", schema_result["power_storage_key"], schema, "power"))
    entries.append(("site.landxml", assemble_result["landxml_storage_key"], "LandXML", "рельеф+дороги"))
    entries.append(("site.cityjson", assemble_result["cityjson_storage_key"], "CityJSON", "buildings"))
    entries.append(("site.dxf", assemble_result["dxf_storage_key"], "DXF", "здания+дороги+вода"))
    entries.append(("site.glb", glb_result["storage_key"], "GLB", "site"))

    files_meta: list[dict[str, Any]] = []
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for archive_path, storage_key, fmt, layer in entries:
            data = storage.download(storage_key)
            zf.writestr(archive_path, data)
            files_meta.append({"path": archive_path, "format": fmt, "layer": layer, "size_bytes": len(data)})

        meta = {
            "task_id": str(job.id),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator_version": __version__,
            "center": {"lon": job.center_lon, "lat": job.center_lat},
            "radius_m": job.radius_m,
            "coordinate_system": {
                "projected_crs": "MSK-59",
                "zone": zone,
                "height_system": "Балтийская",
                "base_point": {"x": base_point["x"], "y": base_point["y"], "height": base_point["height"]},
            },
            "layers": _PACKAGE_LAYERS_META,
            "files": files_meta,
            "sources": _collect_sources(conn, job, zone),
        }
        meta_bytes = json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
        zf.writestr("meta.json", meta_bytes)

    meta_key = f"jobs/{job.id}/meta.json"
    storage.upload(meta_key, meta_bytes, content_type="application/json")

    archive_key = f"jobs/{job.id}/package.zip"
    storage.upload(archive_key, zip_buffer.getvalue(), content_type="application/zip")

    return {"meta_storage_key": meta_key, "archive_storage_key": archive_key, "file_count": len(files_meta) + 1}


def fetch_cadastre(conn: Any, storage: ObjectStorage, job: store.Job) -> dict:
    """Шаг: загрузить кадастровые границы участков из НСПД (Росреестр API).

    По координатам центра задачи ищет участки в НСПД, получает границы в GeoJSON.
    Результаты сохраняются в БД и в JSON-файл в Storage для последующего
    использования в вьюере и проверках (Шаги 3.3-3.9).
    """
    logger = __import__("logging").getLogger("fetch_cadastre")

    try:
        # Поиск участков в НСПД по координатам
        cadastre_list = search_cadastre_by_coords(
            lon=job.center_lon, lat=job.center_lat, radius_m=job.radius_m
        )

        if not cadastre_list:
            logger.warning(f"Нет участков в НСПД для {job.center_lon}, {job.center_lat}")
            return {"cadastre_found": 0, "error": "Участки не найдены"}

        # Сохранить в БД
        cursor = conn.cursor()
        for cadastre in cadastre_list:
            try:
                cursor.execute(
                    """
                    INSERT INTO cadastre_data (job_id, cadastre_number, center_lon, center_lat,
                                              area_m2, owner, address, boundary_geojson)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (job_id, cadastre_number) DO UPDATE SET
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        job.id,
                        cadastre.cadastre_number,
                        cadastre.center_lon,
                        cadastre.center_lat,
                        cadastre.area_m2,
                        cadastre.owner,
                        cadastre.address,
                        json.dumps(cadastre.boundary_geojson) if cadastre.boundary_geojson else None,
                    ),
                )
                logger.info(f"Сохранён участок {cadastre.cadastre_number}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении {cadastre.cadastre_number}: {e}")

        conn.commit()

        # Сохранить GeoJSON в Storage для вьюера
        geojson_features = []
        for cadastre in cadastre_list:
            if cadastre.boundary_geojson:
                feature = cadastre.boundary_geojson.copy()
                feature["properties"] = {
                    "cadastre_number": cadastre.cadastre_number,
                    "area_m2": cadastre.area_m2,
                    "owner": cadastre.owner,
                    "address": cadastre.address,
                }
                geojson_features.append(feature)

        if geojson_features:
            geojson_collection = {"type": "FeatureCollection", "features": geojson_features}
            key = f"jobs/{job.id}/cadastre_boundaries.geojson"
            storage.upload(
                key,
                json.dumps(geojson_collection, ensure_ascii=False).encode("utf-8"),
                content_type="application/geo+json",
            )
            logger.info(f"Сохранён GeoJSON: {key}")

        return {
            "cadastre_found": len(cadastre_list),
            "cadastre_numbers": [c.cadastre_number for c in cadastre_list],
            "geojson_key": f"jobs/{job.id}/cadastre_boundaries.geojson" if geojson_features else None,
        }

    except Exception as e:
        logger.error(f"Ошибка при загрузке кадастра: {e}")
        return {"cadastre_found": 0, "error": str(e)}


DEFAULT_PIPELINE: dict[str, Any] = {
    "select_osm": select_osm,
    "fetch_cadastre": fetch_cadastre,
    "prepare_relief": prepare_relief,
    "select_and_normalize": select_and_normalize,
    "assemble_ifc": assemble_ifc,
    "convert_to_glb": convert_to_glb,
    "generate_tileset": generate_tileset,
    "package_outputs": package_outputs,
}

DEFAULT_STEP_NAMES: list[str] = list(DEFAULT_PIPELINE)
