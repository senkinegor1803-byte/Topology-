"""Реализация шагов пайплайна задачи (Шаг 1.3).

`select_osm`, `prepare_relief`, `select_and_normalize`, `assemble_ifc` и
`convert_to_glb` — реальные шаги, использующие уже реализованные Шаги
1.1, 1.2, 1.4-1.9.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import ifcopenshell
import numpy as np
from affine import Affine

from topology_geo.coords import MSK59_ZONES, pick_msk59_zone, wgs84_to_msk59
from topology_geo.geometry.buildings import NullOvertureSource, extrude_buildings
from topology_geo.geometry.rail import build_rail_ribbons
from topology_geo.geometry.roads import build_road_ribbons
from topology_geo.geometry.streets import StreetNetwork, build_lane_network, is_osm2streets_available
from topology_geo.geometry.vegetation import build_individual_trees, scatter_forest_trees
from topology_geo.geometry.water import build_water_areas, build_waterway_ribbons
from topology_geo.ifc.assemble import BasePoint, SiteModel, build_site_ifc
from topology_geo.ifc.generate_test_ifc import validate_model
from topology_geo.ifc.registry import ensure_schema as ensure_ifc_registry_schema
from topology_geo.ifc.registry import register_global_ids
from topology_geo.ifc.to_glb import convert_ifc_to_glb
from topology_geo.jobs import store
from topology_geo.osm.queries import count_within_radius
from topology_geo.osm.raw_roads import fetch_raw_roads_in_buffer
from topology_geo.relief.cog import to_cog
from topology_geo.relief.service import Grid, get_dem, read_relief_from_storage
from topology_geo.relief.tin import build_site_tin
from topology_geo.selection.geopackage import dataset_to_geopackage_bytes
from topology_geo.selection.service import select_site_data
from topology_geo.storage import ObjectStorage

RELIEF_PIXEL_SIZE_M = 10.0
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
    """Шаги 4-5 (Шаги 1.5-1.8): TIN участка, здания/дороги/вода/рельсы/деревья
    и сборка `site.ifc` в обеих схемах (IFC4, IFC4X3), с реестром GlobalId в
    PostGIS (Шаг 1.8, п. 2)."""
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
    waterways = build_waterway_ribbons(dataset.features)
    rail = build_rail_ribbons(dataset.features)
    trees = build_individual_trees(dataset.features) + scatter_forest_trees(dataset.features)

    # Полосы через osm2streets (Шаг 2.3, п. 1 и 3) - честный водопад: если
    # инструмента нет в окружении (см. `is_osm2streets_available`), участок
    # остаётся с одной лентой на дорогу (Шаг 1.7, `roads` выше), не падает -
    # тот же приём, что `NullOvertureSource` для водопада высоты (Шаг 2.2).
    street_network = StreetNetwork(lanes=[], intersections=[], markings=[])
    if is_osm2streets_available():
        raw_roads = fetch_raw_roads_in_buffer(conn, job.center_lon, job.center_lat, job.radius_m)
        street_network = build_lane_network(raw_roads, job.center_lon, job.center_lat, job.radius_m, zone)

    site_model = SiteModel(
        tin=tin, buildings=buildings, roads=roads,
        water_areas=water_areas, waterways=waterways, rail=rail, trees=trees,
        lanes=street_network.lanes, intersections=street_network.intersections, markings=street_network.markings,
    )
    base_point = BasePoint(
        lon=job.center_lon, lat=job.center_lat, zone=zone, x=center_x, y=center_y,
        height=tin.interpolate_z(0.0, 0.0) or 0.0,
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

    return {
        "schemas": schemas,
        "tin_vertices": int(tin.vertices.shape[0]),
        "buildings": len(buildings),
        "roads": len(roads),
        "water_areas": len(water_areas),
        "waterways": len(waterways),
        "rail": len(rail),
        "trees": len(trees),
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


DEFAULT_PIPELINE: dict[str, Any] = {
    "select_osm": select_osm,
    "prepare_relief": prepare_relief,
    "select_and_normalize": select_and_normalize,
    "assemble_ifc": assemble_ifc,
    "convert_to_glb": convert_to_glb,
}

DEFAULT_STEP_NAMES: list[str] = list(DEFAULT_PIPELINE)
