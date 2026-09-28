"""Тесты Шага 2.10, п. 2: экспорт CityJSON (здания LOD1/LOD2)."""

from __future__ import annotations

from shapely.geometry import box

from topology_geo.export.cityjson import CITYJSON_VERSION, build_cityjson
from topology_geo.geometry.buildings import CONFIDENCE_FACT, SOURCE_OSM
from topology_geo.geometry.buildings import BuildingSolid
from topology_geo.geometry.roofs import ROOF_FLAT, ROOF_GABLED


def _flat_building(osm_id=1) -> BuildingSolid:
    return BuildingSolid(
        osm_id=osm_id, footprint=box(0, 0, 10, 6), height_m=9.0,
        height_confidence=CONFIDENCE_FACT, height_source=SOURCE_OSM, base_z=100.0,
        building_type="жилой", roof_shape=ROOF_FLAT,
    )


def _gabled_building(osm_id=2) -> BuildingSolid:
    return BuildingSolid(
        osm_id=osm_id, footprint=box(20, 0, 30, 8), height_m=9.0,
        height_confidence=CONFIDENCE_FACT, height_source=SOURCE_OSM, base_z=100.0,
        building_type="жилой", roof_shape=ROOF_GABLED, roof_height_m=2.5,
        roof_height_confidence=CONFIDENCE_FACT, roof_ridge_along_long_axis=True,
    )


def test_build_cityjson_top_level_structure():
    doc = build_cityjson([_flat_building()])
    assert doc["type"] == "CityJSON"
    assert doc["version"] == CITYJSON_VERSION
    assert doc["transform"] == {"scale": [1.0, 1.0, 1.0], "translate": [0.0, 0.0, 0.0]}
    assert "building-1" in doc["CityObjects"]


def test_build_cityjson_empty_buildings_list():
    doc = build_cityjson([])
    assert doc["CityObjects"] == {}
    assert doc["vertices"] == []
    assert "metadata" not in doc


def test_build_cityjson_building_has_lod0_lod1_and_lod2_geometry():
    doc = build_cityjson([_flat_building()])
    geoms = {g["lod"]: g for g in doc["CityObjects"]["building-1"]["geometry"]}
    assert set(geoms) == {"0", "1", "2"}
    assert geoms["0"]["type"] == "MultiSurface"
    assert geoms["1"]["type"] == "Solid"
    assert geoms["2"]["type"] == "Solid"


def test_build_cityjson_lod0_is_flat_footprint_ring():
    doc = build_cityjson([_flat_building()])
    lod0 = next(g for g in doc["CityObjects"]["building-1"]["geometry"] if g["lod"] == "0")
    surfaces = lod0["boundaries"]
    assert len(surfaces) == 1  # один полигон (без внутренних отверстий)
    (outer_ring,) = surfaces[0]
    assert len(outer_ring) == 4  # box(0,0,10,6) - 4 угла, без замыкающей точки
    zs = {doc["vertices"][i][2] for i in outer_ring}
    assert zs == {100.0}  # base_z, без экструзии


def test_build_cityjson_flat_roof_lod1_and_lod2_have_equal_face_count():
    doc = build_cityjson([_flat_building()])
    geoms = {g["lod"]: g for g in doc["CityObjects"]["building-1"]["geometry"]}
    faces_lod1 = geoms["1"]["boundaries"][0]
    faces_lod2 = geoms["2"]["boundaries"][0]
    assert len(faces_lod1) == len(faces_lod2)


def test_build_cityjson_gabled_roof_lod2_has_more_faces_than_lod1():
    doc = build_cityjson([_gabled_building()])
    geoms = {g["lod"]: g for g in doc["CityObjects"]["building-2"]["geometry"]}
    faces_lod1 = geoms["1"]["boundaries"][0]
    faces_lod2 = geoms["2"]["boundaries"][0]
    assert len(faces_lod2) > len(faces_lod1)  # скат добавляет грани сверх призмы


def test_build_cityjson_vertex_indices_are_valid_and_disjoint_per_building():
    doc = build_cityjson([_flat_building(1), _gabled_building(2)])
    n_vertices = len(doc["vertices"])
    all_indices = set()
    for co in doc["CityObjects"].values():
        for geom in co["geometry"]:
            # Solid - на уровень глубже (обёрнут оболочкой), чем MultiSurface
            # (LOD0) - см. докстринг `_lod0_footprint_boundaries`.
            shells = geom["boundaries"] if geom["type"] == "Solid" else [geom["boundaries"]]
            for shell in shells:
                for surface in shell:
                    for ring in surface:
                        for idx in ring:
                            assert 0 <= idx < n_vertices
                            all_indices.add(idx)
    assert len(all_indices) > 0


def test_build_cityjson_attributes_include_height_and_type():
    doc = build_cityjson([_flat_building()])
    attrs = doc["CityObjects"]["building-1"]["attributes"]
    assert attrs["высота_м"] == 9.0
    assert attrs["тип"] == "жилой"
    assert attrs["форма_крыши"] == ROOF_FLAT


def test_build_cityjson_geographical_extent_matches_vertex_bbox():
    doc = build_cityjson([_flat_building()])
    extent = doc["metadata"]["geographicalExtent"]
    xs = [v[0] for v in doc["vertices"]]
    ys = [v[1] for v in doc["vertices"]]
    zs = [v[2] for v in doc["vertices"]]
    assert extent == [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)]


def test_build_cityjson_is_json_serializable():
    import json

    doc = build_cityjson([_flat_building(), _gabled_building()])
    json.dumps(doc)  # не должно бросить исключение
