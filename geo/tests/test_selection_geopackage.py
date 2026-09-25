"""Тесты сохранения набора данных участка в GeoPackage (Шаг 1.4, п. 4). Чистая
геометрия/атрибуты — Postgres не требуется."""

from __future__ import annotations

import io

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from topology_geo.selection.geopackage import dataset_to_geopackage_bytes
from topology_geo.selection.service import SiteDataset, SiteFeature


def _sample_dataset() -> SiteDataset:
    building = SiteFeature(
        layer="osm_buildings", osm_id=100, osm_type="W",
        geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        attributes={"type": "yes", "levels": 5},
        confidence={"type": "факт", "levels": "факт"},
    )
    tree = SiteFeature(
        layer="osm_vegetation", osm_id=50, osm_type="N",
        geometry=Point(20, 20),
        attributes={"species": None},
        confidence={"species": "умолчание"},
    )
    return SiteDataset(center_lon=56.24, center_lat=58.01, radius_m=500, zone=2, features=[building, tree])


def _read_layer(gpkg_bytes: bytes, layer: str) -> gpd.GeoDataFrame:
    with io.BytesIO(gpkg_bytes) as buf:
        return gpd.read_file(buf, layer=layer)


def test_geopackage_contains_one_layer_per_object_type():
    gpkg_bytes = dataset_to_geopackage_bytes(_sample_dataset())
    buildings = _read_layer(gpkg_bytes, "osm_buildings")
    vegetation = _read_layer(gpkg_bytes, "osm_vegetation")

    assert len(buildings) == 1
    assert len(vegetation) == 1


def test_geopackage_preserves_attributes_and_confidence():
    gpkg_bytes = dataset_to_geopackage_bytes(_sample_dataset())
    buildings = _read_layer(gpkg_bytes, "osm_buildings")

    row = buildings.iloc[0]
    assert row["type"] == "yes"
    assert row["levels"] == 5
    assert row["type_confidence"] == "факт"
    assert row["levels_confidence"] == "факт"
    assert row["osm_id"] == 100


def test_geopackage_geometry_matches_local_coordinates():
    gpkg_bytes = dataset_to_geopackage_bytes(_sample_dataset())
    buildings = _read_layer(gpkg_bytes, "osm_buildings")
    assert buildings.iloc[0].geometry.bounds == (0, 0, 10, 10)


def test_geopackage_raises_for_empty_dataset():
    empty = SiteDataset(center_lon=56.24, center_lat=58.01, radius_m=500, zone=2, features=[])
    with pytest.raises(ValueError):
        dataset_to_geopackage_bytes(empty)


def test_geopackage_skips_layers_with_no_features():
    dataset = SiteDataset(
        center_lon=56.24, center_lat=58.01, radius_m=500, zone=2,
        features=[
            SiteFeature(
                layer="osm_buildings", osm_id=1, osm_type="W",
                geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                attributes={}, confidence={},
            )
        ],
    )
    gpkg_bytes = dataset_to_geopackage_bytes(dataset)
    import pyogrio
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.gpkg"
        path.write_bytes(gpkg_bytes)
        layers = {name for name, _ in pyogrio.list_layers(path)}
    assert layers == {"osm_buildings"}
