"""Сохранение нормализованного набора данных участка в GeoPackage (Шаг 1.4, п. 4).

Геометрия — в локальных координатах участка (центр = (0,0), метры), без
определённой CRS: как и у геопривязки IFC (`IfcMapConversion`, Шаг 0.2),
реальная точка отсчёта (`center_lon`/`center_lat`/`zone`) хранится отдельно
от самого файла — в `SiteDataset`/результате шага пайплайна, а не в
самом .gpkg, поэтому не подставляем сюда произвольную CRS.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd

from topology_geo.selection.service import SiteDataset


def dataset_to_geopackage_bytes(dataset: SiteDataset) -> bytes:
    """Записать `dataset` в GeoPackage (один слой на тип объекта) и вернуть
    содержимое файла как байты. Слои без объектов не создаются."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "site.gpkg"
        for layer_name, features in dataset.by_layer().items():
            if not features:
                continue
            records = []
            for feature in features:
                record: dict = dict(feature.attributes)
                record["osm_id"] = feature.osm_id
                record["osm_type"] = feature.osm_type
                for attr_name, confidence in feature.confidence.items():
                    record[f"{attr_name}_confidence"] = confidence
                record["geometry"] = feature.geometry
                records.append(record)

            gdf = gpd.GeoDataFrame(records, geometry="geometry")
            gdf.to_file(path, layer=layer_name, driver="GPKG")

        if not path.exists():
            raise ValueError("в наборе данных участка нет ни одного объекта — нечего сохранять")
        return path.read_bytes()
