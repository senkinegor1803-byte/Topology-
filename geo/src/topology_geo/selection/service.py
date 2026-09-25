"""Оркестрация выборки и нормализации данных участка (Шаг 1.4).

Результат — «чистый набор данных участка, одинаковый для всех дальнейших
генераторов» (дословно из плана): нормализованные признаки по слоям, в
локальных координатах участка. Дальнейшие генераторы (здания — Шаг 1.6,
дороги — Шаг 1.7, сборка IFC — Шаг 1.8) будут читать именно этот набор, а не
резолвить теги OSM заново на каждом шаге.

Порядок обработки одного объекта: выборка в буфере (`query.py`, WGS-84) →
репроекция в МСК-59 (`coords.transform_geometry_to_msk59`) → обрезка кругом
радиуса R и перевод в локальные координаты (`clip.py`) → нормализация тегов
(`normalize.py`). Объекты, не попавшие в круг после обрезки, отбрасываются.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from topology_geo.coords import pick_msk59_zone, transform_geometry_to_msk59, wgs84_to_msk59
from topology_geo.selection.clip import clip_and_localize
from topology_geo.selection.normalize import NORMALIZERS
from topology_geo.selection.query import fetch_features_in_buffer


class _Connection(Protocol):
    def cursor(self) -> Any: ...


@dataclass(frozen=True)
class SiteFeature:
    layer: str
    osm_id: int
    osm_type: str
    geometry: Any  # shapely, локальные координаты участка (центр = (0,0))
    attributes: dict[str, Any]
    confidence: dict[str, str]
    # Исходные теги OSM — не всё нужное нормализует Шаг 1.4 (он приводит
    # только тип/этажность/покрытие/напряжение, дословно из плана); более
    # поздним шагам (1.6 — height, 1.7 — waterway/railway и т.д.) нужен доступ
    # к остальным тегам без повторного похода в PostGIS.
    raw_tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SiteDataset:
    center_lon: float
    center_lat: float
    radius_m: float
    zone: int
    features: list[SiteFeature] = field(default_factory=list)

    def by_layer(self) -> dict[str, list[SiteFeature]]:
        result: dict[str, list[SiteFeature]] = {}
        for feature in self.features:
            result.setdefault(feature.layer, []).append(feature)
        return result


def select_site_data(
    conn: _Connection,
    center_lon: float,
    center_lat: float,
    radius_m: float,
    zone: int | None = None,
) -> SiteDataset:
    """Выбрать, обрезать, локализовать и нормализовать данные участка (Шаг 1.4)."""
    zone = zone if zone is not None else pick_msk59_zone(center_lon)
    center_x, center_y, _ = wgs84_to_msk59(center_lon, center_lat, zone=zone)

    raw_features = fetch_features_in_buffer(conn, center_lon, center_lat, radius_m)

    features: list[SiteFeature] = []
    for raw in raw_features:
        geom_msk59 = transform_geometry_to_msk59(raw.geometry, zone=zone)
        local_geom = clip_and_localize(geom_msk59, center_x, center_y, radius_m)
        if local_geom is None:
            continue

        normalizer = NORMALIZERS.get(raw.layer)
        if normalizer is not None:
            normalized = normalizer(raw.tags)
            attributes = {name: attr.value for name, attr in normalized.items()}
            confidence = {name: attr.confidence for name, attr in normalized.items()}
        else:
            attributes = {}
            confidence = {}

        features.append(
            SiteFeature(
                layer=raw.layer,
                osm_id=raw.osm_id,
                osm_type=raw.osm_type,
                geometry=local_geom,
                attributes=attributes,
                confidence=confidence,
                raw_tags=raw.tags,
            )
        )

    return SiteDataset(center_lon=center_lon, center_lat=center_lat, radius_m=radius_m, zone=zone, features=features)
