"""Выборка объектов OSM в буфере вокруг центра участка (Шаг 1.4, п. 1):
«По центру и радиусу построить буфер в МСК-59, выбрать объекты из PostGIS с
запасом 50 м».

Сам буфер — тот же геометрический фильтр (по `geography`, буфер = радиус +
запас), что уже проверен в `topology_geo.osm.queries.count_within_radius`
(Шаг 1.1), только здесь возвращаются не счётчики, а сами объекты (геометрия +
теги) для дальнейшей обрезки (`clip.py`) и нормализации (`normalize.py`).
Геометрия в таблицах OSM хранится в WGS-84 (флекс-стиль Шага 1.1); перевод в
МСК-59 — на вызывающей стороне через `coords.transform_geometry_to_msk59`,
не здесь, чтобы не платить за репроекцию для объектов, которые потом всё
равно окажутся вне круга при обрезке.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from shapely import wkb as shapely_wkb
from shapely.geometry.base import BaseGeometry

from topology_geo.osm.queries import DEFAULT_MARGIN_M, TABLES


class _Connection(Protocol):
    def cursor(self) -> Any: ...


@dataclass(frozen=True)
class RawFeature:
    layer: str
    osm_id: int
    osm_type: str
    tags: dict[str, str]
    geometry: BaseGeometry  # в WGS-84 (EPSG:4326), как хранится в osm_* таблицах


def fetch_features_in_buffer(
    conn: _Connection,
    lon: float,
    lat: float,
    radius_m: float,
    *,
    margin_m: float = DEFAULT_MARGIN_M,
    tables: Iterable[str] = TABLES,
) -> list[RawFeature]:
    """Все объекты из `tables`, попадающие в буфер `radius_m + margin_m`
    вокруг точки `(lon, lat)`, с геометрией (WGS-84) и полным набором тегов.
    """
    buffer_radius = radius_m + margin_m
    features: list[RawFeature] = []

    with conn.cursor() as cur:
        for table in tables:
            if table not in TABLES:
                raise ValueError(f"Неизвестная таблица {table!r}, ожидается одна из {TABLES}")
            cur.execute(
                f"SELECT osm_id, osm_type, tags, ST_AsBinary(geom) FROM {table} "  # noqa: S608 - table из TABLES
                "WHERE ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)",
                (lon, lat, buffer_radius),
            )
            for osm_id, osm_type, tags, geom_wkb in cur.fetchall():
                geometry = shapely_wkb.loads(bytes(geom_wkb))
                features.append(
                    RawFeature(layer=table, osm_id=osm_id, osm_type=osm_type, tags=tags or {}, geometry=geometry)
                )

    return features
