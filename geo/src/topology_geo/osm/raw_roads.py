"""Восстановление графа узлов дорог из PostGIS (Шаг 2.3, п. 1).

`osm2pgsql` (Шаг 1.1) раскладывает объекты по слоям с итоговой геометрией и
тегами, но не хранит связность узлов между разными way — двум дорогам,
делящим общий узел на перекрёстке в исходном OSM, после импорта соответствуют
две независимые записи `osm_roads` с похожими, но не идентичными по смыслу
геометриями. Для построения полос через osm2streets (`geometry.streets`)
нужна именно связность: общий ID узла = настоящий перекрёсток.

Решение — `style.lua` сохраняет для каждого way его `nodes` (массив ID узлов
в порядке вершин `geom`, доп. колонка `osm_roads.nodes`, Шаг 2.3, п. 1).
Здесь эти ID зашиваются обратно в вершины геометрии и собирается валидный
OSM XML (узлы + way с исходными тегами) — вход для osm2streets, без
повторного обращения к исходному `.osm`/`.pbf`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol
from xml.sax.saxutils import quoteattr

from shapely import wkb as shapely_wkb
from shapely.geometry.base import BaseGeometry

from topology_geo.osm.queries import DEFAULT_MARGIN_M


class _Connection(Protocol):
    def cursor(self) -> Any: ...


@dataclass(frozen=True)
class RawRoadWay:
    osm_id: int
    tags: dict[str, str]
    node_ids: list[int]  # тот же порядок, что и вершины geometry
    geometry: BaseGeometry  # LineString, WGS-84 (EPSG:4326), как хранится в osm_roads


def fetch_raw_roads_in_buffer(
    conn: _Connection,
    lon: float,
    lat: float,
    radius_m: float,
    *,
    margin_m: float = DEFAULT_MARGIN_M,
) -> list[RawRoadWay]:
    """Все дороги из буфера `radius_m + margin_m` вокруг `(lon, lat)` с
    исходными тегами, геометрией (WGS-84) и ID узлов (та же формула буфера,
    что и `selection.query.fetch_features_in_buffer`, Шаг 1.4)."""
    buffer_radius = radius_m + margin_m
    roads: list[RawRoadWay] = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT osm_id, tags, nodes, ST_AsBinary(geom) FROM osm_roads "
            "WHERE ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)",
            (lon, lat, buffer_radius),
        )
        for osm_id, tags, nodes, geom_wkb in cur.fetchall():
            geometry = shapely_wkb.loads(bytes(geom_wkb))
            roads.append(
                RawRoadWay(osm_id=osm_id, tags=tags or {}, node_ids=list(nodes or []), geometry=geometry)
            )
    return roads


def build_osm_xml(roads: Iterable[RawRoadWay]) -> str:
    """Собрать валидный OSM XML (узлы, затем way с исходными тегами) из
    `RawRoadWay` — вход для osm2streets (Шаг 2.3, п. 1). Общий ID узла на
    нескольких way восстанавливает настоящий перекрёсток (см. docstring
    модуля). Way с несогласованной длиной `node_ids`/геометрии (не должно
    случаться при консистентном импорте) пропускается, не валит весь набор."""
    node_coords: dict[int, tuple[float, float]] = {}
    way_xml_parts: list[str] = []

    for road in roads:
        coords = list(road.geometry.coords)
        if len(coords) != len(road.node_ids) or len(coords) < 2:
            continue

        for node_id, (lon, lat) in zip(road.node_ids, coords):
            node_coords.setdefault(node_id, (lon, lat))

        nd_xml = "".join(f'<nd ref="{node_id}"/>' for node_id in road.node_ids)
        tag_xml = "".join(
            f"<tag k={quoteattr(str(k))} v={quoteattr(str(v))}/>" for k, v in road.tags.items()
        )
        way_xml_parts.append(f'<way id="{road.osm_id}" version="1">{nd_xml}{tag_xml}</way>')

    node_xml_parts = [
        f'<node id="{node_id}" lat="{lat}" lon="{lon}" version="1"/>'
        for node_id, (lon, lat) in sorted(node_coords.items())
    ]

    body = "".join(node_xml_parts) + "".join(way_xml_parts)
    return f"<?xml version='1.0' encoding='UTF-8'?><osm version=\"0.6\" generator=\"topology-geo\">{body}</osm>"
