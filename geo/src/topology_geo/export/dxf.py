"""DXF экспорт (Шаг 2.10, п. 2: «DXF (2D-чертежи)»).

Через библиотеку `ezdxf` (добавлена в зависимости пакета этим шагом).
Плоский 2D-чертёж в плане (вид сверху): контуры зданий, дорожных лент и
водных объектов, каждый на своём слое — типичный состав экспорта в
CAD-программу для дальнейшей работы (подложка для проектировщиков),
НЕ полноценная 3D BIM-модель (для неё есть IFC, Шаг 1.8+).

Что НЕ входит в этот проход (сознательно, честно):
- Горизонтали/изолинии рельефа (TIN участка есть, но контурные линии по
  сечению плоскостями не строятся - самостоятельная задача триангуляции
  сечений, за рамками этого шага).
- Рельсовые пути, ограждения, опоры ЛЭП, растительность и т.п. - тот же
  принцип, что и для LandXML/CityJSON: явно выбранное подмножество
  (здания + дороги + вода), не "всё, что есть в IFC".
- Штриховка/легенда/рамка листа, аннотации размеров - чертёж без
  оформления, только геометрия по слоям.

Координаты - локальные метры участка (та же система, что в IFC/GLB/
LandXML/CityJSON), 2D-проекция (Z полигонов отброшена - контуры в плане).
"""

from __future__ import annotations

import io
from typing import Protocol

import ezdxf

LAYER_BUILDINGS = "Здания"
LAYER_ROADS = "Дороги"
LAYER_WATER = "Вода"

DXF_VERSION = "R2010"


class _HasFootprint(Protocol):
    osm_id: int
    footprint: object  # shapely Polygon


class _HasRibbon(Protocol):
    osm_id: int
    ribbon: object  # shapely Polygon | MultiPolygon | None


class _HasPolygon(Protocol):
    osm_id: int
    polygon: object  # shapely Polygon


def _as_polygons(geom) -> list:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type.startswith("Multi"):
        return [g for g in geom.geoms if not g.is_empty]
    return [geom]


def _add_polygon_outlines(msp, geom, layer: str) -> None:
    """Добавить внешний контур и контуры отверстий полигона (или каждой
    части MultiPolygon) как замкнутые LWPOLYLINE на заданном слое."""
    for polygon in _as_polygons(geom):
        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            # shapely замыкает кольцо явно (первая точка = последняя) - для
            # LWPOLYLINE замыкание уже даёт `close=True`, дублирующая точка не нужна.
            points = [(x, y) for x, y in ring.coords[:-1]]
            msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})


def build_dxf(
    buildings: list[_HasFootprint] | None = None,
    roads: list[_HasRibbon] | None = None,
    water_areas: list[_HasPolygon] | None = None,
    waterways: list[_HasRibbon] | None = None,
) -> bytes:
    """Собрать 2D DXF-чертёж: контуры зданий/дорожных лент/воды по слоям."""
    doc = ezdxf.new(DXF_VERSION)
    doc.layers.add(name=LAYER_BUILDINGS, color=1)  # красный
    doc.layers.add(name=LAYER_ROADS, color=8)  # серый
    doc.layers.add(name=LAYER_WATER, color=5)  # синий
    msp = doc.modelspace()

    for building in buildings or []:
        _add_polygon_outlines(msp, building.footprint, LAYER_BUILDINGS)
    for road in roads or []:
        _add_polygon_outlines(msp, road.ribbon, LAYER_ROADS)
    for water in water_areas or []:
        _add_polygon_outlines(msp, water.polygon, LAYER_WATER)
    for waterway in waterways or []:
        _add_polygon_outlines(msp, waterway.ribbon, LAYER_WATER)

    return _write_bytes(doc)


def _write_bytes(doc) -> bytes:
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")
