"""Тесты Шага 2.10, п. 2: экспорт DXF (контуры зданий/дорог/воды по слоям)."""

from __future__ import annotations

import io
from dataclasses import dataclass

import ezdxf
from shapely.geometry import Polygon, box

from topology_geo.export.dxf import LAYER_BUILDINGS, LAYER_ROADS, LAYER_WATER, build_dxf


@dataclass
class _FakeBuilding:
    osm_id: int
    footprint: object


@dataclass
class _FakeRoad:
    osm_id: int
    ribbon: object


@dataclass
class _FakeWaterArea:
    osm_id: int
    polygon: object


def _read_back(dxf_bytes: bytes):
    return ezdxf.read(io.StringIO(dxf_bytes.decode("utf-8")))


def test_build_dxf_empty_inputs_produces_readable_empty_drawing():
    dxf_bytes = build_dxf()
    doc = _read_back(dxf_bytes)
    msp = doc.modelspace()
    assert len(list(msp)) == 0


def test_build_dxf_building_footprint_on_buildings_layer():
    building = _FakeBuilding(osm_id=1, footprint=box(0, 0, 10, 6))
    doc = _read_back(build_dxf(buildings=[building]))
    entities = list(doc.modelspace().query(f'LWPOLYLINE[layer=="{LAYER_BUILDINGS}"]'))
    assert len(entities) == 1
    assert entities[0].closed is True
    assert len(entities[0]) == 4  # 4 угла прямоугольника (без замыкающей точки)


def test_build_dxf_building_with_hole_produces_two_polylines():
    outer = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(3, 3), (3, 6), (6, 6), (6, 3)]])
    building = _FakeBuilding(osm_id=2, footprint=outer)
    doc = _read_back(build_dxf(buildings=[building]))
    entities = list(doc.modelspace().query(f'LWPOLYLINE[layer=="{LAYER_BUILDINGS}"]'))
    assert len(entities) == 2  # внешний контур + отверстие


def test_build_dxf_road_ribbon_on_roads_layer():
    road = _FakeRoad(osm_id=3, ribbon=box(0, 0, 20, 6))
    doc = _read_back(build_dxf(roads=[road]))
    entities = list(doc.modelspace().query(f'LWPOLYLINE[layer=="{LAYER_ROADS}"]'))
    assert len(entities) == 1


def test_build_dxf_water_area_on_water_layer():
    water = _FakeWaterArea(osm_id=4, polygon=box(0, 0, 15, 15))
    doc = _read_back(build_dxf(water_areas=[water]))
    entities = list(doc.modelspace().query(f'LWPOLYLINE[layer=="{LAYER_WATER}"]'))
    assert len(entities) == 1


def test_build_dxf_skips_road_with_none_ribbon():
    road = _FakeRoad(osm_id=5, ribbon=None)
    doc = _read_back(build_dxf(roads=[road]))
    assert len(list(doc.modelspace())) == 0


def test_build_dxf_all_three_layers_are_registered():
    doc = _read_back(build_dxf())
    names = {layer.dxf.name for layer in doc.layers}
    assert {LAYER_BUILDINGS, LAYER_ROADS, LAYER_WATER} <= names
