"""Тесты Шага 3.6: приём и привязка ЖК. Реальный IFC-фикстур собран через
`ifcopenshell.api` (тот же приём, что `ifc/generate_test_ifc.py`) с ДВУМЯ
`IfcBuilding` (секциями) — там нет готового генератора с несколькими
зданиями, строим минимальный здесь."""

from __future__ import annotations

import math

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.geometry
import ifcopenshell.api.georeference
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import pytest
from shapely.geometry import Point, Polygon, box

from topology_geo.siting.building_placement import (
    building_footprints_by_building,
    place_building_manually,
    read_map_conversion,
)


def _box_mesh(length: float, width: float, height: float):
    hl, hw = length / 2, width / 2
    verts = [
        (-hl, -hw, 0.0), (hl, -hw, 0.0), (hl, hw, 0.0), (-hl, hw, 0.0),
        (-hl, -hw, height), (hl, -hw, height), (hl, hw, height), (-hl, hw, height),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
    ]
    return verts, faces


def _translation_matrix(x: float, y: float, z: float):
    return [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]


@pytest.fixture()
def two_section_ifc(tmp_path):
    f = ifcopenshell.file(schema="IFC4")
    project = ifcopenshell.api.root.create_entity(f, ifc_class="IfcProject", name="ЖК (тест)")
    ifcopenshell.api.context.add_context(f, context_type="Model")
    model_context = f.by_type("IfcGeometricRepresentationContext")[0]
    body_context = ifcopenshell.api.context.add_context(
        f, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=model_context
    )

    site = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSite", name="Участок")
    ifcopenshell.api.aggregate.assign_object(f, products=[site], relating_object=project)

    ifcopenshell.api.georeference.add_georeferencing(f)
    ifcopenshell.api.georeference.edit_georeferencing(
        f, projected_crs={"Name": "MSK-59 zone 2 (тест)"},
        coordinate_operation={
            "Eastings": 2_310_450.0, "Northings": -5_857_320.0, "OrthogonalHeight": 150.0,
            "XAxisAbscissa": math.cos(math.radians(30.0)), "XAxisOrdinate": math.sin(math.radians(30.0)),
            "Scale": 1.0,
        },
    )

    for name, x_offset in [("Секция 1", 0.0), ("Секция 2", 30.0)]:
        building = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuilding", name=name)
        ifcopenshell.api.aggregate.assign_object(f, products=[building], relating_object=site)
        storey = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuildingStorey", name="Этаж 1")
        ifcopenshell.api.aggregate.assign_object(f, products=[storey], relating_object=building)

        wall = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuildingElementProxy", name="Корпус")
        verts, faces = _box_mesh(20.0, 10.0, 15.0)
        rep = ifcopenshell.api.geometry.add_mesh_representation(f, context=body_context, vertices=[verts], faces=[faces])
        ifcopenshell.api.geometry.assign_representation(f, product=wall, representation=rep)
        ifcopenshell.api.geometry.edit_object_placement(f, product=wall, matrix=_translation_matrix(x_offset, 0.0, 0.0))
        ifcopenshell.api.spatial.assign_container(f, products=[wall], relating_structure=storey)

    ifc_path = tmp_path / "zhk.ifc"
    f.write(str(ifc_path))
    return f, ifc_path


def test_read_map_conversion_real_ifc(two_section_ifc):
    model, _path = two_section_ifc

    mc = read_map_conversion(model)

    assert mc is not None
    assert mc.eastings == pytest.approx(2_310_450.0)
    assert mc.northings == pytest.approx(-5_857_320.0)
    assert mc.height == pytest.approx(150.0)
    assert mc.rotation_deg == pytest.approx(30.0, abs=0.01)
    assert "MSK-59" in mc.crs_name


def test_read_map_conversion_returns_none_without_georeference():
    f = ifcopenshell.file(schema="IFC4")
    ifcopenshell.api.root.create_entity(f, ifc_class="IfcProject", name="Без геопривязки")

    assert read_map_conversion(f) is None


def test_building_footprints_by_building_groups_by_ifcbuilding(two_section_ifc):
    model, _path = two_section_ifc

    footprints = building_footprints_by_building(model)

    assert set(footprints.keys()) == {"Секция 1", "Секция 2"}
    # Корпус 20x10 вокруг (0,0) и (30,0) - оболочки не должны пересекаться
    assert not footprints["Секция 1"].intersects(footprints["Секция 2"])
    assert footprints["Секция 1"].area == pytest.approx(20.0 * 10.0, rel=0.05)


def test_place_building_manually_within_site():
    footprint_local = box(-10, -5, 10, 5)  # 20x10, центр в (0,0)
    site_boundary = box(0, 0, 100, 100)

    result = place_building_manually(
        footprint_local, origin_x_m=50.0, origin_y_m=50.0, rotation_deg=0.0, site_boundary=site_boundary,
    )

    assert result.is_within_site is True
    assert result.overlap_ratio == pytest.approx(1.0)
    assert result.footprint.centroid.equals(Point(50.0, 50.0))


def test_place_building_manually_outside_site_detected():
    footprint_local = box(-10, -5, 10, 5)
    site_boundary = box(0, 0, 100, 100)

    result = place_building_manually(
        footprint_local, origin_x_m=5.0, origin_y_m=50.0, rotation_deg=0.0, site_boundary=site_boundary,
    )

    assert result.is_within_site is False
    assert 0.0 < result.overlap_ratio < 1.0


def test_place_building_manually_applies_rotation():
    footprint_local = box(-10, -1, 10, 1)  # узкий прямоугольник вдоль X
    site_boundary = box(-100, -100, 100, 100)

    result = place_building_manually(
        footprint_local, origin_x_m=0.0, origin_y_m=0.0, rotation_deg=90.0, site_boundary=site_boundary,
    )

    minx, miny, maxx, maxy = result.footprint.bounds
    assert (maxx - minx) == pytest.approx(2.0, abs=1e-6)  # после поворота на 90° узкая сторона стала X
    assert (maxy - miny) == pytest.approx(20.0, abs=1e-6)
