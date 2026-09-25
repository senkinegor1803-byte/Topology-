"""Тесты генератора тестовых IFC-сцен (Шаг 0.2)."""

from __future__ import annotations

import ifcopenshell.util.element as element_util
import pytest

from topology_geo.ifc.generate_test_ifc import (
    CYRILLIC_PSETS,
    SUPPORTED_SCHEMAS,
    MapConversionParams,
    build_test_model,
    mesh_box,
    mesh_cylinder,
    mesh_terrain_grid,
    validate_model,
)

EXPECTED_CLASSES = {
    "IfcGeographicElement",
    "IfcBuildingElementProxy",
    "IfcSpatialZone",
    "IfcPipeSegment",
    "IfcSite",
}


@pytest.mark.parametrize("schema", SUPPORTED_SCHEMAS)
def test_build_test_model_validates_clean(schema: str):
    model = build_test_model(schema, relief_grid=3)
    assert model.schema == (schema if schema != "IFC4X3" else "IFC4X3")
    issues = validate_model(model)
    assert issues == []


def test_ifc43_uses_native_road_and_bridge():
    model = build_test_model("IFC4X3", relief_grid=3)
    classes = {p.is_a() for p in model.by_type("IfcProduct")}
    assert {"IfcRoad", "IfcBridge"} <= classes


def test_ifc4_falls_back_to_proxy_for_road_and_bridge():
    model = build_test_model("IFC4", relief_grid=3)
    classes = {p.is_a() for p in model.by_type("IfcProduct")}
    assert "IfcRoad" not in classes
    assert "IfcBridge" not in classes
    proxies = model.by_type("IfcBuildingElementProxy")
    replaced = {
        p["Заменяет_класс"]
        for proxy in proxies
        for name, p in element_util.get_psets(proxy).items()
        if name == "Pset_Контекст" and "Заменяет_класс" in p
    }
    assert replaced == {"IfcRoad (недоступен в IFC4)", "IfcBridge (недоступен в IFC4)"}


@pytest.mark.parametrize("schema", SUPPORTED_SCHEMAS)
def test_all_products_carry_expected_classes(schema: str):
    model = build_test_model(schema, relief_grid=3)
    classes = {p.is_a() for p in model.by_type("IfcProduct")}
    assert EXPECTED_CLASSES <= classes


def test_cyrillic_psets_present_on_every_product():
    model = build_test_model("IFC4X3", relief_grid=3)
    products = [p for p in model.by_type("IfcProduct") if p.is_a("IfcElement") or p.is_a("IfcSpatialZone")]
    assert products
    for product in products:
        psets = element_util.get_psets(product)
        for pset_name, props in CYRILLIC_PSETS.items():
            assert pset_name in psets, f"{product}: нет {pset_name}"
            for key in props:
                assert key in psets[pset_name]


def test_map_conversion_uses_given_coordinates():
    params = MapConversionParams(eastings=2_400_000.0, northings=-5_900_000.0, height=173.0)
    model = build_test_model("IFC4X3", relief_grid=3, map_conversion=params)
    mc = model.by_type("IfcMapConversion")[0]
    assert mc.Eastings == pytest.approx(2_400_000.0)
    assert mc.Northings == pytest.approx(-5_900_000.0)
    assert mc.OrthogonalHeight == pytest.approx(173.0)


def test_extra_buildings_increase_product_count():
    small = build_test_model("IFC4", relief_grid=3, extra_buildings=0)
    big = build_test_model("IFC4", relief_grid=3, extra_buildings=10)
    assert len(big.by_type("IfcBuildingElementProxy")) == len(small.by_type("IfcBuildingElementProxy")) + 10


def test_mesh_box_is_closed_manifold_triangle_count():
    verts, faces = mesh_box(2.0, 3.0, 4.0)
    assert len(verts) == 8
    assert len(faces) == 12  # 6 граней * 2 треугольника


def test_mesh_cylinder_vertex_and_face_counts():
    verts, faces = mesh_cylinder(1.0, 5.0, segments=8)
    assert len(verts) == 16  # 8 низ + 8 верх
    # боковые: 2 треугольника * 8 сегментов, + крышки (segments-2)*2
    assert len(faces) == 2 * 8 + 2 * (8 - 2)


def test_mesh_terrain_grid_scales_with_resolution():
    verts_small, faces_small = mesh_terrain_grid(2, 2)
    verts_big, faces_big = mesh_terrain_grid(10, 10)
    assert len(verts_small) == 3 * 3
    assert len(faces_small) == 2 * 2 * 2
    assert len(verts_big) > len(verts_small)
    assert len(faces_big) > len(faces_small)


def test_build_test_model_rejects_unknown_schema():
    with pytest.raises(ValueError):
        build_test_model("IFC2X3")
