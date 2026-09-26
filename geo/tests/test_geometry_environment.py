"""Тесты геометрии окружения (Шаг 1.7): дороги, вода, ж/д, деревья.
Проверка плана — визуальная приёмка BIM-специалистом (не автоматизируется);
здесь — геометрическая корректность (ширина/площадь лент, отметки, плотность
расстановки), которая и должна давать тот результат, что BIM-специалист
проверяет глазами."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from topology_geo.geometry.rail import DEFAULT_RAIL_TYPE, build_rail_ribbons
from topology_geo.geometry.road_network import NETWORK_BACKBONE, NETWORK_INTERNAL
from topology_geo.geometry.roads import (
    CONFIDENCE_DEFAULT,
    CONFIDENCE_FACT,
    DEFAULT_WIDTH_M,
    WIDTH_BY_HIGHWAY_CLASS,
    build_road_ribbons,
    compute_width_m,
    rebuild_road_ribbon,
)
from topology_geo.geometry.vegetation import (
    DEFAULT_TREE_SPECIES,
    build_individual_trees,
    scatter_forest_trees,
)
from topology_geo.geometry.water import build_water_areas, build_waterway_ribbons
from topology_geo.selection.service import SiteFeature


def _feature(layer, geometry, osm_id=1, attributes=None, confidence=None, raw_tags=None) -> SiteFeature:
    return SiteFeature(
        layer=layer, osm_id=osm_id, osm_type="W", geometry=geometry,
        attributes=attributes or {}, confidence=confidence or {}, raw_tags=raw_tags or {},
    )


# --- roads ------------------------------------------------------------------


def test_compute_width_uses_tag_when_present():
    feature = _feature("osm_roads", LineString([(0, 0), (10, 0)]), raw_tags={"width": "8"})
    width, confidence = compute_width_m(feature)
    assert width == pytest.approx(8.0)
    assert confidence == CONFIDENCE_FACT


def test_compute_width_falls_back_to_class_default():
    feature = _feature("osm_roads", LineString([(0, 0), (10, 0)]), attributes={"highway_class": "residential"})
    width, confidence = compute_width_m(feature)
    assert width == pytest.approx(WIDTH_BY_HIGHWAY_CLASS["residential"])
    assert confidence == CONFIDENCE_DEFAULT


def test_compute_width_unknown_class_uses_generic_default():
    feature = _feature("osm_roads", LineString([(0, 0), (10, 0)]), attributes={"highway_class": "something_odd"})
    width, _ = compute_width_m(feature)
    assert width == pytest.approx(DEFAULT_WIDTH_M)


def test_road_ribbon_area_matches_length_times_width_for_straight_line():
    length = 100.0
    width = 6.0
    feature = _feature(
        "osm_roads", LineString([(0, 0), (length, 0)]),
        attributes={"highway_class": "residential"}, raw_tags={"width": str(width)},
    )
    ribbons = build_road_ribbons([feature])
    assert len(ribbons) == 1
    # buffer с плоскими торцами точной прямой -> ровно прямоугольник length x width
    assert ribbons[0].ribbon.area == pytest.approx(length * width, rel=1e-6)


def test_road_ribbon_handles_multilinestring():
    feature = _feature(
        "osm_roads", MultiLineString([[(0, 0), (10, 0)], [(20, 0), (30, 0)]]),
        raw_tags={"width": "4"},
    )
    ribbons = build_road_ribbons([feature])
    assert len(ribbons) == 1
    assert ribbons[0].ribbon.area == pytest.approx(2 * 10 * 4, rel=1e-6)


def test_build_road_ribbons_ignores_non_road_layers():
    feature = _feature("osm_buildings", Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
    assert build_road_ribbons([feature]) == []


# --- каркасная/внутриквартальная сеть (Шаг 2.4, п. 1 и 5) -------------------


def test_build_road_ribbons_classifies_backbone_and_internal_network():
    backbone = _feature(
        "osm_roads", LineString([(0, 0), (10, 0)]), osm_id=1, attributes={"highway_class": "primary"},
    )
    internal = _feature(
        "osm_roads", LineString([(0, 0), (10, 0)]), osm_id=2, attributes={"highway_class": "residential"},
    )
    ribbons = {r.osm_id: r for r in build_road_ribbons([backbone, internal])}
    assert ribbons[1].network == NETWORK_BACKBONE
    assert ribbons[2].network == NETWORK_INTERNAL


def test_build_road_ribbons_stores_axis_for_single_line():
    line = LineString([(0, 0), (10, 0)])
    feature = _feature("osm_roads", line, attributes={"highway_class": "residential"})
    ribbons = build_road_ribbons([feature])
    assert ribbons[0].axis == line


def test_build_road_ribbons_axis_is_none_for_multilinestring():
    feature = _feature(
        "osm_roads", MultiLineString([[(0, 0), (10, 0)], [(20, 0), (30, 0)]]),
        attributes={"highway_class": "residential"},
    )
    ribbons = build_road_ribbons([feature])
    assert ribbons[0].axis is None


def test_rebuild_road_ribbon_regenerates_geometry_from_new_axis():
    feature = _feature(
        "osm_roads", LineString([(0, 0), (10, 0)]),
        attributes={"highway_class": "residential"}, raw_tags={"width": "6"},
    )
    road = build_road_ribbons([feature])[0]
    new_axis = LineString([(0, 0), (0, 20)])  # тот же профиль, другое направление/длина

    rebuilt = rebuild_road_ribbon(road, new_axis)

    assert rebuilt.axis == new_axis
    assert rebuilt.width_m == road.width_m  # профиль не меняется, только геометрия
    assert rebuilt.surface == road.surface
    assert rebuilt.network == road.network
    assert rebuilt.ribbon.area == pytest.approx(20.0 * 6.0, rel=1e-6)  # новая длина x старая ширина


def test_rebuild_road_ribbon_rejects_backbone_road():
    feature = _feature("osm_roads", LineString([(0, 0), (10, 0)]), attributes={"highway_class": "primary"})
    road = build_road_ribbons([feature])[0]
    with pytest.raises(ValueError, match="каркасной"):
        rebuild_road_ribbon(road, LineString([(0, 0), (0, 20)]))


def test_rebuild_road_ribbon_rejects_road_without_stored_axis():
    feature = _feature(
        "osm_roads", MultiLineString([[(0, 0), (10, 0)], [(20, 0), (30, 0)]]),
        attributes={"highway_class": "residential"},
    )
    road = build_road_ribbons([feature])[0]
    with pytest.raises(ValueError, match="оси"):
        rebuild_road_ribbon(road, LineString([(0, 0), (0, 20)]))


# --- water ------------------------------------------------------------------


def _flat_terrain(z: float):
    return lambda x, y: z


def test_water_area_level_is_min_boundary_elevation():
    def terrain(x, y):
        return 100.0 + x

    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    feature = _feature("osm_water_areas", poly)
    areas = build_water_areas([feature], terrain)
    assert len(areas) == 1
    assert areas[0].level_z == pytest.approx(100.0)  # min по границе, x=0


def test_waterway_ribbon_area_matches_length_times_width():
    length = 50.0
    width = 3.0
    feature = _feature("osm_waterways", LineString([(0, 0), (length, 0)]))
    ribbons = build_waterway_ribbons([feature], _flat_terrain(0.0), width_m=width)
    assert len(ribbons) == 1
    assert ribbons[0].ribbon.area == pytest.approx(length * width, rel=1e-6)


def test_build_water_areas_ignores_non_water_layers():
    feature = _feature("osm_roads", LineString([(0, 0), (10, 0)]))
    assert build_water_areas([feature], _flat_terrain(0.0)) == []


def test_build_water_areas_smooths_jagged_boundary():
    """Реки/озёра в OSM часто оцифрованы ломаной с резкими углами — контур
    должен быть реально сглажен (метод Чайкина), не просто пропущен как есть."""
    poly = Polygon([(0, 0), (5, 0), (5, 5), (2.5, 1.0), (0, 5)])  # острый "шип" внутрь на (2.5, 1.0)
    feature = _feature("osm_water_areas", poly)
    areas = build_water_areas([feature], _flat_terrain(0.0))
    smoothed = areas[0].polygon

    assert smoothed.is_valid
    assert len(list(smoothed.exterior.coords)) > len(list(poly.exterior.coords))  # срезание углов добавляет точки
    assert smoothed.area == pytest.approx(poly.area, rel=0.3)  # форма не искажена радикально


def test_water_area_level_varies_along_slope_for_elongated_shape_not_flat():
    """Вытянутый (не компактный) водоём на уклоне — уровень воды меняется
    вдоль длины, а не единая плоская отметка (единая «рыла бы траншею» на
    одном конце, см. докстринг модуля/`_long_axis_line`)."""
    def terrain(x, y):
        return 100.0 + 0.1 * x

    # длина 200, ширина 5 - соотношение сторон явно выше WATER_AREA_ELONGATION_RATIO
    poly = Polygon([(0, -2.5), (200, -2.5), (200, 2.5), (0, 2.5)])
    feature = _feature("osm_water_areas", poly)
    water = build_water_areas([feature], terrain)[0]

    assert water.level_fn(5.0, 0.0) == pytest.approx(terrain(5.0, 0.0), abs=1e-6)
    assert water.level_fn(195.0, 0.0) == pytest.approx(terrain(195.0, 0.0), abs=1e-6)
    assert water.level_fn(195.0, 0.0) > water.level_fn(5.0, 0.0)  # следует уклону, не единой отметке

    # сечение, перпендикулярное длине (один и тот же x, разные "берега") -
    # одна и та же отметка, как и должно быть физически
    assert water.level_fn(100.0, -2.4) == pytest.approx(water.level_fn(100.0, 2.4), abs=1e-6)


def test_waterway_ribbon_smoothing_shortens_sharp_zigzag():
    """Срезание углов методом Чайкина всегда укорачивает путь на резких
    поворотах — реальная, проверяемая геометрическая проверка того, что
    сглаживание действительно произошло, а не просто скопировало вход."""
    line = LineString([(0, 0), (10, 0), (10, 10), (0, 10), (0, 20)])
    width = 2.0
    feature = _feature("osm_waterways", line, osm_id=7)
    ribbons = build_waterway_ribbons([feature], _flat_terrain(0.0), width_m=width)
    ribbon = ribbons[0].ribbon

    unsmoothed_ribbon = line.buffer(width / 2, cap_style="flat")
    assert ribbon.area < unsmoothed_ribbon.area

    # концы НЕ сглаживаются (иначе слияние рек в одном узле разошлось бы) -
    # плоский торец буфера проходит РОВНО через конец линии (расстояние 0);
    # если бы конец сдвинулся сглаживанием, точка не лежала бы на границе.
    assert ribbon.boundary.distance(Point(0, 0)) == pytest.approx(0.0, abs=1e-9)
    assert ribbon.boundary.distance(Point(0, 20)) == pytest.approx(0.0, abs=1e-9)


def test_waterway_ribbon_level_follows_line_profile():
    def terrain(x, y):
        return 100.0 + 0.2 * x

    line = LineString([(0, 0), (100, 0)])
    feature = _feature("osm_waterways", line, osm_id=9)
    ribbon = build_waterway_ribbons([feature], terrain, width_m=2.0)[0]

    assert ribbon.level_fn(10.0, 0.0) == pytest.approx(terrain(10.0, 0.0), abs=1e-6)
    assert ribbon.level_fn(90.0, 0.0) == pytest.approx(terrain(90.0, 0.0), abs=1e-6)
    assert ribbon.level_fn(10.0, 0.0) != ribbon.level_fn(90.0, 0.0)


# --- rail ---------------------------------------------------------------


def test_rail_ballast_area_matches_length_times_width():
    length = 200.0
    feature = _feature("osm_railways", LineString([(0, 0), (length, 0)]), raw_tags={"railway": "tram"})
    ribbons = build_rail_ribbons([feature])
    assert len(ribbons) == 1
    from topology_geo.geometry.rail import BALLAST_WIDTH_M

    assert ribbons[0].ballast.area == pytest.approx(length * BALLAST_WIDTH_M, rel=1e-6)
    assert ribbons[0].rail_type == "tram"


def test_rail_type_defaults_when_tag_missing():
    feature = _feature("osm_railways", LineString([(0, 0), (10, 0)]))
    ribbons = build_rail_ribbons([feature])
    assert ribbons[0].rail_type == DEFAULT_RAIL_TYPE


# --- vegetation -----------------------------------------------------------


def test_individual_tree_uses_species_when_present():
    feature = _feature(
        "osm_vegetation", Point(5, 5), attributes={"species": "Betula pendula"}, confidence={"species": "факт"}
    )
    trees = build_individual_trees([feature])
    assert len(trees) == 1
    assert trees[0].species == "Betula pendula"
    assert trees[0].confidence == "факт"


def test_individual_tree_defaults_species_when_missing():
    feature = _feature("osm_vegetation", Point(5, 5), attributes={"species": None}, confidence={"species": "умолчание"})
    trees = build_individual_trees([feature])
    assert trees[0].species == DEFAULT_TREE_SPECIES
    assert trees[0].confidence == "умолчание"


def test_build_individual_trees_ignores_polygon_vegetation():
    forest = _feature("osm_vegetation", Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]))
    assert build_individual_trees([forest]) == []


def test_scatter_forest_trees_matches_expected_density_and_stays_inside_polygon():
    # 1 гектар (100x100 м) при плотности 400/га -> ровно 400 деревьев
    forest = _feature("osm_vegetation", Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]))
    trees = scatter_forest_trees([forest], density_per_ha=400.0, seed=42)

    assert len(trees) == 400
    poly = forest.geometry
    assert all(poly.contains(Point(t.x, t.y)) for t in trees)
    assert all(t.species == DEFAULT_TREE_SPECIES for t in trees)


def test_scatter_forest_trees_is_deterministic_with_seed():
    forest = _feature("osm_vegetation", Polygon([(0, 0), (50, 0), (50, 50), (0, 50)]))
    trees_a = scatter_forest_trees([forest], density_per_ha=200.0, seed=7)
    trees_b = scatter_forest_trees([forest], density_per_ha=200.0, seed=7)
    assert [(t.x, t.y) for t in trees_a] == [(t.x, t.y) for t in trees_b]


def test_scatter_forest_trees_skips_tiny_polygons():
    tiny = _feature("osm_vegetation", Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]))
    trees = scatter_forest_trees([tiny], density_per_ha=400.0, seed=1)
    assert trees == []
