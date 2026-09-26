"""Тесты нормализации атрибутов (Шаг 1.4, п. 3). Критерий приёмки шага —
модульные тесты на 20+ типичных случаев тегов, включая ошибочные: ни один
"плохой" тег не должен приводить к исключению, только к
confidence="умолчание"."""

from __future__ import annotations

import pytest

from topology_geo.selection.normalize import (
    CONFIDENCE_DEFAULT,
    CONFIDENCE_FACT,
    NORMALIZERS,
    normalize_building,
    normalize_building_part,
    normalize_entrance,
    normalize_power,
    normalize_road,
    normalize_vegetation,
)

# --- Здания: тип и этажность -------------------------------------------------


def test_building_with_type_and_levels_is_fact():
    result = normalize_building({"building": "yes", "building:levels": "5"})
    assert result["type"].value == "yes"
    assert result["type"].confidence == CONFIDENCE_FACT
    assert result["levels"].value == 5
    assert result["levels"].confidence == CONFIDENCE_FACT


def test_building_specific_type_without_levels_defaults_levels():
    result = normalize_building({"building": "house"})
    assert result["type"].value == "house"
    assert result["type"].confidence == CONFIDENCE_FACT
    assert result["levels"].value == 1
    assert result["levels"].confidence == CONFIDENCE_DEFAULT


def test_building_no_tags_at_all_defaults_everything():
    result = normalize_building({})
    assert result["type"].value == "yes"
    assert result["type"].confidence == CONFIDENCE_DEFAULT
    assert result["levels"].value == 1
    assert result["levels"].confidence == CONFIDENCE_DEFAULT


def test_building_empty_string_type_treated_as_absent():
    result = normalize_building({"building": "   "})
    assert result["type"].value == "yes"
    assert result["type"].confidence == CONFIDENCE_DEFAULT


@pytest.mark.parametrize(
    ("levels_tag", "expected_value", "expected_confidence"),
    [
        ("5", 5, CONFIDENCE_FACT),
        ("3.7", 3, CONFIDENCE_FACT),  # дробная строка -> усечение, но это факт из тега
        ("  7  ", 7, CONFIDENCE_FACT),  # пробелы вокруг числа
        ("-2", 1, CONFIDENCE_DEFAULT),  # отрицательное -> некорректно
        ("0", 1, CONFIDENCE_DEFAULT),  # ноль этажей не бывает
        ("abc", 1, CONFIDENCE_DEFAULT),  # не число
        ("", 1, CONFIDENCE_DEFAULT),  # пустая строка
        (None, 1, CONFIDENCE_DEFAULT),  # тег отсутствует
        ("١٢", 12, CONFIDENCE_FACT),  # арабо-индийские цифры - float() в Python их разбирает как 12
    ],
)
def test_building_levels_parsing_cases(levels_tag, expected_value, expected_confidence):
    tags = {"building": "yes"}
    if levels_tag is not None:
        tags["building:levels"] = levels_tag
    result = normalize_building(tags)
    assert result["levels"].value == expected_value
    assert result["levels"].confidence == expected_confidence


# --- Части зданий (building:part) --------------------------------------------


def test_building_part_with_type_and_levels_is_fact():
    result = normalize_building_part({"building:part": "roof", "building:levels": "2"})
    assert result["type"].value == "roof"
    assert result["type"].confidence == CONFIDENCE_FACT
    assert result["levels"].value == 2
    assert result["levels"].confidence == CONFIDENCE_FACT


def test_building_part_no_tags_at_all_defaults_everything():
    result = normalize_building_part({})
    assert result["type"].value == "yes"
    assert result["type"].confidence == CONFIDENCE_DEFAULT
    assert result["levels"].value == 1
    assert result["levels"].confidence == CONFIDENCE_DEFAULT


def test_building_part_ignores_plain_building_tag():
    # building:part и building - разные теги (вики Key:building:part); значение
    # building не должно просачиваться в тип части.
    result = normalize_building_part({"building": "house", "building:part": "verticalpassage"})
    assert result["type"].value == "verticalpassage"


# --- Входы (entrance) ----------------------------------------------------------


@pytest.mark.parametrize(
    ("entrance_tag", "expected_value", "expected_confidence"),
    [
        ("yes", "yes", CONFIDENCE_FACT),
        ("main", "main", CONFIDENCE_FACT),
        ("staircase", "staircase", CONFIDENCE_FACT),
        ("   ", "yes", CONFIDENCE_DEFAULT),  # пробелы -> считается отсутствующим
        (None, "yes", CONFIDENCE_DEFAULT),  # тег отсутствует (не должно случаться в osm_entrances, но не падает)
    ],
)
def test_entrance_type_parsing_cases(entrance_tag, expected_value, expected_confidence):
    tags = {} if entrance_tag is None else {"entrance": entrance_tag}
    result = normalize_entrance(tags)
    assert result["type"].value == expected_value
    assert result["type"].confidence == expected_confidence


# --- Дороги: класс, покрытие, полосы -----------------------------------------


def test_road_full_tags_is_fact():
    result = normalize_road({"highway": "residential", "surface": "asphalt", "lanes": "2"})
    assert result["highway_class"].value == "residential"
    assert result["highway_class"].confidence == CONFIDENCE_FACT
    assert result["surface"].value == "asphalt"
    assert result["surface"].confidence == CONFIDENCE_FACT
    assert result["lanes"].value == 2
    assert result["lanes"].confidence == CONFIDENCE_FACT


def test_road_minimal_tags_defaults_surface_and_lanes():
    result = normalize_road({"highway": "service"})
    assert result["highway_class"].value == "service"
    assert result["highway_class"].confidence == CONFIDENCE_FACT
    assert result["surface"].value is None
    assert result["surface"].confidence == CONFIDENCE_DEFAULT
    assert result["lanes"].value is None
    assert result["lanes"].confidence == CONFIDENCE_DEFAULT


def test_road_no_highway_tag_uses_unclassified_default():
    result = normalize_road({"surface": "asphalt"})
    assert result["highway_class"].value == "unclassified"
    assert result["highway_class"].confidence == CONFIDENCE_DEFAULT


def test_road_whitespace_only_surface_treated_as_absent():
    result = normalize_road({"highway": "track", "surface": "   "})
    assert result["surface"].value is None
    assert result["surface"].confidence == CONFIDENCE_DEFAULT


@pytest.mark.parametrize(
    ("lanes_tag", "expected_value", "expected_confidence"),
    [
        ("2", 2, CONFIDENCE_FACT),
        ("4.0", 4, CONFIDENCE_FACT),
        ("two", None, CONFIDENCE_DEFAULT),  # словом, не числом
        ("-1", None, CONFIDENCE_DEFAULT),  # отрицательное
        ("0", None, CONFIDENCE_DEFAULT),  # ноль полос не бывает
        (None, None, CONFIDENCE_DEFAULT),  # тег отсутствует
    ],
)
def test_road_lanes_parsing_cases(lanes_tag, expected_value, expected_confidence):
    tags = {"highway": "residential"}
    if lanes_tag is not None:
        tags["lanes"] = lanes_tag
    result = normalize_road(tags)
    assert result["lanes"].value == expected_value
    assert result["lanes"].confidence == expected_confidence


# --- Электросети: напряжение --------------------------------------------------


@pytest.mark.parametrize(
    ("voltage_tag", "expected_kv", "expected_confidence"),
    [
        ("15000", 15.0, CONFIDENCE_FACT),
        ("10000", 10.0, CONFIDENCE_FACT),
        ("400", 0.4, CONFIDENCE_FACT),
        ("not_a_number", None, CONFIDENCE_DEFAULT),
        ("-15000", None, CONFIDENCE_DEFAULT),  # отрицательное напряжение некорректно
        ("0", None, CONFIDENCE_DEFAULT),
        (None, None, CONFIDENCE_DEFAULT),
        ("15000;20000", None, CONFIDENCE_DEFAULT),  # список значений OSM (несколько цепей) - не разбираем здесь
    ],
)
def test_power_voltage_parsing_cases(voltage_tag, expected_kv, expected_confidence):
    tags = {"power": "line"}
    if voltage_tag is not None:
        tags["voltage"] = voltage_tag
    result = normalize_power(tags)
    assert result["voltage_kv"].value == expected_kv
    assert result["voltage_kv"].confidence == expected_confidence


# --- Растительность: порода ---------------------------------------------------


def test_vegetation_species_present_is_fact():
    result = normalize_vegetation({"natural": "tree", "species": "Betula pendula"})
    assert result["species"].value == "Betula pendula"
    assert result["species"].confidence == CONFIDENCE_FACT


def test_vegetation_falls_back_to_genus():
    result = normalize_vegetation({"natural": "tree", "genus": "Betula"})
    assert result["species"].value == "Betula"
    assert result["species"].confidence == CONFIDENCE_FACT


def test_vegetation_no_species_or_genus_defaults_to_none():
    result = normalize_vegetation({"natural": "tree"})
    assert result["species"].value is None
    assert result["species"].confidence == CONFIDENCE_DEFAULT


# --- Общее ---------------------------------------------------------------


def test_normalizers_registry_covers_expected_layers():
    assert set(NORMALIZERS) == {
        "osm_buildings",
        "osm_building_parts",
        "osm_roads",
        "osm_power",
        "osm_vegetation",
        "osm_entrances",
    }
    assert NORMALIZERS["osm_buildings"] is normalize_building
    assert NORMALIZERS["osm_building_parts"] is normalize_building_part
    assert NORMALIZERS["osm_entrances"] is normalize_entrance
