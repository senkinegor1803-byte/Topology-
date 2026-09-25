"""Нормализация атрибутов объектов по словарю данных (Шаг 1.4, п. 3;
словарь — `docs/data-dictionary.md`).

Приводит теги OSM к типизированным полям с пометкой `confidence`
(`"факт"` — значение взято из тега, `"умолчание"` — тег отсутствовал или не
разобрался). Плановый набор атрибутов (дословно из Шага 1.4, п. 3): тип и
этажность зданий, покрытие дорог, напряжение ЛЭП.

Терпимо к любым «плохим» тегам (отсутствующим, пустым, нечисловым,
отрицательным) — ни один такой случай не должен приводить к исключению,
только к `confidence = "умолчание"`. Это и есть предмет проверки Шага 1.4
(«модульные тесты на 20 типичных случаев тегов, включая ошибочные») —
см. `geo/tests/test_selection_normalize.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

CONFIDENCE_FACT = "факт"
CONFIDENCE_DEFAULT = "умолчание"

DEFAULT_BUILDING_TYPE = "yes"
DEFAULT_BUILDING_LEVELS = 1
DEFAULT_HIGHWAY_CLASS = "unclassified"


@dataclass(frozen=True)
class NormalizedAttribute:
    value: Any
    confidence: str


def _parse_positive_int(value: Any) -> int | None:
    """Разобрать положительное целое из тега; None — тег отсутствует/некорректен.

    Терпимо к дробным строкам (`"3.0"`), пробелам, отрицательным и нулевым
    значениям (трактуются как некорректные — этажность/полосы не бывают <= 0).
    """
    if value is None:
        return None
    try:
        n = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _parse_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        n = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_building(tags: Mapping[str, str]) -> dict[str, NormalizedAttribute]:
    """Тип и этажность здания (Шаг 1.4, п. 3). Высота в метрах и её
    формула-заготовка по умолчанию — забота Шага 1.6 (`docs/math-model.md` §2.5),
    здесь нормализуется только то, что прямо перечислено в Шаге 1.4."""
    raw_type = _non_empty_str(tags.get("building"))
    building_type = raw_type or DEFAULT_BUILDING_TYPE
    type_confidence = CONFIDENCE_FACT if raw_type else CONFIDENCE_DEFAULT

    levels = _parse_positive_int(tags.get("building:levels"))
    levels_confidence = CONFIDENCE_FACT if levels is not None else CONFIDENCE_DEFAULT
    if levels is None:
        levels = DEFAULT_BUILDING_LEVELS

    return {
        "type": NormalizedAttribute(building_type, type_confidence),
        "levels": NormalizedAttribute(levels, levels_confidence),
    }


def normalize_road(tags: Mapping[str, str]) -> dict[str, NormalizedAttribute]:
    """Класс дороги и покрытие (Шаг 1.4, п. 3). Дефолт покрытия по классу —
    забота Шага 1.7; здесь тег либо есть (факт), либо нет (умолчание, значение
    не подставляется — генератор дороги сам решит, чем заменить)."""
    highway_class = _non_empty_str(tags.get("highway"))
    highway_confidence = CONFIDENCE_FACT if highway_class else CONFIDENCE_DEFAULT
    highway_class = highway_class or DEFAULT_HIGHWAY_CLASS

    surface = _non_empty_str(tags.get("surface"))
    surface_confidence = CONFIDENCE_FACT if surface else CONFIDENCE_DEFAULT

    lanes = _parse_positive_int(tags.get("lanes"))
    lanes_confidence = CONFIDENCE_FACT if lanes is not None else CONFIDENCE_DEFAULT

    return {
        "highway_class": NormalizedAttribute(highway_class, highway_confidence),
        "surface": NormalizedAttribute(surface, surface_confidence),
        "lanes": NormalizedAttribute(lanes, lanes_confidence),
    }


def normalize_power(tags: Mapping[str, str]) -> dict[str, NormalizedAttribute]:
    """Напряжение ЛЭП в кВ (Шаг 1.4, п. 3; пример из словаря: `voltage=15000`
    -> `Напряжение_кВ=15`, `docs/data-dictionary.md` §2)."""
    voltage_v = _parse_positive_float(tags.get("voltage"))
    if voltage_v is not None:
        voltage_kv: float | None = round(voltage_v / 1000.0, 3)
        confidence = CONFIDENCE_FACT
    else:
        voltage_kv = None
        confidence = CONFIDENCE_DEFAULT
    return {"voltage_kv": NormalizedAttribute(voltage_kv, confidence)}


def normalize_vegetation(tags: Mapping[str, str]) -> dict[str, NormalizedAttribute]:
    """Порода дерева (словарь данных §2: `species`/`leaf_type`) — не входит в
    список Шага 1.4, п. 3 дословно, но уже описана в словаре Шага 0.8 и нужна
    следующим генераторам не меньше остальных."""
    species = _non_empty_str(tags.get("species")) or _non_empty_str(tags.get("genus"))
    confidence = CONFIDENCE_FACT if species else CONFIDENCE_DEFAULT
    return {"species": NormalizedAttribute(species, confidence)}


NORMALIZERS: dict[str, Any] = {
    "osm_buildings": normalize_building,
    "osm_roads": normalize_road,
    "osm_power": normalize_power,
    "osm_vegetation": normalize_vegetation,
}
