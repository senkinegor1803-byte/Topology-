"""Аудит полноты открытых данных OSM (Шаг 0.6 плана).

Считает, какая доля объектов пилотной территории (в круге аудита, обычно
3 км — см. `docs/plan.md`, Шаг 0.1) несёт ключевые теги, нужные генератору
конвейера: этажность зданий, покрытие/число полос дорог, дворовые проезды,
опоры ЛЭП, деревья. Сравнение с эталоном (топосъёмка/ортофото/выезд на место)
и решение «достаточно / дополнить вручную / нужен другой источник» по
каждому слою — по плану ответственность GEO (+ AI считает, человек решает);
здесь есть вспомогательная функция-подсказка `suggest_decisions`, но
финальное решение остаётся за разработчиком.

Модуль сознательно не завязан на конкретный парсер `.osm.pbf` (pyrosm,
osmium, ...): он работает над уже разобранными `OsmElement` — те же функции
одинаково применимы к выгрузке из PostGIS после Шага 1.1. Настоящий прогон на
реальном пилоте (скачать Geofabrik Приволжский ФО, вырезать 3 км, сравнить с
эталоном) — задача, которая выполняется на реальных данных после Шага 0.1 и
здесь не производилась: тесты используют синтетические элементы.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from shapely.geometry.base import BaseGeometry

Kind = Literal["node", "way", "relation"]


@dataclass(frozen=True)
class OsmElement:
    """Один разобранный объект OSM (независимо от источника разбора)."""

    id: int
    kind: Kind
    tags: dict[str, str]
    geometry: BaseGeometry | None = None


@dataclass(frozen=True)
class LayerCompleteness:
    layer: str
    total: int
    complete: int

    @property
    def ratio(self) -> float:
        return self.complete / self.total if self.total else 0.0


@dataclass(frozen=True)
class CompletenessReport:
    buildings_with_levels: LayerCompleteness
    roads_with_surface: LayerCompleteness
    roads_with_lanes: LayerCompleteness
    courtyard_driveways: int
    power_poles: int
    trees: int
    layers: dict[str, LayerCompleteness] = field(default_factory=dict)


def _within(elements: Iterable[OsmElement], boundary: BaseGeometry | None) -> list[OsmElement]:
    if boundary is None:
        return list(elements)
    return [e for e in elements if e.geometry is not None and boundary.intersects(e.geometry)]


def is_building(e: OsmElement) -> bool:
    return "building" in e.tags


def is_highway(e: OsmElement) -> bool:
    return "highway" in e.tags


def is_courtyard_driveway(e: OsmElement) -> bool:
    """Дворовой проезд: `highway=service`, кроме явно магистральных сервисов."""
    if e.tags.get("highway") != "service":
        return False
    return e.tags.get("service", "") in ("", "driveway", "alley", "parking_aisle")


def is_power_pole(e: OsmElement) -> bool:
    return e.tags.get("power") in ("pole", "tower")


def is_tree(e: OsmElement) -> bool:
    return e.tags.get("natural") == "tree"


def _layer_completeness(layer: str, elements: list[OsmElement], has_tag) -> LayerCompleteness:
    total = len(elements)
    complete = sum(1 for e in elements if has_tag(e))
    return LayerCompleteness(layer=layer, total=total, complete=complete)


def build_completeness_report(
    elements: Iterable[OsmElement], boundary: BaseGeometry | None = None
) -> CompletenessReport:
    """Посчитать полноту ключевых слоёв в границе `boundary` (или по всем `elements`)."""
    scoped = _within(elements, boundary)

    buildings = [e for e in scoped if is_building(e)]
    highways = [e for e in scoped if is_highway(e)]

    buildings_with_levels = _layer_completeness(
        "buildings_with_levels", buildings, lambda e: "height" in e.tags or "building:levels" in e.tags
    )
    roads_with_surface = _layer_completeness("roads_with_surface", highways, lambda e: "surface" in e.tags)
    roads_with_lanes = _layer_completeness("roads_with_lanes", highways, lambda e: "lanes" in e.tags)

    courtyard_driveways = sum(1 for e in scoped if is_courtyard_driveway(e))
    power_poles = sum(1 for e in scoped if is_power_pole(e))
    trees = sum(1 for e in scoped if is_tree(e))

    layers = {
        "buildings_with_levels": buildings_with_levels,
        "roads_with_surface": roads_with_surface,
        "roads_with_lanes": roads_with_lanes,
    }

    return CompletenessReport(
        buildings_with_levels=buildings_with_levels,
        roads_with_surface=roads_with_surface,
        roads_with_lanes=roads_with_lanes,
        courtyard_driveways=courtyard_driveways,
        power_poles=power_poles,
        trees=trees,
        layers=layers,
    )


Decision = Literal["достаточно", "дополнить вручную", "нужен другой источник"]


def suggest_decisions(
    report: CompletenessReport,
    *,
    good_threshold: float = 0.8,
    poor_threshold: float = 0.4,
) -> dict[str, Decision]:
    """Подсказка решения по тегированным слоям на основе доли заполненности.

    Не заменяет решение человека (плановый критерий Шага 0.6 явно требует
    "принято решение" по каждому слою) — только сортирует слои по порогам,
    чтобы не пересматривать каждый вручную с нуля.
    """
    decisions: dict[str, Decision] = {}
    for name, layer in report.layers.items():
        if layer.ratio >= good_threshold:
            decisions[name] = "достаточно"
        elif layer.ratio >= poor_threshold:
            decisions[name] = "дополнить вручную"
        else:
            decisions[name] = "нужен другой источник"
    return decisions


def compare_to_reference(report: CompletenessReport, reference_counts: dict[str, int]) -> dict[str, dict[str, float]]:
    """Сравнить число найденных объектов с эталоном (топосъёмка/ортофото/выезд).

    `reference_counts` — словарь {метка: эталонное_число}, ключи произвольные,
    например {"courtyard_driveways": 42, "power_poles": 17, "trees": 260}.
    Возвращает разницу и относительную полноту по каждой метке, присутствующей
    и в отчёте (через `report.layers`/агрегаты), и в эталоне.
    """
    counts = {
        "courtyard_driveways": report.courtyard_driveways,
        "power_poles": report.power_poles,
        "trees": report.trees,
        **{name: layer.complete for name, layer in report.layers.items()},
    }
    result = {}
    for key, reference in reference_counts.items():
        found = counts.get(key)
        if found is None:
            continue
        missing = max(reference - found, 0)
        result[key] = {
            "found": found,
            "reference": reference,
            "missing": missing,
            "coverage": found / reference if reference else 1.0,
        }
    return result
