"""Растительность и благоустройство (Шаг 1.7, п. 4 — базовые деревья; Шаг
2.9 — высота из `height`, плотность массива по типу леса, газоны отдельно
от леса, кустарники).

`osm2pgsql` (Шаг 1.1; `natural=scrub` добавлен Шагом 2.9) собирает точки
(`natural=tree`) и полигоны (`natural=wood/scrub`, `landuse=forest/grass`)
в одну таблицу `osm_vegetation` — те же принципы разбора по тегу внутри
слоя, что и у `osm_power`/`osm_landscaping`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from shapely.geometry import Point

from topology_geo.selection.service import SiteFeature

CONFIDENCE_FACT = "факт"
CONFIDENCE_DEFAULT = "умолчание"

DEFAULT_TREE_SPECIES = "лиственное (умолчание)"
# Типовая высота дерева, если тег `height` отсутствует (Шаг 2.9, действие
# «высотой из height») — характерный взрослый экземпляр уличного/паркового
# дерева средней полосы, не расчёт по породе/возрасту.
DEFAULT_TREE_HEIGHT_M = 12.0

# Плотность массива по типу леса (Шаг 2.9: «рассеивание с плотностью по типу
# леса») — по тегу `leaf_type` полигона (`natural=wood`/`landuse=forest`),
# типовые иллюстративные значения (хвойный лес обычно гуще смешанного/
# лиственного в средней полосе), не из авторитетного источника лесоустройства.
FOREST_DENSITY_PER_HA_BY_LEAF_TYPE = {
    "needleleaved": 500.0,
    "mixed": 400.0,
    "broadleaved": 350.0,
}
# Умолчание при отсутствии тега `leaf_type` вовсе — прежнее плоское значение
# Шага 1.7, поведение по умолчанию не меняется.
DEFAULT_FOREST_DENSITY_PER_HA = 400.0
MAX_PLACEMENT_ATTEMPTS_PER_TREE = 20

# Кустарник (Шаг 2.9, `natural=scrub`) - гораздо ниже и гуще дерева;
# типовые иллюстративные значения, тот же принцип честности, что и у
# плотности леса выше.
SHRUB_HEIGHT_M = 1.2
SHRUB_DENSITY_PER_HA = 1500.0


def _parse_positive_float(value) -> float | None:
    if value is None:
        return None
    try:
        n = float(str(value).strip())
    except ValueError:
        return None
    return n if n > 0 else None


@dataclass(frozen=True)
class TreePoint:
    x: float
    y: float
    species: str
    confidence: str
    source_osm_id: int
    height_m: float = DEFAULT_TREE_HEIGHT_M
    height_confidence: str = CONFIDENCE_DEFAULT


@dataclass(frozen=True)
class ShrubPoint:
    x: float
    y: float
    height_m: float
    source_osm_id: int


@dataclass(frozen=True)
class LawnArea:
    osm_id: int
    polygon: object  # shapely Polygon/MultiPolygon, локальные координаты


def build_individual_trees(features: list[SiteFeature]) -> list[TreePoint]:
    """Одиночные деревья (`osm_vegetation`, точки — `natural=tree`)."""
    result: list[TreePoint] = []
    for feature in features:
        if feature.layer != "osm_vegetation" or feature.geometry.geom_type != "Point":
            continue
        species = feature.attributes.get("species")
        confidence = feature.confidence.get("species", CONFIDENCE_DEFAULT)
        height = _parse_positive_float(feature.raw_tags.get("height"))
        height_confidence = CONFIDENCE_FACT if height is not None else CONFIDENCE_DEFAULT
        result.append(
            TreePoint(
                x=feature.geometry.x, y=feature.geometry.y,
                species=species or DEFAULT_TREE_SPECIES, confidence=confidence,
                source_osm_id=feature.osm_id,
                height_m=height if height is not None else DEFAULT_TREE_HEIGHT_M,
                height_confidence=height_confidence,
            )
        )
    return result


def _as_polygons(geom):
    if geom.geom_type == "MultiPolygon":
        return [g for g in geom.geoms if not g.is_empty]
    return [] if geom.geom_type != "Polygon" or geom.is_empty else [geom]


def _scatter_points(polygon, target_count: int, rng: random.Random) -> list[tuple[float, float]]:
    minx, miny, maxx, maxy = polygon.bounds
    placed: list[tuple[float, float]] = []
    attempts = 0
    max_attempts = target_count * MAX_PLACEMENT_ATTEMPTS_PER_TREE
    while len(placed) < target_count and attempts < max_attempts:
        attempts += 1
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if polygon.contains(Point(x, y)):
            placed.append((x, y))
    return placed


def scatter_forest_trees(
    features: list[SiteFeature],
    *,
    density_per_ha: float | None = None,
    seed: int | None = None,
) -> list[TreePoint]:
    """Случайная расстановка деревьев в полигонах леса (`osm_vegetation`,
    `natural=wood`/`landuse=forest`) с плотностью ПО ТИПУ ЛЕСА (Шаг 2.9) —
    тег `leaf_type` полигона выбирает
    `FOREST_DENSITY_PER_HA_BY_LEAF_TYPE`, при отсутствии тега — прежнее
    плоское `DEFAULT_FOREST_DENSITY_PER_HA` (Шаг 1.7, поведение по
    умолчанию не меняется). Метод отбраковки (rejection sampling): точка в
    ограничивающем прямоугольнике принимается, если попадает в сам полигон.

    `landuse=grass` (газон) сюда НЕ входит с Шага 2.9 — газон не лес, лес
    на нём не рассеивается, см. `build_lawns`. Явный `density_per_ha`
    отключает выбор по типу леса (используется одно и то же значение для
    всех полигонов — нужен, например, детерминированным тестам)."""
    rng = random.Random(seed)
    result: list[TreePoint] = []

    for feature in features:
        if feature.layer != "osm_vegetation":
            continue
        if str(feature.raw_tags.get("landuse", "")).strip().lower() == "grass":
            continue
        if str(feature.raw_tags.get("natural", "")).strip().lower() == "scrub":
            continue
        if density_per_ha is not None:
            density = density_per_ha
        else:
            leaf_type = str(feature.raw_tags.get("leaf_type", "")).strip().lower()
            density = FOREST_DENSITY_PER_HA_BY_LEAF_TYPE.get(leaf_type, DEFAULT_FOREST_DENSITY_PER_HA)

        for polygon in _as_polygons(feature.geometry):
            target_count = int(round(polygon.area / 10_000.0 * density))
            if target_count <= 0:
                continue
            for x, y in _scatter_points(polygon, target_count, rng):
                result.append(
                    TreePoint(
                        x=x, y=y, species=DEFAULT_TREE_SPECIES, confidence=CONFIDENCE_DEFAULT,
                        source_osm_id=feature.osm_id,
                        height_m=DEFAULT_TREE_HEIGHT_M, height_confidence=CONFIDENCE_DEFAULT,
                    )
                )
    return result


def build_lawns(features: list[SiteFeature]) -> list[LawnArea]:
    """Газоны (`landuse=grass`, Шаг 2.9) — плоская зелёная поверхность, БЕЗ
    рассеивания деревьев (в отличие от леса/массива)."""
    result: list[LawnArea] = []
    for feature in features:
        if feature.layer != "osm_vegetation":
            continue
        if str(feature.raw_tags.get("landuse", "")).strip().lower() != "grass":
            continue
        for polygon in _as_polygons(feature.geometry):
            result.append(LawnArea(osm_id=feature.osm_id, polygon=polygon))
    return result


def scatter_shrubs(
    features: list[SiteFeature],
    *,
    density_per_ha: float = SHRUB_DENSITY_PER_HA,
    seed: int | None = None,
) -> list[ShrubPoint]:
    """Кустарники (`natural=scrub`, Шаг 2.9) — та же расстановка отбраковкой,
    что и у леса, но НИЖЕ и ГУЩЕ (`SHRUB_HEIGHT_M`/`SHRUB_DENSITY_PER_HA`),
    отдельным типом объекта (`ShrubPoint`, не `TreePoint` — куст не дерево)."""
    rng = random.Random(seed)
    result: list[ShrubPoint] = []
    for feature in features:
        if feature.layer != "osm_vegetation":
            continue
        if str(feature.raw_tags.get("natural", "")).strip().lower() != "scrub":
            continue
        for polygon in _as_polygons(feature.geometry):
            target_count = int(round(polygon.area / 10_000.0 * density_per_ha))
            if target_count <= 0:
                continue
            for x, y in _scatter_points(polygon, target_count, rng):
                result.append(ShrubPoint(x=x, y=y, height_m=SHRUB_HEIGHT_M, source_osm_id=feature.osm_id))
    return result
