"""Деревья: точки с породой/дефолтом; в массивах — случайная расстановка
заданной плотности (Шаг 1.7, п. 4)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from shapely.geometry import Point

from topology_geo.selection.service import SiteFeature

CONFIDENCE_FACT = "факт"
CONFIDENCE_DEFAULT = "умолчание"

DEFAULT_TREE_SPECIES = "лиственное (умолчание)"
DEFAULT_FOREST_DENSITY_PER_HA = 400.0  # деревьев/га, условно для леса средней полосы
MAX_PLACEMENT_ATTEMPTS_PER_TREE = 20


@dataclass(frozen=True)
class TreePoint:
    x: float
    y: float
    species: str
    confidence: str
    source_osm_id: int


def build_individual_trees(features: list[SiteFeature]) -> list[TreePoint]:
    """Одиночные деревья (`osm_vegetation`, точки — `natural=tree`)."""
    result: list[TreePoint] = []
    for feature in features:
        if feature.layer != "osm_vegetation" or feature.geometry.geom_type != "Point":
            continue
        species = feature.attributes.get("species")
        confidence = feature.confidence.get("species", CONFIDENCE_DEFAULT)
        result.append(
            TreePoint(
                x=feature.geometry.x, y=feature.geometry.y,
                species=species or DEFAULT_TREE_SPECIES, confidence=confidence,
                source_osm_id=feature.osm_id,
            )
        )
    return result


def _as_polygons(geom):
    if geom.geom_type == "MultiPolygon":
        return [g for g in geom.geoms if not g.is_empty]
    return [] if geom.geom_type != "Polygon" or geom.is_empty else [geom]


def scatter_forest_trees(
    features: list[SiteFeature],
    *,
    density_per_ha: float = DEFAULT_FOREST_DENSITY_PER_HA,
    seed: int | None = None,
) -> list[TreePoint]:
    """Случайная расстановка деревьев в полигонах леса/массива (`osm_vegetation`,
    полигоны — `natural=wood`/`landuse=forest,grass`) с заданной плотностью
    (Шаг 1.7, п. 4). Метод отбраковки (rejection sampling): точка в
    ограничивающем прямоугольнике принимается, если попадает в сам полигон.
    """
    rng = random.Random(seed)
    result: list[TreePoint] = []

    for feature in features:
        if feature.layer != "osm_vegetation":
            continue
        for polygon in _as_polygons(feature.geometry):
            area_ha = polygon.area / 10_000.0
            target_count = int(round(area_ha * density_per_ha))
            if target_count <= 0:
                continue

            minx, miny, maxx, maxy = polygon.bounds
            placed = 0
            attempts = 0
            max_attempts = target_count * MAX_PLACEMENT_ATTEMPTS_PER_TREE
            while placed < target_count and attempts < max_attempts:
                attempts += 1
                x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
                if polygon.contains(Point(x, y)):
                    result.append(
                        TreePoint(
                            x=x, y=y, species=DEFAULT_TREE_SPECIES, confidence=CONFIDENCE_DEFAULT,
                            source_osm_id=feature.osm_id,
                        )
                    )
                    placed += 1
    return result
