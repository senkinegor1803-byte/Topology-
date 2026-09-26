"""Классификация дорог на каркасную и внутриквартальную сеть (Шаг 2.4, п. 1).

По плану: «каркасные (motorway–tertiary) и внутриквартальные (residential,
service, living_street, footway и т. д.)» — весь диапазон классов OSM
`highway` от `motorway` до `tertiary` включительно, вместе с их `_link`
съездами (реальная часть той же дороги: развязка/съезд каркасной трассы
остаётся каркасной, не становится дворовым проездом на самой развязке).
Всё остальное (включая `unclassified` — по вики Key:highway это НИЖЕ
`tertiary` в иерархии, а не отдельная каркасная категория) — внутриквартальная
сеть. Единая точка классификации: и `geometry.roads` (Шаг 1.7, простые ленты),
и `geometry.streets` (Шаг 2.3, полосы через osm2streets) используют её же —
одна дорога не должна получить разный класс на разных слоях модели.
"""

from __future__ import annotations

NETWORK_BACKBONE = "каркасная"
NETWORK_INTERNAL = "внутриквартальная"

BACKBONE_HIGHWAY_CLASSES = frozenset({
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
})


def classify_road_network(highway_class: str) -> str:
    """Каркасная сеть (`BACKBONE_HIGHWAY_CLASSES`) или внутриквартальная
    (всё остальное — умолчание для неизвестного/отсутствующего класса тоже
    внутриквартальная, не каркасная: если тип дороги не удалось определить,
    считать её значимой федеральной/региональной трассой было бы менее
    безопасным умолчанием, чем считать дворовым проездом)."""
    return NETWORK_BACKBONE if highway_class in BACKBONE_HIGHWAY_CLASSES else NETWORK_INTERNAL
