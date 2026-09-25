"""Обрезка объектов участка по кругу и перевод в локальные координаты
(Шаг 1.4, п. 2).

Круг — ровно радиус задачи `R` (без запаса; запас нужен только для выборки
из PostGIS, чтобы не терять объекты, чьи части попадают в круг — см.
`docs/math-model.md` §2.1). Локальные координаты — участок сдвинут так, что
его центр становится (0, 0), как ожидают дальнейшие генераторы (Шаги 1.5-1.8)
и уже принято в тестовом генераторе IFC (Шаг 0.2).
"""

from __future__ import annotations

from shapely.affinity import translate
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry


def clip_and_localize(
    geom: BaseGeometry, center_x: float, center_y: float, radius_m: float
) -> BaseGeometry | None:
    """Обрезать `geom` (в проекционных координатах, например МСК-59) кругом
    радиуса `radius_m` вокруг `(center_x, center_y)` и сдвинуть в локальные
    координаты участка (центр -> (0, 0)).

    Возвращает `None`, если после обрезки геометрия не попадает в круг вовсе
    (пуста) — вызывающий код должен отбросить такой объект.
    """
    circle = Point(center_x, center_y).buffer(radius_m)
    clipped = geom.intersection(circle)
    if clipped.is_empty:
        return None
    return translate(clipped, xoff=-center_x, yoff=-center_y)
