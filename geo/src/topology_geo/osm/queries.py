"""Пространственные выборки по таблицам OSM (проверка Шага 1.1; та же формула
буфера переиспользуется выборкой участка на Шаге 1.4 — см. `docs/math-model.md` §2.1).
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol

TABLES: tuple[str, ...] = (
    "osm_buildings",
    "osm_building_parts",
    "osm_roads",
    "osm_railways",
    "osm_water_areas",
    "osm_waterways",
    "osm_vegetation",
    "osm_power",
    "osm_landscaping",
    "osm_entrances",
)

DEFAULT_MARGIN_M = 50.0


class _Connection(Protocol):
    def cursor(self) -> Any: ...


def count_within_radius(
    conn: _Connection,
    lon: float,
    lat: float,
    radius_m: float,
    *,
    margin_m: float = DEFAULT_MARGIN_M,
    tables: Iterable[str] = TABLES,
) -> dict[str, int]:
    """Число объектов каждой таблицы в буфере `radius_m + margin_m` вокруг точки.

    Критерий Шага 1.1: выборка в круге 3 км выполняется за <= 5 с — на реальном
    объёме края (после настоящей загрузки OSM); здесь функция и её корректность
    проверены на синтетических данных, замер времени на масштабе — Шаг 1.10.
    """
    buffer_radius = radius_m + margin_m
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        for table in tables:
            if table not in TABLES:
                raise ValueError(f"Неизвестная таблица {table!r}, ожидается одна из {TABLES}")
            cur.execute(
                f"SELECT count(*) FROM {table} "  # noqa: S608 - table из фиксированного белого списка TABLES
                "WHERE ST_DWithin(geom::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)",
                (lon, lat, buffer_radius),
            )
            counts[table] = cur.fetchone()[0]
    return counts
