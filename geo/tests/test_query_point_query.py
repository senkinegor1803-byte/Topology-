"""Тесты Шага 3.5, п. 1-2: карточка объекта по клику (зоны рядом + ПЗЗ),
реальный PostGIS. Сценарий из критерия шага - «где проходит ВЛ и её
охранная зона» - воспроизведён буквально (see
test_query_point_finds_power_line_safety_zone_scenario)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from topology_geo.constraints.pzz import PzzZone, ensure_schema as ensure_pzz_schema, load_pzz_zone
from topology_geo.constraints.store import ConstraintZone, STATUS_CALCULATED, STATUS_OFFICIAL, ensure_schema, load_zone
from topology_geo.query.point_query import query_point

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)

# Небольшой квадрат вокруг точки клика (56.20, 58.00)
NEAR_SQUARE = {"type": "Polygon", "coordinates": [[
    [56.199, 57.999], [56.201, 57.999], [56.201, 58.001], [56.199, 58.001], [56.199, 57.999],
]]}
FAR_SQUARE = {"type": "Polygon", "coordinates": [[
    [10.0, 10.0], [10.01, 10.0], [10.01, 10.01], [10.0, 10.01], [10.0, 10.0],
]]}


@pytest.fixture()
def db(pg_test_db):
    ensure_schema(pg_test_db)
    ensure_pzz_schema(pg_test_db)
    return pg_test_db


def test_query_point_returns_nearby_zones_sorted_by_distance(db):
    load_zone(db, ConstraintZone(
        zone_type="охранная зона ЛЭП", status=STATUS_CALCULATED, source_name="расчёт (тест)",
        data_timestamp=TS, geom_geojson=NEAR_SQUARE, regime="не ближе 10 м, ВЛ 10 кВ",
    ))
    load_zone(db, ConstraintZone(
        zone_type="ООПТ", status=STATUS_OFFICIAL, source_name="НСПД (тест)",
        data_timestamp=TS, geom_geojson=FAR_SQUARE, registry_number="OOPT-1",
    ))

    card = query_point(db, lon=56.200, lat=58.000, max_distance_m=1000.0)

    assert len(card.nearby_zones) == 1  # дальняя зона (~7000 км) вне 1000 м
    assert card.nearby_zones[0].zone.zone_type == "охранная зона ЛЭП"
    assert card.nearby_zones[0].distance_m < 50.0  # точка внутри зоны - расстояние около 0


def test_query_point_includes_pzz_zone(db):
    load_pzz_zone(db, PzzZone(
        zone_code="Ж-1", vri=["жилая застройка"], setback_m=3.0, max_height_m=40.0,
        source_name="ИСОГД (тест)", data_timestamp=TS, geom_geojson=NEAR_SQUARE,
    ))

    card = query_point(db, lon=56.200, lat=58.000)

    assert card.pzz_zone is not None
    assert card.pzz_zone.zone_code == "Ж-1"


def test_query_point_power_line_safety_zone_scenario_two_clicks(db):
    """Критерий Шага 3.5: «где проходит ВЛ и её охранная зона» - 2 клика.
    Здесь один запрос `query_point` = один «клик» (второй клик - открыть
    карточку, что и возвращает функция) - находит охранную зону ЛЭП и её
    атрибуты (напряжение в `regime`, статус «расчётно») без доп. действий."""
    load_zone(db, ConstraintZone(
        zone_type="охранная зона ЛЭП", status=STATUS_CALCULATED, source_name="расчёт (тест)",
        data_timestamp=TS, geom_geojson=NEAR_SQUARE, regime="ВЛ 110 кВ, полуширина 20 м",
    ))

    card = query_point(db, lon=56.200, lat=58.000)
    grouped = card.to_grouped_dict()

    power_zones = [z for z in grouped["ограничения"] if z["вид"] == "охранная зона ЛЭП"]
    assert len(power_zones) == 1
    assert "110 кВ" in power_zones[0]["режим"]


def test_point_query_card_grouped_dict_has_plan_required_groups(db):
    load_zone(db, ConstraintZone(
        zone_type="ОКН, зона охраны", status=STATUS_OFFICIAL, source_name="ИСОГД (тест)",
        data_timestamp=TS, geom_geojson=NEAR_SQUARE, registry_number="OKN-1",
        document_basis="Приказ №5",
    ))

    card = query_point(db, lon=56.200, lat=58.000)
    grouped = card.to_grouped_dict()

    assert set(grouped.keys()) == {"основное", "ограничения", "источник", "ссылки"}
    assert grouped["ссылки"][0]["документ"] == "Приказ №5"


def test_query_point_no_zones_returns_empty_card(db):
    card = query_point(db, lon=0.0, lat=0.0)
    assert card.nearby_zones == []
    assert card.pzz_zone is None
