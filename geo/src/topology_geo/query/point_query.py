"""Карточка объекта по клику (Шаг 3.5, п. 1-2): «клик по любой точке →
запрос в PostGIS: зоны, регламент, участок, ближайшие сети с расстояниями,
отметка рельефа».

Панель свойств с группировкой (п. 1: «основное, источник, ограничения,
ссылки») реализуется во вьюере (фронтенд, `web/*/index.html`, тот же приём,
что уже есть у панели свойств IFC-объектов, Шаг 1.9) — здесь собирается
СОДЕРЖИМОЕ карточки на бэкенде; `PointQueryCard.to_grouped_dict()` уже
раскладывает поля по тем же группам, так что фронтенду остаётся отрисовать
структуру, не придумывать её заново.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from topology_geo.constraints.pzz import PzzZone, find_pzz_zone_for_point
from topology_geo.constraints.store import ConstraintZone, find_zones_near_point


@dataclass(frozen=True)
class NearbyZone:
    zone: ConstraintZone
    distance_m: float


@dataclass(frozen=True)
class PointQueryCard:
    lon: float
    lat: float
    nearby_zones: list[NearbyZone]
    pzz_zone: PzzZone | None

    def to_grouped_dict(self) -> dict[str, Any]:
        """Группировка панели свойств (п. 1): «основное, источник,
        ограничения, ссылки» — буквально поля плана, не произвольная
        структура."""
        return {
            "основное": {"lon": self.lon, "lat": self.lat, "территориальная_зона": self.pzz_zone.zone_code if self.pzz_zone else None},
            "ограничения": [
                {
                    "вид": nz.zone.zone_type, "статус": nz.zone.status,
                    "расстояние_м": round(nz.distance_m, 1), "режим": nz.zone.regime,
                    "реестровый_номер": nz.zone.registry_number,
                }
                for nz in self.nearby_zones
            ],
            "источник": [
                {"вид": nz.zone.zone_type, "источник": nz.zone.source_name, "дата": nz.zone.data_timestamp.isoformat()}
                for nz in self.nearby_zones
            ],
            "ссылки": [
                {"вид": nz.zone.zone_type, "документ": nz.zone.document_basis}
                for nz in self.nearby_zones if nz.zone.document_basis
            ],
        }


def query_point(conn, lon: float, lat: float, *, max_distance_m: float = 500.0) -> PointQueryCard:
    """Действие п. 2, буквально: зоны + регламент участка (ПЗЗ) в точке
    клика. Ближайшие ПОДЗЕМНЫЕ сети (`networks.dxf_import`) и отметка
    рельефа в этой точке ЧЕСТНО НЕ включены — обе требуют доступа к
    объектному хранилищу (растр рельефа; DXF-импорт Шага 3.4 не
    персистится в PostGIS, только парсится в память), которого у этой
    функции сознательно нет (принимает только `conn`, как остальной
    `constraints`-слой) — добавлять его ради одной функции означало бы
    завязать запрос карточки на Storage, которого нет ни у одной другой
    функции этого модуля; естественное место для сборки — API-слой (Шаг
    1.3, `api/app.py`), где conn И storage уже оба доступны."""
    nearby = [
        NearbyZone(zone=zone, distance_m=distance_m)
        for zone, distance_m in find_zones_near_point(conn, lon, lat, max_distance_m=max_distance_m)
    ]
    pzz_zone = find_pzz_zone_for_point(conn, lon, lat)
    return PointQueryCard(lon=lon, lat=lat, nearby_zones=nearby, pzz_zone=pzz_zone)
