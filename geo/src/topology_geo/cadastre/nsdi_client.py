"""Клиент для интеграции с НСПД (Росреестр API).

Поиск кадастровых участков по координатам и получение их границ.
API: https://pkk.rosreestr.ru/api/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests
from shapely.geometry import shape

logger = logging.getLogger(__name__)

NSPD_API_BASE = "https://pkk.rosreestr.ru/api"
SEARCH_TIMEOUT = 10


@dataclass(frozen=True)
class CadastreData:
    """Данные об участке из НСПД."""

    cadastre_number: str
    center_lon: float
    center_lat: float
    area_m2: Optional[float] = None
    owner: Optional[str] = None
    address: Optional[str] = None
    boundary_geojson: Optional[dict] = None

    def boundary_polygon(self):
        """Полигон границы участка (Shapely)."""
        if self.boundary_geojson is None:
            return None
        try:
            return shape(self.boundary_geojson["geometry"])
        except (KeyError, ValueError):
            return None


class NSPDClient:
    """Клиент для поиска участков в НСПД."""

    def __init__(self, timeout: int = SEARCH_TIMEOUT, cache: Optional[dict] = None):
        self.timeout = timeout
        self.cache = cache or {}

    def search_by_coords(
        self, lon: float, lat: float, radius_m: int = 500
    ) -> list[CadastreData]:
        """Найти участки по центральным координатам в радиусе.

        Args:
            lon: долгота (МСК-59 переводится в WGS84)
            lat: широта
            radius_m: радиус поиска в метрах

        Returns:
            Список найденных участков с границами.
        """
        cache_key = f"search_{lon}_{lat}_{radius_m}"
        if cache_key in self.cache:
            logger.info(f"Cache hit: {cache_key}")
            return self.cache[cache_key]

        try:
            # Запрос к API Росреестра
            response = requests.get(
                f"{NSPD_API_BASE}/features",
                params={
                    "extent": f"{lon-0.01},{lat-0.01},{lon+0.01},{lat+0.01}",
                    "limit": 100,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            features = response.json().get("features", [])
            results = []

            for feature in features:
                props = feature.get("properties", {})
                cadastre_num = props.get("cn") or props.get("cadastral_number")

                if cadastre_num:
                    data = CadastreData(
                        cadastre_number=str(cadastre_num),
                        center_lon=lon,
                        center_lat=lat,
                        area_m2=props.get("area_m2"),
                        owner=props.get("owner"),
                        address=props.get("address"),
                        boundary_geojson=feature,
                    )
                    results.append(data)
                    logger.info(f"Found cadastre: {cadastre_num}")

            self.cache[cache_key] = results
            return results

        except requests.RequestException as e:
            logger.error(f"NSPD API error: {e}")
            return []

    def get_by_cadastre_number(self, number: str) -> Optional[CadastreData]:
        """Получить участок по кадастровому номеру.

        Args:
            number: кадастровый номер (例: 59:10:0101010:123)

        Returns:
            Данные об участке или None.
        """
        cache_key = f"cadastre_{number}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            response = requests.get(
                f"{NSPD_API_BASE}/features",
                params={"cadastral_number": number},
                timeout=self.timeout,
            )
            response.raise_for_status()

            features = response.json().get("features", [])
            if not features:
                return None

            feature = features[0]
            props = feature.get("properties", {})

            data = CadastreData(
                cadastre_number=number,
                center_lon=props.get("center_longitude", 0),
                center_lat=props.get("center_latitude", 0),
                area_m2=props.get("area_m2"),
                owner=props.get("owner"),
                address=props.get("address"),
                boundary_geojson=feature,
            )

            self.cache[cache_key] = data
            return data

        except requests.RequestException as e:
            logger.error(f"NSPD API error for {number}: {e}")
            return None


# Глобальный клиент (переиспользуется с кэшем)
_global_client = NSPDClient()


def search_cadastre_by_coords(
    lon: float, lat: float, radius_m: int = 500
) -> list[CadastreData]:
    """Поиск участков по координатам (быстрый интерфейс с глобальным кэшем)."""
    return _global_client.search_by_coords(lon, lat, radius_m)


def get_cadastre_by_number(number: str) -> Optional[CadastreData]:
    """Получить участок по номеру (быстрый интерфейс с глобальным кэшем)."""
    return _global_client.get_by_cadastre_number(number)
