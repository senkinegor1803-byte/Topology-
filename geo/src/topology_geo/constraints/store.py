"""Официальные ограничения (ЗОУИТ, ПЗЗ, красные линии, ОКН, ООПТ, ...) —
Шаг 3.1: «загрузчик слоёв с фиксацией источника и даты», версионирование при
обновлении. Расчётные зоны Шага 3.2 (`status="расчётно"`) используют ту же
таблицу и тот же загрузчик — разница только в значении `status` и в том, что
`registry_number` у них, как правило, отсутствует (нет официального реестра).

Источник данных — GeoJSON (`FeatureCollection`), а не прямая интеграция с
НСПД/ИСОГД: оба портала реально проверены на доступность из этой среды
(`curl https://nspd.gov.ru`/`pkk.rosreestr.ru` — `Connection timed out`, тот
же класс сетевого ограничения, что и Overpass/Geofabrik в Шаге 2.12, см.
`docs/constraints.md`) — план сам предусматривает эту ситуацию п. 2 («для
слоёв без массовой выгрузки — файл, полученным по запросу»): загрузчик
принимает именно такой файл, GeoJSON — общепринятый формат выгрузки с НСПД
(«скачать» на публичной карте даёт GeoJSON/KML), не придуман для проекта.

Версионирование (п. 4): загрузка зоны с уже известным `registry_number`
(текущей, `superseded_at IS NULL`) не перезаписывает старую строку —
старая помечается `superseded_at = now()`, новая получает
`version = старая.version + 1`. Зоны без `registry_number` (нет реестра —
типично для расчётных, Шаг 3.2) версионированию не подлежат, каждая
загрузка — новая независимая строка.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

STATUS_OFFICIAL = "официально"
STATUS_CALCULATED = "расчётно"


class _Connection(Protocol):
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...


CONSTRAINT_ZONES_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS constraint_zones (
    id BIGSERIAL PRIMARY KEY,
    zone_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('официально', 'расчётно')),
    regime TEXT,
    registry_number TEXT,
    document_basis TEXT,
    source_name TEXT NOT NULL,
    data_timestamp TIMESTAMPTZ NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    version INTEGER NOT NULL DEFAULT 1,
    superseded_at TIMESTAMPTZ,
    geom GEOMETRY(MultiPolygon, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS constraint_zones_geom_idx ON constraint_zones USING GIST (geom);
CREATE INDEX IF NOT EXISTS constraint_zones_zone_type_idx ON constraint_zones (zone_type);
CREATE INDEX IF NOT EXISTS constraint_zones_registry_number_idx ON constraint_zones (registry_number);
CREATE INDEX IF NOT EXISTS constraint_zones_current_idx ON constraint_zones (registry_number) WHERE superseded_at IS NULL;
"""


@dataclass(frozen=True)
class ConstraintZone:
    zone_type: str
    status: str
    source_name: str
    data_timestamp: datetime
    geom_geojson: dict[str, Any]  # геометрия как GeoJSON (Polygon/MultiPolygon, EPSG:4326)
    regime: str | None = None
    registry_number: str | None = None
    document_basis: str | None = None
    id: int | None = None
    version: int = 1
    superseded_at: datetime | None = None


def ensure_schema(conn: _Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(CONSTRAINT_ZONES_SCHEMA_SQL)
    conn.commit()


def _insert_zone(cur, *, zone_type, status, regime, registry_number, document_basis,
                  source_name, data_timestamp, version, geom_geojson) -> int:
    import json

    cur.execute(
        "INSERT INTO constraint_zones (zone_type, status, regime, registry_number, "
        "document_basis, source_name, data_timestamp, version, geom) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, "
        "ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))) RETURNING id",
        (zone_type, status, regime, registry_number, document_basis, source_name,
         data_timestamp, version, json.dumps(geom_geojson)),
    )
    return cur.fetchone()[0]


def load_zone(conn: _Connection, zone: ConstraintZone) -> int:
    """Загрузить (или обновить версией) одну зону. Возвращает `id` новой
    строки. Если `zone.registry_number` совпадает с текущей (не
    `superseded`) версией — та помечается устаревшей, новая получает
    `version + 1` (Шаг 3.1, п. 4)."""
    with conn.cursor() as cur:
        next_version = 1
        if zone.registry_number is not None:
            cur.execute(
                "SELECT id, version FROM constraint_zones "
                "WHERE registry_number = %s AND superseded_at IS NULL",
                (zone.registry_number,),
            )
            row = cur.fetchone()
            if row is not None:
                current_id, current_version = row
                cur.execute(
                    "UPDATE constraint_zones SET superseded_at = now() WHERE id = %s",
                    (current_id,),
                )
                next_version = current_version + 1

        new_id = _insert_zone(
            cur, zone_type=zone.zone_type, status=zone.status, regime=zone.regime,
            registry_number=zone.registry_number, document_basis=zone.document_basis,
            source_name=zone.source_name, data_timestamp=zone.data_timestamp,
            version=next_version, geom_geojson=zone.geom_geojson,
        )
    conn.commit()
    return new_id


def load_zones_from_geojson(
    conn: _Connection, feature_collection: dict[str, Any], *, source_name: str,
    data_timestamp: datetime, default_status: str = STATUS_OFFICIAL,
) -> list[int]:
    """Загрузчик п. 1-2: `FeatureCollection` (выгрузка с НСПД/ИСОГД или файл,
    полученный по запросу — оба приходят как GeoJSON, разница только в том,
    автоматом получен файл или вручную, для самого загрузчика она не важна).

    Каждый `Feature.properties` может нести `zone_type`/`status`/`regime`/
    `registry_number`/`document_basis` — отсутствующий `zone_type`
    обязателен и вызывает `ValueError` (честно, не подставляется
    заглушка); `status` по умолчанию — `default_status`. Для `status=
    "официально"` (Шаг 3.1) `registry_number` ОБЯЗАТЕЛЕН — таков критерий
    приёмки шага («все ЗОУИТ пилота из НСПД присутствуют в модели с
    реестровыми номерами»); для `status="расчётно"` (Шаг 3.2) он, как
    правило, отсутствует и это НЕ ошибка."""
    ids = []
    for feature in feature_collection.get("features", []):
        props = feature.get("properties") or {}
        zone_type = props.get("zone_type")
        if not zone_type:
            raise ValueError(f"у объекта нет zone_type: {feature!r}")
        status = props.get("status", default_status)
        registry_number = props.get("registry_number")
        if status == STATUS_OFFICIAL and not registry_number:
            raise ValueError(f"официальная зона без реестрового номера: {zone_type!r}")

        zone = ConstraintZone(
            zone_type=zone_type, status=status, source_name=source_name,
            data_timestamp=data_timestamp, geom_geojson=feature["geometry"],
            regime=props.get("regime"), registry_number=registry_number,
            document_basis=props.get("document_basis"),
        )
        ids.append(load_zone(conn, zone))
    return ids


def _row_to_zone(row) -> ConstraintZone:
    import json

    (id_, zone_type, status, regime, registry_number, document_basis, source_name,
     data_timestamp, version, superseded_at, geom_geojson) = row
    return ConstraintZone(
        id=id_, zone_type=zone_type, status=status, regime=regime,
        registry_number=registry_number, document_basis=document_basis,
        source_name=source_name, data_timestamp=data_timestamp, version=version,
        superseded_at=superseded_at, geom_geojson=json.loads(geom_geojson),
    )


def find_zones(
    conn: _Connection, bbox: tuple[float, float, float, float], *, include_superseded: bool = False,
) -> list[ConstraintZone]:
    """Текущие (или все, если `include_superseded=True`) зоны, пересекающие
    `bbox` (minx, miny, maxx, maxy, EPSG:4326)."""
    minx, miny, maxx, maxy = bbox
    where_superseded = "" if include_superseded else "AND superseded_at IS NULL"
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id, zone_type, status, regime, registry_number, document_basis, "
            f"source_name, data_timestamp, version, superseded_at, ST_AsGeoJSON(geom) "
            f"FROM constraint_zones "
            f"WHERE ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 4326)) {where_superseded} "
            f"ORDER BY id",
            (minx, miny, maxx, maxy),
        )
        rows = cur.fetchall()
    return [_row_to_zone(r) for r in rows]


def zone_history(conn: _Connection, registry_number: str) -> list[ConstraintZone]:
    """Все версии зоны с данным реестровым номером, от новой к старой
    (Шаг 3.1, п. 4 — проверить, что история версий действительно хранится,
    а не перезаписывается)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, zone_type, status, regime, registry_number, document_basis, "
            "source_name, data_timestamp, version, superseded_at, ST_AsGeoJSON(geom) "
            "FROM constraint_zones WHERE registry_number = %s ORDER BY version DESC",
            (registry_number,),
        )
        rows = cur.fetchall()
    return [_row_to_zone(r) for r in rows]
