"""Реестр GlobalId в PostGIS: связь `osm_id -> GlobalId` собранной модели
(Шаг 1.8, п. 2) — чтобы по объекту OSM найти его представление в `site.ifc`
и наоборот (например, для последующего обновления модели без пересчёта
всех GlobalId заново).

`IFC_GLOBALID_REGISTRY_SCHEMA_SQL` — встроенная копия
`geo/sql/004_ifc_globalid_registry.sql` (тот же приём, что и в
`topology_geo.osm.import_log`/`topology_geo.relief.coverage`, синхронность
проверена тестом `geo/tests/test_ifc_registry_sql_migrations.py`).
"""

from __future__ import annotations

from typing import Any, Protocol

from topology_geo.ifc.assemble import GlobalIdRegistry


class _Connection(Protocol):
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...


IFC_GLOBALID_REGISTRY_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS ifc_globalid_registry (
    id BIGSERIAL PRIMARY KEY,
    model_id TEXT NOT NULL,
    layer TEXT NOT NULL,
    osm_id BIGINT NOT NULL,
    global_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_id, layer, osm_id)
);

CREATE INDEX IF NOT EXISTS ifc_globalid_registry_model_idx ON ifc_globalid_registry (model_id);
CREATE INDEX IF NOT EXISTS ifc_globalid_registry_global_id_idx ON ifc_globalid_registry (global_id);
"""


def ensure_schema(conn: _Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(IFC_GLOBALID_REGISTRY_SCHEMA_SQL)
    conn.commit()


def register_global_ids(conn: _Connection, model_id: str, entries: GlobalIdRegistry) -> None:
    """Сохранить пары `(слой, osm_id, GlobalId)`, собранные `assemble.build_site_ifc`,
    для модели `model_id`. Повторный вызов с теми же значениями — не ошибка
    (`ON CONFLICT DO UPDATE`), это учитывает пересборку модели с теми же
    исходными объектами."""
    if not entries:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO ifc_globalid_registry (model_id, layer, osm_id, global_id) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (model_id, layer, osm_id) DO UPDATE SET global_id = EXCLUDED.global_id",
            [(model_id, layer, osm_id, global_id) for layer, osm_id, global_id in entries],
        )
    conn.commit()


def find_global_id(conn: _Connection, model_id: str, layer: str, osm_id: int) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT global_id FROM ifc_globalid_registry WHERE model_id = %s AND layer = %s AND osm_id = %s",
            (model_id, layer, osm_id),
        )
        row = cur.fetchone()
    return row[0] if row else None


def find_by_global_id(conn: _Connection, global_id: str) -> tuple[str, str, int] | None:
    """Обратный поиск: по `GlobalId` найти `(model_id, layer, osm_id)`."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT model_id, layer, osm_id FROM ifc_globalid_registry WHERE global_id = %s",
            (global_id,),
        )
        row = cur.fetchone()
    return (row[0], row[1], row[2]) if row else None
