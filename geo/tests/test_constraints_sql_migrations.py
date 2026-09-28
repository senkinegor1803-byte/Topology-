"""Проверка, что geo/sql/006_constraint_zones.sql не разошёлся с
embedded-копией в topology_geo.constraints.store. Не требует Postgres —
чистая проверка файлов (тот же приём, что test_osm_sql_migrations.py)."""

from __future__ import annotations

from pathlib import Path

from topology_geo.constraints.store import CONSTRAINT_ZONES_SCHEMA_SQL

SQL_FILE = Path(__file__).resolve().parents[1] / "sql" / "006_constraint_zones.sql"


def _strip_leading_comment_block(text: str) -> str:
    lines = text.splitlines(keepends=True)
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "" or stripped.startswith("--"):
            body_start = i + 1
        else:
            break
    return "".join(lines[body_start:])


def test_sql_file_matches_embedded_schema():
    file_body = _strip_leading_comment_block(SQL_FILE.read_text(encoding="utf-8"))
    assert file_body.strip() == CONSTRAINT_ZONES_SCHEMA_SQL.strip()
