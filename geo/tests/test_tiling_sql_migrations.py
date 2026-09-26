"""Проверка, что geo/sql/005_tile_cache.sql не разошёлся с embedded-копией в
topology_geo.tiling.cache. Не требует Postgres."""

from __future__ import annotations

from pathlib import Path

from topology_geo.tiling.cache import TILE_CACHE_SCHEMA_SQL

SQL_FILE = Path(__file__).resolve().parents[1] / "sql" / "005_tile_cache.sql"


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
    assert file_body.strip() == TILE_CACHE_SCHEMA_SQL.strip()
