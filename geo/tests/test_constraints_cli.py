"""Тест CLI загрузки ограничений (Шаг 3.1, п. 1-2: полуавтоматический режим
«файл, полученным по запросу») — реальный психоп-коннект к тестовой БД, тот
же приём окружения, что test_api.py (`POSTGRES_DB` через monkeypatch)."""

from __future__ import annotations

import json
import os

import pytest

from topology_geo.constraints.cli import main
from topology_geo.constraints.store import ensure_schema, find_zones


@pytest.fixture()
def cli_env(pg_test_db, monkeypatch):
    monkeypatch.setenv("POSTGRES_DB", pg_test_db.info.dbname)
    monkeypatch.setenv("POSTGRES_HOST", os.environ.get("POSTGRES_HOST", "localhost"))
    monkeypatch.setenv("POSTGRES_PORT", os.environ.get("POSTGRES_PORT", "5432"))
    monkeypatch.setenv("POSTGRES_USER", os.environ.get("POSTGRES_USER", "topology"))
    monkeypatch.setenv("POSTGRES_PASSWORD", os.environ.get("POSTGRES_PASSWORD", "topology"))
    return pg_test_db


def test_load_file_command_loads_real_geojson(cli_env, tmp_path, capsys):
    ensure_schema(cli_env)
    geojson_path = tmp_path / "zones.geojson"
    geojson_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [56.20, 58.00], [56.21, 58.00], [56.21, 58.01], [56.20, 58.01], [56.20, 58.00],
            ]]},
            "properties": {"zone_type": "Красная линия", "registry_number": "RL-1"},
        }],
    }), encoding="utf-8")

    exit_code = main([
        "load-file", str(geojson_path),
        "--source-name", "ИСОГД (файл по запросу)",
        "--data-timestamp", "2026-03-01",
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "загружено зон: 1" in out

    zones = find_zones(cli_env, bbox=(56.19, 57.99, 56.22, 58.02))
    assert len(zones) == 1
    assert zones[0].zone_type == "Красная линия"
    assert zones[0].source_name == "ИСОГД (файл по запросу)"
