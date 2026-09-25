"""Тест JSON-схемы meta.json (Шаг 0.8): схема валидна и пример проходит по ней."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "meta.schema.json"


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_schema_file_is_valid_json_schema(schema: dict):
    jsonschema.Draft202012Validator.check_schema(schema)


def test_sample_meta_document_validates(schema: dict):
    sample = {
        "task_id": "job-0001",
        "created_at": "2026-09-25T12:00:00Z",
        "generator_version": "0.0.1",
        "center": {"lon": 56.2431, "lat": 58.0105},
        "radius_m": 500,
        "coordinate_system": {
            "projected_crs": "MSK-59",
            "zone": 2,
            "height_system": "Балтийская",
            "base_point": {"x": 2310450.0, "y": -5857320.0, "height": 150.0},
        },
        "layers": [{"name": "relief", "lod": "LOD2", "status": "расчётно"}],
        "files": [{"path": "relief.ifc", "format": "IFC4X3", "layer": "relief", "size_bytes": 12345}],
        "sources": [
            {"name": "OSM Приволжский ФО (Geofabrik)", "retrieved_at": "2026-09-20", "license": "ODbL"}
        ],
    }
    jsonschema.validate(sample, schema)


def test_missing_required_field_is_rejected(schema: dict):
    incomplete = {"task_id": "job-0002"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(incomplete, schema)


def test_radius_out_of_range_is_rejected(schema: dict):
    sample = {
        "task_id": "job-0003",
        "created_at": "2026-09-25T12:00:00Z",
        "generator_version": "0.0.1",
        "center": {"lon": 56.2431, "lat": 58.0105},
        "radius_m": 5000,
        "coordinate_system": {"projected_crs": "MSK-59", "zone": 2, "height_system": "Балтийская"},
        "layers": [],
        "files": [],
        "sources": [],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(sample, schema)
