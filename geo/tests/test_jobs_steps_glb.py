"""Тест шага пайплайна `convert_to_glb` (Шаг 1.9, п. 1) в изоляции: реальная
сборка site.ifc (Шаг 1.8) в хранилище -> реальная конвертация в GLB, без
Postgres (шаг сам его не использует) — только хранилище в памяти."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

from pygltflib import GLTF2

from topology_geo.ifc.assemble import BasePoint, SiteModel, build_site_ifc
from topology_geo.jobs.steps import convert_to_glb
from topology_geo.storage import InMemoryObjectStorage

BASE_POINT = BasePoint(lon=56.25, lat=58.0, zone=2, x=2_310_450.0, y=-5_857_320.0, height=150.0)


def test_convert_to_glb_reads_ifc4x3_from_storage_and_uploads_glb():
    job_id = uuid.uuid4()
    job = SimpleNamespace(id=job_id)
    storage = InMemoryObjectStorage()

    model, _ = build_site_ifc("IFC4X3", SiteModel(), BASE_POINT)
    storage.upload(f"jobs/{job_id}/site_ifc4x3.ifc", model.to_string().encode("utf-8"))

    result = convert_to_glb(None, storage, job)

    assert result["storage_key"] == f"jobs/{job_id}/site.glb"
    assert result["size_bytes"] > 0
    glb_bytes = storage.download(result["storage_key"])
    assert len(glb_bytes) == result["size_bytes"]

    gltf = GLTF2.load_from_bytes(glb_bytes)
    assert gltf.asset.generator.startswith("topology-geo")
