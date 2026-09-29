"""Тесты Шага 4.11, п. 3: выгрузка в Pilot-BIM через папку обмена — реальная
запись файлов на диск из реального хранилища, без Postgres."""

from __future__ import annotations

import uuid
from pathlib import Path

from topology_geo.export.exchange_folder import export_job_to_exchange_folder
from topology_geo.storage import InMemoryObjectStorage


def test_export_copies_real_bytes_to_job_subfolder(tmp_path: Path):
    storage = InMemoryObjectStorage()
    job_id = uuid.uuid4()
    storage.upload(f"jobs/{job_id}/site.ifc", b"real-ifc-bytes", content_type="application/x-step")
    storage.upload(f"jobs/{job_id}/site.glb", b"real-glb-bytes", content_type="model/gltf-binary")

    result = export_job_to_exchange_folder(
        storage, job_id, [f"jobs/{job_id}/site.ifc", f"jobs/{job_id}/site.glb"], tmp_path / "exchange",
    )

    assert result.job_folder == tmp_path / "exchange" / str(job_id)
    assert result.missing_keys == []
    assert len(result.exported_files) == 2

    ifc_path = tmp_path / "exchange" / str(job_id) / "site.ifc"
    assert ifc_path.read_bytes() == b"real-ifc-bytes"
    glb_path = tmp_path / "exchange" / str(job_id) / "site.glb"
    assert glb_path.read_bytes() == b"real-glb-bytes"


def test_export_reports_missing_keys_without_failing(tmp_path: Path):
    storage = InMemoryObjectStorage()
    job_id = uuid.uuid4()
    storage.upload(f"jobs/{job_id}/site.ifc", b"real-ifc-bytes")

    result = export_job_to_exchange_folder(
        storage, job_id, [f"jobs/{job_id}/site.ifc", f"jobs/{job_id}/does_not_exist.ifc"], tmp_path / "exchange",
    )

    assert len(result.exported_files) == 1
    assert result.missing_keys == [f"jobs/{job_id}/does_not_exist.ifc"]


def test_export_creates_exchange_folder_if_missing(tmp_path: Path):
    storage = InMemoryObjectStorage()
    job_id = uuid.uuid4()
    storage.upload(f"jobs/{job_id}/site.ifc", b"data")
    nested = tmp_path / "does" / "not" / "exist" / "yet"

    result = export_job_to_exchange_folder(storage, job_id, [f"jobs/{job_id}/site.ifc"], nested)

    assert result.job_folder.is_dir()
    assert (result.job_folder / "site.ifc").exists()
