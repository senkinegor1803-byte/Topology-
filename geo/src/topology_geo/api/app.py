"""FastAPI-приложение конвейера (Шаг 1.3, п. 1).

`POST /jobs` ставит задачу в очередь (Celery+Redis, Шаг 1.3, п. 2); реальное
исполнение шагов — `topology_geo.jobs.pipeline` + `topology_geo.jobs.steps`
(Шаги 1.1/1.2). Валидация входа — на уровне Pydantic-схем (`schemas.py`), так
ошибка возвращается сразу, до постановки в очередь.
"""

from __future__ import annotations

import mimetypes
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles

from topology_geo.api.schemas import (
    Center,
    FileOut,
    FilesResponse,
    JobCreateRequest,
    JobCreateResponse,
    JobOut,
    JobStepOut,
)
from topology_geo.devcheck import load_environment_config
from topology_geo.jobs import store
from topology_geo.jobs.steps import DEFAULT_STEP_NAMES
from topology_geo.tasks.pipeline_tasks import enqueue_job, get_storage

VIEWER_DIR = Path(__file__).resolve().parents[1] / "web" / "viewer"


def _connect() -> psycopg.Connection:
    config = load_environment_config()
    return psycopg.connect(config.postgres.dsn, autocommit=True)


def get_connection():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = _connect()
    try:
        store.ensure_schema(conn)
    finally:
        conn.close()
    yield


app = FastAPI(title="Топология: API конвейера", version="0.1.0", lifespan=lifespan)
app.mount("/viewer", StaticFiles(directory=VIEWER_DIR), name="viewer")


def _download_url(job_id: uuid.UUID, storage_key: str) -> str:
    return f"/models/{job_id}/download?key={quote(storage_key, safe='')}"


def _viewer_url(job_id: uuid.UUID, download_url: str) -> str:
    return f"/viewer/index.html?model={quote(download_url, safe='')}&job={job_id}"


def _job_to_out(job: store.Job) -> JobOut:
    return JobOut(
        id=job.id,
        center=Center(lon=job.center_lon, lat=job.center_lat),
        radius_m=job.radius_m,
        layers=job.layers,
        detail=job.detail,
        status=job.status,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        steps=[
            JobStepOut(
                step_name=s.step_name,
                step_order=s.step_order,
                status=s.status,
                started_at=s.started_at,
                finished_at=s.finished_at,
                error_message=s.error_message,
                result=s.result,
            )
            for s in job.steps
        ],
    )


@app.post("/jobs", response_model=JobCreateResponse, status_code=201)
def create_job(payload: JobCreateRequest, conn=Depends(get_connection)) -> JobCreateResponse:
    job = store.create_job(
        conn,
        center_lon=payload.center.lon,
        center_lat=payload.center.lat,
        radius_m=payload.radius_m,
        layers=payload.layers,
        detail=payload.detail,
        step_names=DEFAULT_STEP_NAMES,
    )
    enqueue_job(str(job.id))
    return JobCreateResponse(id=job.id, status=store.get_job(conn, job.id).status)


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: uuid.UUID, conn=Depends(get_connection)) -> JobOut:
    job = store.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="задача не найдена")
    return _job_to_out(job)


def _file_out(job_id: uuid.UUID, step_name: str, storage_key: str) -> FileOut:
    download_url = _download_url(job_id, storage_key)
    viewer_url = _viewer_url(job_id, download_url) if storage_key.endswith(".glb") else None
    return FileOut(step_name=step_name, storage_key=storage_key, download_url=download_url, viewer_url=viewer_url)


@app.get("/models/{job_id}/files", response_model=FilesResponse)
def get_job_files(job_id: uuid.UUID, conn=Depends(get_connection)) -> FilesResponse:
    job = store.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="задача не найдена")
    files: list[FileOut] = []
    for s in job.steps:
        if s.status != store.STATUS_DONE or not s.result:
            continue
        if "storage_key" in s.result:
            files.append(_file_out(job.id, s.step_name, s.result["storage_key"]))
        for schema_name, schema_result in s.result.get("schemas", {}).items():
            if "storage_key" in schema_result:
                files.append(_file_out(job.id, f"{s.step_name}:{schema_name}", schema_result["storage_key"]))
            # Каркасная/внутриквартальная сеть отдельными файлами (Шаг 2.4,
            # п. 4) - тот же schema_result, но под своими ключами
            # (`roads_backbone_storage_key`/`roads_internal_storage_key`,
            # `assemble_ifc`), не единственным `storage_key`.
            for key, value in schema_result.items():
                if key != "storage_key" and key.endswith("_storage_key"):
                    prefix = key.removesuffix("_storage_key")
                    files.append(_file_out(job.id, f"{s.step_name}:{schema_name}:{prefix}", value))
    return FilesResponse(job_id=job.id, status=job.status, files=files)


@app.get("/models/{job_id}/download")
def download_job_file(job_id: uuid.UUID, key: str, conn=Depends(get_connection)) -> Response:
    """Отдать байты результата шага из хранилища (Шаг 1.9, п. 3: ссылка на
    модель из журнала задачи ведёт сюда) — префикс `key` привязан к job_id,
    чтобы одна задача не могла скачать файлы другой."""
    job = store.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="задача не найдена")
    if not key.startswith(f"jobs/{job_id}/"):
        raise HTTPException(status_code=404, detail="файл не найден")
    try:
        data = get_storage().download(key)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="файл не найден") from None
    media_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    if key.endswith(".glb"):
        media_type = "model/gltf-binary"
    elif key.endswith(".ifc"):
        media_type = "application/x-step"
    return Response(content=data, media_type=media_type)
