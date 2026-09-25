"""Pydantic-схемы API (Шаг 1.3, п. 1, п. 3: валидация входа).

Радиус 0.5-3 км и координаты в поддерживаемом регионе (зоны МСК-59, Шаг 0.4)
проверяются здесь же, до постановки задачи в очередь — ошибка приходит сразу
и с понятным сообщением, а не после того, как воркер провалит первый шаг.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from topology_geo.coords import pick_msk59_zone


class Center(BaseModel):
    lon: float = Field(..., ge=-180, le=180)
    lat: float = Field(..., ge=-90, le=90)


class JobCreateRequest(BaseModel):
    center: Center
    radius_m: float = Field(..., ge=500, le=3000, description="Радиус задачи, м (0.5-3 км, Шаг 1.3 п.3)")
    layers: list[str] = Field(default_factory=list)
    detail: str = "LOD1"

    @field_validator("center")
    @classmethod
    def _center_in_supported_region(cls, value: Center) -> Center:
        try:
            pick_msk59_zone(value.lon)
        except ValueError as exc:
            raise ValueError(f"координаты вне поддерживаемого региона: {exc}") from exc
        return value


class JobCreateResponse(BaseModel):
    id: uuid.UUID
    status: str


class JobStepOut(BaseModel):
    step_name: str
    step_order: int
    status: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None
    result: dict[str, Any] | None = None


class JobOut(BaseModel):
    id: uuid.UUID
    center: Center
    radius_m: float
    layers: list[str]
    detail: str
    status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    steps: list[JobStepOut]


class FileOut(BaseModel):
    step_name: str
    storage_key: str
    download_url: str
    viewer_url: str | None = None


class FilesResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    files: list[FileOut]
