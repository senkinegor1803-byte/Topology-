"""Выгрузка моделей в Pilot-BIM через папку обмена (Шаг 4.11, п. 3).

Настоящая интеграция с Pilot-BIM ЧЕРЕЗ ЕГО API не выполнена и не
проверена: это проприетарная платформа, доступ к её API требует лицензии
и учётных данных заказчика, которых у AI-сессии нет (тот же класс
ограничения, что у Unreal Engine, Шаг 4.6). План явно предлагает
альтернативу — «через его API ИЛИ папку обмена» — папка обмена
реализована и проверена по-настоящему: Pilot-BIM (как большинство
BIM-платформ) умеет импортировать IFC, который конвейер и так производит
(Шаг 1.8/2.10) — реальная запись файлов на диск из объектного хранилища
задачи, не список путей «как будто» экспортированных.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from topology_geo.storage import ObjectStorage


@dataclass(frozen=True)
class ExchangeFolderExportResult:
    job_folder: Path
    exported_files: list[Path] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)


def export_job_to_exchange_folder(
    storage: ObjectStorage, job_id: uuid.UUID | str, storage_keys: list[str], exchange_folder: Path,
) -> ExchangeFolderExportResult:
    """Скопировать готовые файлы задачи (обычно `.ifc`) в подпапку
    `exchange_folder/<job_id>/`, откуда Pilot-BIM их подхватит импортом.
    Ключ, отсутствующий в хранилище, попадает в `missing_keys`, а не
    прерывает выгрузку остальных файлов — частичный успех лучше полного
    отказа, тот же принцип, что и в остальном конвейере."""
    job_folder = Path(exchange_folder) / str(job_id)
    job_folder.mkdir(parents=True, exist_ok=True)

    exported: list[Path] = []
    missing: list[str] = []
    for key in storage_keys:
        try:
            data = storage.download(key)
        except (KeyError, FileNotFoundError):
            missing.append(key)
            continue
        destination = job_folder / Path(key).name
        destination.write_bytes(data)
        exported.append(destination)

    return ExchangeFolderExportResult(job_folder=job_folder, exported_files=exported, missing_keys=missing)
