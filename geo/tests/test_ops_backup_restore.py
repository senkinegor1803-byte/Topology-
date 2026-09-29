"""Репетиция резервного копирования/восстановления (Шаг 5.1, п. 3:
«учебное восстановление из копии»). Реально вызывает
`infra/backup/backup_postgres.sh`/`restore_postgres.sh` как настоящие
subprocess'ы (не переписывает их логику в Python) на реальном PostGIS —
доказывает, что сами эксплуатационные скрипты рабочие, а не только
описаны в документации."""

from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

import pytest

from topology_geo.auth.store import create_user, ensure_schema, get_user_by_id
from topology_geo.jobs.steps import DEFAULT_STEP_NAMES
from topology_geo.jobs.store import create_job, get_job

BACKUP_DIR = Path(__file__).resolve().parents[2] / "infra" / "backup"
BACKUP_SCRIPT = BACKUP_DIR / "backup_postgres.sh"
RESTORE_SCRIPT = BACKUP_DIR / "restore_postgres.sh"


@pytest.fixture()
def db(pg_test_db):
    ensure_schema(pg_test_db)  # auth (users/sessions/...) каскадом создаёт и jobs
    return pg_test_db


def _run(args: list[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(args, env=env, capture_output=True, text=True, check=False)


def test_backup_scripts_exist_and_are_executable():
    assert BACKUP_SCRIPT.is_file() and BACKUP_SCRIPT.stat().st_mode & 0o111
    assert RESTORE_SCRIPT.is_file() and RESTORE_SCRIPT.stat().st_mode & 0o111


def test_full_backup_and_restore_rehearsal(db, tmp_path, monkeypatch):
    """Настоящая репетиция: реальные данные -> настоящий `pg_dump` (через
    `backup_postgres.sh`) -> восстановление в НОВУЮ базу (через
    `restore_postgres.sh`, настоящий `pg_restore`) -> проверка, что
    восстановленные данные совпадают с исходными байт в байт по ключевым
    полям. Обе базы — реальный Postgres, не мок."""
    import os

    source_db = db.info.dbname
    user = create_user(db, email="backup-test@example.com", password="secret123")
    job = create_job(
        db, center_lon=56.24, center_lat=58.01, radius_m=750, layers=["buildings"], detail="LOD2",
        step_names=DEFAULT_STEP_NAMES,
    )

    pg_env = {
        **os.environ,
        "POSTGRES_HOST": os.environ.get("POSTGRES_HOST", "localhost"),
        "POSTGRES_PORT": os.environ.get("POSTGRES_PORT", "5432"),
        "POSTGRES_USER": os.environ.get("POSTGRES_USER", "topology"),
        "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD", "topology"),
        "POSTGRES_DB": source_db,
        "BACKUP_DIR": str(tmp_path / "backups"),
    }

    backup_started = time.monotonic()
    backup_result = _run(["bash", str(BACKUP_SCRIPT)], env=pg_env)
    backup_elapsed = time.monotonic() - backup_started
    assert backup_result.returncode == 0, backup_result.stderr

    dump_files = list((tmp_path / "backups").glob("*.dump"))
    assert len(dump_files) == 1, "ожидался ровно один файл дампа"
    dump_file = dump_files[0]
    assert dump_file.stat().st_size > 0

    restore_db = f"topology_restore_test_{uuid.uuid4().hex[:8]}"
    try:
        restore_started = time.monotonic()
        restore_result = _run(
            ["bash", str(RESTORE_SCRIPT), str(dump_file), restore_db], env=pg_env,
        )
        restore_elapsed = time.monotonic() - restore_started
        assert restore_result.returncode == 0, restore_result.stderr

        import psycopg

        restored_conn = psycopg.connect(
            dbname=restore_db, host=pg_env["POSTGRES_HOST"], port=pg_env["POSTGRES_PORT"],
            user=pg_env["POSTGRES_USER"], password=pg_env["POSTGRES_PASSWORD"], autocommit=True,
        )
        try:
            restored_user = get_user_by_id(restored_conn, user.id)
            assert restored_user is not None
            assert restored_user.email == "backup-test@example.com"

            restored_job = get_job(restored_conn, job.id)
            assert restored_job is not None
            assert restored_job.radius_m == 750
            assert restored_job.detail == "LOD2"
            assert len(restored_job.steps) == len(DEFAULT_STEP_NAMES)
        finally:
            restored_conn.close()

        print(f"\nРепетиция восстановления: backup={backup_elapsed:.2f}с, restore={restore_elapsed:.2f}с "
              f"(тестовая БД — счётные секунды; реальный объём данных займёт больше, но тот же механизм)")
    finally:
        admin_conn_env = {**pg_env}
        cleanup = psycopg.connect(  # type: ignore[possibly-undefined]
            dbname="postgres", host=admin_conn_env["POSTGRES_HOST"], port=admin_conn_env["POSTGRES_PORT"],
            user=admin_conn_env["POSTGRES_USER"], password=admin_conn_env["POSTGRES_PASSWORD"], autocommit=True,
        )
        try:
            cleanup.execute(f'DROP DATABASE IF EXISTS "{restore_db}" WITH (FORCE)')
        finally:
            cleanup.close()


def test_restore_script_rejects_missing_dump_file(tmp_path):
    import os

    result = _run(["bash", str(RESTORE_SCRIPT), str(tmp_path / "does_not_exist.dump")], env=dict(os.environ))
    assert result.returncode != 0
    assert "не найден" in result.stderr
