#!/usr/bin/env bash
# Резервное копирование PostGIS (Шаг 5.1, п. 2-3): реальный pg_dump в
# custom-формате (-Fc — сжатый, поддерживает параллельное восстановление
# через pg_restore -j), не текстовый SQL-дамп. Один файл на снимок,
# имя — таймстемп UTC, чтобы несколько копий не перезаписывали друг друга.
#
# Переменные окружения (те же имена, что читает topology_geo.devcheck):
#   POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD,
#   POSTGRES_DB, BACKUP_DIR (каталог для файлов дампа, по умолчанию ./backups)
set -euo pipefail

POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-topology}"
POSTGRES_DB="${POSTGRES_DB:-topology}"
BACKUP_DIR="${BACKUP_DIR:-./backups}"

mkdir -p "$BACKUP_DIR"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_file="$BACKUP_DIR/${POSTGRES_DB}_${timestamp}.dump"

PGPASSWORD="${POSTGRES_PASSWORD:-}" pg_dump \
    --host="$POSTGRES_HOST" --port="$POSTGRES_PORT" --username="$POSTGRES_USER" \
    --format=custom --compress=9 --file="$output_file" \
    "$POSTGRES_DB"

echo "Резервная копия создана: $output_file ($(du -h "$output_file" | cut -f1))"
