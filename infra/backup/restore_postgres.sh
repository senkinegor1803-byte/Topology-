#!/usr/bin/env bash
# Восстановление PostGIS из резервной копии (Шаг 5.1, п. 3: «учебное
# восстановление из копии»). Создаёт (или пересоздаёт) целевую БД и
# восстанавливает в неё дамп из `backup_postgres.sh` (`pg_restore -j`
# — параллельное восстановление по числу CPU, реально ускоряет
# восстановление больших дампов, не только однопоточный проход).
#
# Использование: restore_postgres.sh <файл дампа> [имя целевой БД]
# Переменные окружения: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER,
# POSTGRES_PASSWORD; целевая БД по умолчанию — POSTGRES_DB.
set -euo pipefail

DUMP_FILE="${1:?Использование: restore_postgres.sh <файл дампа> [имя целевой БД]}"
TARGET_DB="${2:-${POSTGRES_DB:-topology}}"

POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-topology}"

if [ ! -f "$DUMP_FILE" ]; then
    echo "Файл дампа не найден: $DUMP_FILE" >&2
    exit 1
fi

export PGPASSWORD="${POSTGRES_PASSWORD:-}"

psql --host="$POSTGRES_HOST" --port="$POSTGRES_PORT" --username="$POSTGRES_USER" \
    --dbname=postgres --set=ON_ERROR_STOP=1 \
    -c "DROP DATABASE IF EXISTS \"$TARGET_DB\" WITH (FORCE);" \
    -c "CREATE DATABASE \"$TARGET_DB\";"

psql --host="$POSTGRES_HOST" --port="$POSTGRES_PORT" --username="$POSTGRES_USER" \
    --dbname="$TARGET_DB" --set=ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS postgis;"

pg_restore --host="$POSTGRES_HOST" --port="$POSTGRES_PORT" --username="$POSTGRES_USER" \
    --dbname="$TARGET_DB" --jobs="$(nproc)" --no-owner \
    "$DUMP_FILE"

echo "База «$TARGET_DB» восстановлена из $DUMP_FILE"
