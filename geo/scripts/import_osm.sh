#!/usr/bin/env bash
# Полный (пере-)импорт выгрузки OSM в PostGIS через флекс-стиль (Шаг 1.1, п. 1-2).
#
# Использование:
#   ./import_osm.sh path/to/privolzhsky-fed-district-latest.osm.pbf "Geofabrik Приволжский ФО" 2026-09-20
#
# Аргументы:
#   1. путь к .osm.pbf
#   2. человекочитаемое имя источника (для журнала osm_import_log)
#   3. дата актуальности данных источника, ISO 8601 (например дата выгрузки Geofabrik)
#
# Переменные окружения подключения — как в geo/src/topology_geo/devcheck.py
# (POSTGRES_HOST/PORT/DB/USER/PASSWORD), чтобы использовать один и тот же
# .env что и остальной конвейер (infra/.env.example).

set -euo pipefail

PBF_PATH="${1:?укажите путь к .osm.pbf}"
SOURCE_NAME="${2:?укажите имя источника}"
DATA_TIMESTAMP="${3:?укажите дату актуальности данных (ISO 8601)}"

POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-topology}"
POSTGRES_USER="${POSTGRES_USER:-topology}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STYLE_LUA="${SCRIPT_DIR}/../osm2pgsql/style.lua"

echo "Импорт ${PBF_PATH} (${SOURCE_NAME}, данные на ${DATA_TIMESTAMP}) в ${POSTGRES_DB}@${POSTGRES_HOST}:${POSTGRES_PORT}"

osm2pgsql \
  --output=flex \
  --style="${STYLE_LUA}" \
  --slim \
  --host="${POSTGRES_HOST}" \
  --port="${POSTGRES_PORT}" \
  --database="${POSTGRES_DB}" \
  --user="${POSTGRES_USER}" \
  "${PBF_PATH}"

python3 -m topology_geo.osm.cli record-import \
  --source-name "${SOURCE_NAME}" \
  --source-file "${PBF_PATH}" \
  --data-timestamp "${DATA_TIMESTAMP}"

echo "Готово."
