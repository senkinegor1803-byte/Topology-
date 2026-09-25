#!/usr/bin/env bash
# Инкрементальное обновление через osm2pgsql-replication (Шаг 1.1, п. 3,
# вариант "минутные дифы"). Для варианта "еженедельно новая выгрузка"
# используется import_osm.sh поверх свежего .osm.pbf (полная переливка).
#
# Использование:
#   ./update_osm_replication.sh init path/to/privolzhsky-fed-district-latest.osm.pbf \
#       https://download.geofabrik.de/russia/privolzhskiy-fed-district-updates/
#   ./update_osm_replication.sh update
#
# `init` запускается один раз после первой полной загрузки (import_osm.sh) —
# он находит по .pbf, на каком моменте репликации база уже находится.
# `update` запускается по расписанию (см. docs/osm-import.md — там же пример
# systemd-таймера/cron; в этой среде разработки нет постоянного крона, чтобы
# держать демон, поэтому здесь только сама команда).

set -euo pipefail

MODE="${1:?укажите режим: init | update}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STYLE_LUA="${SCRIPT_DIR}/../osm2pgsql/style.lua"

POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-topology}"
POSTGRES_USER="${POSTGRES_USER:-topology}"

case "$MODE" in
  init)
    PBF_PATH="${2:?укажите путь к исходному .osm.pbf}"
    SERVER_URL="${3:?укажите URL сервера обновлений (Geofabrik updates/)}"
    osm2pgsql-replication init \
      --host="${POSTGRES_HOST}" --port="${POSTGRES_PORT}" \
      --database="${POSTGRES_DB}" --username="${POSTGRES_USER}" \
      --osm-file="${PBF_PATH}" \
      --server="${SERVER_URL}"
    ;;
  update)
    osm2pgsql-replication update \
      --host="${POSTGRES_HOST}" --port="${POSTGRES_PORT}" \
      --database="${POSTGRES_DB}" --username="${POSTGRES_USER}" \
      -- --output=flex --style="${STYLE_LUA}" --slim
    ;;
  *)
    echo "Неизвестный режим: ${MODE} (ожидается init|update)" >&2
    exit 1
    ;;
esac
