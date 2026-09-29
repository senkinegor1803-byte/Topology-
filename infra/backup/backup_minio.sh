#!/usr/bin/env bash
# Резервное копирование объектного хранилища MinIO (Шаг 5.1, п. 2-3) —
# зеркалирование бакета через официальный клиент `mc` (MinIO Client).
#
# ЧЕСТНО: этот скрипт НЕ выполнялся и не проверялся в этой среде — здесь
# нет запущенного MinIO (нет демона Docker, см. `docs/dev-tree.md`,
# Шаг 4.5) и не установлен `mc`. Логика (алиас + `mc mirror`) —
# стандартная, документированная самим MinIO, но без реального прогона
# это честно не то же самое, что проверенный `backup_postgres.sh`/
# `restore_postgres.sh` (оба реально выполнены и покрыты тестом
# `geo/tests/test_ops_backup_restore.py`).
#
# Переменные окружения: MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
# MINIO_BUCKET (бакет с результатами задач, по умолчанию "topology"),
# BACKUP_DIR (каталог для зеркала, по умолчанию ./backups/minio).
set -euo pipefail

MINIO_ENDPOINT="${MINIO_ENDPOINT:?переменная MINIO_ENDPOINT обязательна}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:?переменная MINIO_ACCESS_KEY обязательна}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:?переменная MINIO_SECRET_KEY обязательна}"
MINIO_BUCKET="${MINIO_BUCKET:-topology}"
BACKUP_DIR="${BACKUP_DIR:-./backups/minio}"

mkdir -p "$BACKUP_DIR"

mc alias set topology-backup-source "http://$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
mc mirror --overwrite "topology-backup-source/$MINIO_BUCKET" "$BACKUP_DIR"

echo "Зеркало бакета «$MINIO_BUCKET» обновлено в $BACKUP_DIR"
