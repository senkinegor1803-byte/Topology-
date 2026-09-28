#!/usr/bin/env bash
# Векторные тайлы OSM в PMTiles через Planetiler (Шаг 2.12, п. 1).
#
# Использование:
#   ./build_citymap_vector_tiles.sh path/to/perm.osm.pbf path/to/output.pmtiles
#
# Аргументы:
#   1. путь к .osm.pbf (Planetiler не понимает .osm XML напрямую — если
#      источник в XML, сконвертировать через `osmium cat in.osm -o out.osm.pbf`)
#   2. путь к выходному .pmtiles
#
# Переменные окружения:
#   PLANETILER_JAR — путь к planetiler.jar (по умолчанию ./planetiler.jar
#     рядом со скриптом; скачать: https://github.com/onthegomap/planetiler/
#     releases/latest/download/planetiler.jar)
#
# Профиль — встроенный OpenMapTiles (открытая, широко используемая схема
# слоёв: building/transportation/water/landcover/place/poi и т.д., та же,
# что у большинства публичных OSM-карт) — свой профиль не пишется: критерий
# шага («поднятые здания, стили дорог по классам, вода, зелень») это уже
# покрывает готовыми слоями схемы, изобретать собственную не нужно.
#
# Реальный прогон (Шаг 2.12, honestly) проверен на небольшом тестовом
# фрагменте реальных данных Перми (~1 км² вокруг точки), не на выгрузке
# всего города — сетевая политика этой песочницы блокирует Geofabrik/
# Overpass (см. docs/citymap.md), которые нужны для регулярной
# («еженедельная пересборка», действие плана) автоматической выгрузки
# города целиком; сам скрипт не завязан на конкретный источник и отработает
# на любом реальном .osm.pbf с доступом к сети без изменений.

set -euo pipefail

OSM_PBF_PATH="${1:?укажите путь к .osm.pbf}"
OUTPUT_PMTILES="${2:?укажите путь к выходному .pmtiles}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANETILER_JAR="${PLANETILER_JAR:-${SCRIPT_DIR}/planetiler.jar}"

if [ ! -f "${PLANETILER_JAR}" ]; then
  echo "planetiler.jar не найден по ${PLANETILER_JAR}" >&2
  echo "скачать: curl -sSL -o \"${PLANETILER_JAR}\" https://github.com/onthegomap/planetiler/releases/latest/download/planetiler.jar" >&2
  exit 1
fi

echo "Сборка ${OUTPUT_PMTILES} из ${OSM_PBF_PATH} (Planetiler, профиль OpenMapTiles)"

java -jar "${PLANETILER_JAR}" \
  --osm_path="${OSM_PBF_PATH}" \
  --output="${OUTPUT_PMTILES}" \
  --download \
  --force

echo "Готово."
