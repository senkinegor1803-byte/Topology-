# Топология

Комплекс для автоматизированного построения BIM/IFC-модели окружения (рельеф,
здания, дороги, сети, ограничения) вокруг проектируемого ЖК по координатам
участка, с публичной картой города, посадкой ЖК, проверкой ограничений,
формированием запросов ТУ и фотореалистичной визуализацией.

Полный план реализации — в [docs/plan.md](docs/plan.md) (6 этапов, ~50 шагов).
Текущий статус по каждому шагу — в [STATUS.md](STATUS.md) и подробнее в
[docs/dev-tree.md](docs/dev-tree.md) (дерево этапов/шагов со ссылками на код —
чтобы всегда можно было вернуться к конкретному шагу). Формулы и алгоритмы,
на которых строится конвейер — в [docs/math-model.md](docs/math-model.md).

## Структура репозитория

```
geo/            Python-пакет конвейера геоданных (PostGIS, IFC, OSM, рельеф)
  osm2pgsql/style.lua  флекс-стиль импорта OSM (Шаг 1.1)
  sql/             SQL-миграции (служебные таблицы)
  scripts/         обёртки импорта/обновления OSM
  src/topology_geo/
    coords.py       пересчёт координат и высот (МСК-59, EGM96 -> Балтийская)
    ifc/             генерация, сборка (site.ifc), IFC -> GLB, валидация, реестр GlobalId
    geometry/        здания/дороги/вода/рельсы/деревья участка (Шаги 1.6-1.7)
    web/viewer/      статическая страница-вьюер (three.js, Шаг 1.9)
    osm/             импорт, журнал источников, запросы, аудит полноты OSM
    relief/          репроекция, высоты, COG, слияние, покрытие, get_dem, TIN участка
    selection/       выборка, обрезка, нормализация, GeoPackage (Шаг 1.4)
    jobs/            модель задач, движок пайплайна, реальные шаги (Шаг 1.3)
    tasks/           очередь Celery + Redis (Шаг 1.3)
    api/             FastAPI: POST /jobs, GET /jobs/{id}, GET /models/{id}/files
    storage.py       объектное хранилище (MinIO/файловое/в памяти)
  tests/            модульные и интеграционные тесты (pytest)
infra/          Docker Compose для локальной разработки (PostGIS, MinIO, Redis)
docs/           план, словарь данных, системы координат, прочая документация
schemas/        JSON-схемы (meta.json и др.)
.github/workflows/  CI (lint + тесты)
```

## Быстрый старт (разработка)

```bash
cd infra
cp .env.example .env
docker compose up -d
```

```bash
cd geo
pip install -e ".[dev]"
pytest
```

Импорт OSM в PostGIS (Шаг 1.1) — см. [docs/osm-import.md](docs/osm-import.md)
(`geo/scripts/import_osm.sh`, флекс-стиль `geo/osm2pgsql/style.lua`).
Подготовка рельефа (Шаг 1.2, `get_dem(bbox)`) — см.
[docs/relief.md](docs/relief.md). API и очередь задач (Шаг 1.3,
FastAPI + Celery/Redis) — см. [docs/api.md](docs/api.md):

```bash
uvicorn topology_geo.api.app:app --reload   # API
celery -A topology_geo.tasks.celery_app worker --loglevel=info  # воркер
```

Выборка и нормализация данных участка (Шаг 1.4) — см.
[docs/selection.md](docs/selection.md). Протокол приёмки Этапа 1 (Шаг 1.10,
сквозной тест на 3 профилях участка) — см.
[docs/stage1-acceptance.md](docs/stage1-acceptance.md).

Интеграционные тесты Шагов 1.1-1.10 требуют системный `osm2pgsql`
(`apt-get install osm2pgsql`), доступные Postgres+PostGIS и Redis, и Chromium
для теста вьюера (`playwright install chromium`) — без них соответствующие
тесты пропускаются, но реально прогоняются в CI.

## Роли и участие AI

Проект ведётся по плану в `docs/plan.md`. Роль AI (Claude) в плане ограничена
кодом, тестами, разбором данных и черновиками документации — под контролем
разработчика; решения, приёмка и взаимодействие с организациями остаются за
человеком (РП, BIM, ЮР, ИЗ). См. STATUS.md — там явно отмечено, какие шаги
требуют действий вне кода (выбор пилотного участка, запросы в организации,
юридическая оценка, топосъёмка, закупка оборудования, проверка в Renga /
Pilot-BIM) и не могут быть закрыты одним AI-сеансом.
