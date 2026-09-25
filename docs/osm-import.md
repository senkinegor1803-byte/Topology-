# Импорт OSM в PostGIS (Шаг 1.1)

## Таблицы

Флекс-стиль `geo/osm2pgsql/style.lua` раскладывает объекты по слоям (все теги
объекта целиком — в jsonb `tags`, привязка к словарю данных из
`docs/data-dictionary.md` делается позже, на выборке участка — Шаг 1.4, не
здесь, чтобы смена правил соответствия не требовала перезаливки всего края):

| Таблица | Геометрия | Что попадает |
| --- | --- | --- |
| `osm_buildings` | polygon/multipolygon | `building=*` (way, и multipolygon-отношения) |
| `osm_roads` | linestring | `highway=*` |
| `osm_railways` | linestring | `railway=*` |
| `osm_water_areas` | polygon/multipolygon | `natural=water`, `landuse=reservoir` |
| `osm_waterways` | linestring | `waterway=*` |
| `osm_vegetation` | точки + полигоны | `natural=tree` (точки), `natural=wood`/`landuse=forest,grass` (полигоны) |
| `osm_power` | точки + линии + полигоны | `power=pole/tower/substation` (точки), `power=line/minor_line` (линии), `power=substation/plant` (полигоны) |
| `osm_landscaping` | точки + полигоны | скамейки, урны, фонари (`highway=street_lamp`), ограждения, площадки, парки |
| `osm_import_log` | — | служебный журнал: источник, дата данных, дата импорта (Шаг 1.1, п. 4) |

Каждая геометрическая таблица получает GIST-индекс на `geom` автоматически
(поведение osm2pgsql flex по умолчанию — проверено локальным прогоном, не
нужно создавать вручную).

Известный краевой случай: если внешний way мультиполигон-отношения сам несёт
тот же тег (например `building=yes`) не только на отношении, но и на самом
way, он может попасть в таблицу дважды (как way и как часть отношения) — это
общепринятый в примерах osm2pgsql компромисс для этого уровня детализации;
устраняется дедупликацией по `osm_id`/`osm_type` при выборке (Шаг 1.4), если
станет заметно на реальных данных.

## Как запустить

Полная (пере-)загрузка выгрузки региона:

```bash
./geo/scripts/import_osm.sh privolzhsky-fed-district-latest.osm.pbf \
    "Geofabrik Приволжский ФО" 2026-09-20
```

Записывает факт импорта в `osm_import_log` (дата данных источника, а не дата
запуска импорта — это две разные вещи, важно для мониторинга отставания).

## Обновление (Шаг 1.1, п. 3)

Два варианта, выбор — по факту нагрузки на этапе 1.10/2.13:

1. **Еженедельная полная перезаливка** — повторный запуск `import_osm.sh` с
   свежим `.pbf`. Проще эксплуатационно, но требует полного времени импорта
   каждую неделю.
2. **Минутные дифы через `osm2pgsql-replication`:**

   ```bash
   # один раз после первой полной загрузки
   ./geo/scripts/update_osm_replication.sh init privolzhsky-fed-district-latest.osm.pbf \
       https://download.geofabrik.de/russia/privolzhskiy-fed-district-updates/

   # по расписанию
   ./geo/scripts/update_osm_replication.sh update
   ```

   Пример systemd-таймера (предпочтительнее cron на проде — есть журнал и
   защита от наложения запусков):

   ```ini
   # /etc/systemd/system/topology-osm-update.service
   [Service]
   Type=oneshot
   EnvironmentFile=/opt/topology/infra/.env
   ExecStart=/opt/topology/geo/scripts/update_osm_replication.sh update

   # /etc/systemd/system/topology-osm-update.timer
   [Timer]
   OnCalendar=*:0/1
   Persistent=true

   [Install]
   WantedBy=timers.target
   ```

   Эквивалент на cron: `* * * * * /opt/topology/geo/scripts/update_osm_replication.sh update`.

Эта среда разработки не держит постоянного демона/крона — таймер выше
проверяется на реальном сервере (Шаг 5.1), здесь проверен только сам прогон
`osm2pgsql`/скриптов вручную.

## Проверка (критерий Шага 1.1)

- Выборка объектов в круге 3 км ≤ 5 с — `topology_geo.osm.queries.count_within_radius`;
  сам запрос и его корректность проверены тестами на синтетических данных
  (`geo/tests/test_osm_import_style.py`), время на реальном объёме края
  измеряется на Шаге 1.10 (нужна настоящая загруженная база, а не пилотная
  синтетика).
- Число объектов сходится с аудитом Шага 0.6 — сравнение делается после того,
  как выбран пилотный участок (Шаг 0.1) и посчитан реальный аудит.
