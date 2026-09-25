# Таблица совместимости IFC (Шаг 0.2)

Статус: **шаблон, не заполнен**. Заполняется BIM-специалистом вручную после
открытия файлов из `geo/src/topology_geo/ifc/generate_test_ifc.py` в Renga,
Pilot-BIM и свободном вьюере — этого AI-сессия сделать не может (нет доступа к
этим программам).

## Как сгенерировать тестовые файлы

```bash
cd geo
pip install -e .
python -m topology_geo.ifc.generate_test_ifc --schema both --out out --validate
```

Файлы `out/test_ifc4.ifc` и `out/test_ifc4x3.ifc` содержат: рельеф-TIN
(`IfcGeographicElement`, TERRAIN), `IfcRoad`/`IfcBridge` (только в IFC4X3 — в
IFC4 их заменяет `IfcBuildingElementProxy` с пометкой `Заменяет_класс` в
`Pset_Контекст`), `IfcBuildingElementProxy`, `IfcSpatialZone`,
`IfcPipeSegment`, дерево (`IfcGeographicElement`). На каждом объекте — наборы
свойств с кириллицей `Pset_Контекст`, `Pset_Источник`, `Pset_Ограничение`.
Геопривязка задана через `IfcMapConversion` в условных крупных координатах
МСК-59 (уточнить на реальные после Шага 0.1).

Оба файла уже прошли `ifcopenshell.validate` без замечаний (см.
`geo/tests/test_ifc_generate.py`) — то есть они синтаксически корректны;
остаётся открыть их в целевых программах.

## Таблица «класс IFC × программа»

Заполнить: читается / открывается как прокси / не читается; для «прокси» —
как называется класс на стороне программы.

| Класс IFC | Renga (IFC4) | Renga (IFC 4.3) | Pilot-BIM | Свободный вьюер |
| --- | --- | --- | --- | --- |
| `IfcGeographicElement` (TERRAIN) | | | | |
| `IfcRoad` | н/п в IFC4 | | | |
| `IfcBridge` | н/п в IFC4 | | | |
| `IfcBuildingElementProxy` | | | | |
| `IfcSpatialZone` | | | | |
| `IfcPipeSegment` | | | | |
| Pset с кириллицей | | | | |
| `IfcMapConversion` (геопривязка) | | | | |

## Нагрузочный тест (п. 5 Шага 0.2)

Сгенерировать увеличенные файлы и замерить время открытия/отзывчивость:

```bash
python -m topology_geo.ifc.generate_test_ifc --schema IFC4X3 --out out --relief-grid 60 --extra-buildings 500
```

Подобрать `--relief-grid` и `--extra-buildings` опытным путём под целевые
50 / 200 / 1000 МБ (в этой среде разработки такая генерация не запускалась —
большие файлы и открытие в Renga/Pilot-BIM возможны только на рабочей
станции с этими программами).

| Целевой размер | Файл | Фактический размер | Время открытия (Renga) | Время открытия (Pilot-BIM) |
| --- | --- | --- | --- | --- |
| 50 МБ | | | | |
| 200 МБ | | | | |
| 1000 МБ | | | | |

## Проверка (критерий закрытия Шага 0.2)

- [ ] Таблица выше заполнена по всем классам.
- [ ] Выбран формат выгрузки для Renga: IFC 4.3 или IFC4 с упрощением.
- [ ] Зафиксирован максимальный размер одного файла.

Решение и обоснование записать здесь после заполнения таблиц.
