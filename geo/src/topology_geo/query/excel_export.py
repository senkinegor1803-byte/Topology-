"""Выгрузка результата поиска в Excel (Шаг 3.5, п. 3). Новая зависимость
`openpyxl` — стандартная, самая распространённая библиотека для записи
`.xlsx` в Python, требование плана буквально называет Excel, не просто
табличный CSV."""

from __future__ import annotations

import io

from openpyxl import Workbook

from topology_geo.constraints.store import ConstraintZone

COLUMNS = ["Вид", "Статус", "Реестровый номер", "Режим", "Документ-основание", "Источник", "Дата данных"]


def zones_to_excel(zones: list[ConstraintZone]) -> bytes:
    """Строит `.xlsx` в памяти (не на диск - тот же приём, что `export.
    dxf.build_dxf`/`export.cityjson`, Шаг 2.10: функция возвращает байты,
    вызывающая сторона решает, куда их положить — Storage, HTTP-ответ,
    файл)."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Ограничения"
    ws.append(COLUMNS)
    for zone in zones:
        ws.append([
            zone.zone_type, zone.status, zone.registry_number or "", zone.regime or "",
            zone.document_basis or "", zone.source_name, zone.data_timestamp.isoformat(),
        ])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
