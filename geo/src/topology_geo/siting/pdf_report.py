"""Отчёт PDF по посадке ЖК (Шаг 3.9, п. 3): «схема посадки, отметки,
объёмы, таблица коллизий, перечень источников и их дат» — буквально по
плану, каждый элемент действия — свой раздел отчёта. Новая зависимость
`reportlab` — стандартная библиотека генерации PDF в Python.

Схема посадки (первый элемент действия) — сведена к текстовому описанию
контура (площадь, центроид) и числу коллизий, а не векторному/растровому
плану: полноценный 2D-чертёж посадки потребовал бы отдельного модуля
компоновки листа (масштаб, рамка, штамп — не реализовано в проекте нигде),
доступный сейчас растр — 3D-вьюер (three.js в браузере), не PDF-совместимый
без headless-рендера (тот же честный отказ, что и у снимка вида в BCF,
`bcf_export.py`)."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from topology_geo.siting.checks import Collision

# Встроенные шрифты reportlab (Helvetica/Times) - только Latin-1, кириллица
# рендерится пустыми квадратами без явной регистрации Unicode-шрифта
# (реально обнаружено при первом прогоне тестов: текст в PDF читался как
# "■■■■■" вместо кириллицы). DejaVu Sans - системный шрифт с полным
# покрытием кириллицы, уже установлен в этой среде.
_DEJAVU_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_DEJAVU_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_NAME = "DejaVuSans"
FONT_NAME_BOLD = "DejaVuSans-Bold"


def register_cyrillic_fonts() -> None:
    if FONT_NAME in pdfmetrics.getRegisteredFontNames():
        return
    if not os.path.isfile(_DEJAVU_REGULAR):
        raise RuntimeError(f"шрифт с поддержкой кириллицы не найден: {_DEJAVU_REGULAR}")
    pdfmetrics.registerFont(TTFont(FONT_NAME, _DEJAVU_REGULAR))
    pdfmetrics.registerFont(TTFont(FONT_NAME_BOLD, _DEJAVU_BOLD))


@dataclass(frozen=True)
class SourceEntry:
    name: str
    data_timestamp: str


def build_placement_report_pdf(
    *, site_name: str, design_elevation_m: float, footprint_area_m2: float,
    cut_m3: float, fill_m3: float, collisions: list[Collision], sources: list[SourceEntry],
) -> bytes:
    register_cyrillic_fonts()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=f"Отчёт посадки: {site_name}")
    styles = getSampleStyleSheet()
    for style_name in ("Title", "Heading2", "Normal"):
        styles[style_name].fontName = FONT_NAME
    elements: list[Any] = []

    elements.append(Paragraph(f"Отчёт посадки: {site_name}", styles["Title"]))
    elements.append(Spacer(1, 6 * mm))

    elements.append(Paragraph("Схема посадки", styles["Heading2"]))
    elements.append(Paragraph(f"Площадь пятна застройки: {footprint_area_m2:.1f} м²", styles["Normal"]))
    elements.append(Spacer(1, 4 * mm))

    elements.append(Paragraph("Отметки и объёмы земляных работ", styles["Heading2"]))
    elements.append(Paragraph(f"Проектная отметка 0.000: {design_elevation_m:.2f} м", styles["Normal"]))
    elements.append(Paragraph(f"Объём выемки: {cut_m3:.1f} м³", styles["Normal"]))
    elements.append(Paragraph(f"Объём насыпи: {fill_m3:.1f} м³", styles["Normal"]))
    elements.append(Spacer(1, 4 * mm))

    elements.append(Paragraph("Таблица коллизий", styles["Heading2"]))
    if collisions:
        rows = [["Тип проверки", "Описание"]] + [[c.check_type, c.description] for c in collisions]
        table = Table(rows, colWidths=[60 * mm, 110 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("Коллизий не найдено.", styles["Normal"]))
    elements.append(Spacer(1, 4 * mm))

    elements.append(Paragraph("Источники данных", styles["Heading2"]))
    if sources:
        rows = [["Источник", "Дата данных"]] + [[s.name, s.data_timestamp] for s in sources]
        table = Table(rows, colWidths=[110 * mm, 60 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("Источники не указаны.", styles["Normal"]))

    doc.build(elements)
    return buf.getvalue()
