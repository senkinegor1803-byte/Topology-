"""Шаблоны и формирование запросов ТУ (Шаг 3.11).

Реальный шаблон организации (в фирменном стиле) в этой среде недоступен —
план явно предполагает шаблон СВОЙ у каждой РСО, полученный от пользователя
(«после получения форм от пользователя», срок шага в плане). `build_
default_template` строит МИНИМАЛЬНЫЙ, но настоящий `.docx` с теми же
плейсхолдерами, что `fill_docx_template` умеет заполнять — честная
заглушка для примера/тестов, не притворство, что это реальный фирменный
бланк.

Действие п. 4 «отправка только после проверки человеком» реализовано
ОТСУТСТВИЕМ функции автоотправки: модуль строит пакет файлов и ничего
никуда не отправляет — ни одной сетевой интеграции с порталами РСО в
проекте нет (да и не может быть без реальных РСО-порталов пилота).
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field

import docx

from topology_geo.siting.pdf_report import FONT_NAME, register_cyrillic_fonts

PLACEHOLDERS = ("заявитель", "кадастровый_номер", "адрес", "назначение_объекта", "нагрузки", "сроки")


@dataclass(frozen=True)
class TuRequestFields:
    """Действие п. 1: «разобрать формы запросов пользователя на поля»."""

    applicant: str
    cadastral_number: str
    address: str
    object_purpose: str
    loads: dict[str, str] = field(default_factory=dict)  # тип сети -> "значение, ед. изм."
    deadlines: str = ""

    def as_replacements(self) -> dict[str, str]:
        loads_text = "; ".join(f"{network_type}: {value}" for network_type, value in self.loads.items())
        return {
            "{{заявитель}}": self.applicant,
            "{{кадастровый_номер}}": self.cadastral_number,
            "{{адрес}}": self.address,
            "{{назначение_объекта}}": self.object_purpose,
            "{{нагрузки}}": loads_text,
            "{{сроки}}": self.deadlines,
        }


def build_default_template() -> bytes:
    doc = docx.Document()
    doc.add_heading("Заявка на технические условия", level=1)
    doc.add_paragraph("Заявитель: {{заявитель}}")
    doc.add_paragraph("Кадастровый номер участка: {{кадастровый_номер}}")
    doc.add_paragraph("Адрес: {{адрес}}")
    doc.add_paragraph("Назначение объекта: {{назначение_объекта}}")
    doc.add_paragraph("Требуемые нагрузки: {{нагрузки}}")
    doc.add_paragraph("Планируемые сроки: {{сроки}}")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _replace_in_paragraph(paragraph, replacements: dict[str, str]) -> None:
    """Плейсхолдер в реальном `.docx` часто разбит на несколько `run`
    (Word хранит форматирование по фрагментам текста, даже если весь
    текст выглядит одинаково) — прямая замена внутри одного `run.text`
    находит плейсхолдер не всегда; здесь текст параграфа собирается
    целиком, заменяется, и записывается обратно в первый `run`, остальные
    очищаются — реальная, документированная особенность python-docx, не
    гипотетическая."""
    full_text = "".join(run.text for run in paragraph.runs)
    if not any(ph in full_text for ph in replacements):
        return
    for placeholder, value in replacements.items():
        full_text = full_text.replace(placeholder, value)
    if not paragraph.runs:
        return
    paragraph.runs[0].text = full_text
    for run in paragraph.runs[1:]:
        run.text = ""


def fill_docx_template(template_bytes: bytes, fields: TuRequestFields) -> bytes:
    """Действие п. 2: «заполнение DOCX в стиле шаблона»."""
    doc = docx.Document(io.BytesIO(template_bytes))
    replacements = fields.as_replacements()

    for paragraph in doc.paragraphs:
        _replace_in_paragraph(paragraph, replacements)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    _replace_in_paragraph(paragraph, replacements)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_site_excerpt_pdf(*, site_name: str, footprint_area_m2: float, network_summary: list[str]) -> bytes:
    """Действие п. 2: «приложение — выкопировка из модели со схемой сетей и
    участка (PDF)» — переиспользует регистрацию кириллического шрифта из
    `siting.pdf_report` (та же реальная находка про Helvetica без
    кириллицы), не дублирует её."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    register_cyrillic_fonts()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=f"Выкопировка: {site_name}")
    styles = getSampleStyleSheet()
    for style_name in ("Title", "Heading2", "Normal"):
        styles[style_name].fontName = FONT_NAME

    elements = [
        Paragraph(f"Выкопировка из модели: {site_name}", styles["Title"]),
        Spacer(1, 4 * mm),
        Paragraph(f"Площадь участка: {footprint_area_m2:.1f} м²", styles["Normal"]),
        Spacer(1, 4 * mm),
        Paragraph("Сети рядом с участком", styles["Heading2"]),
    ]
    for line in network_summary or ["Сетей рядом с участком не найдено."]:
        elements.append(Paragraph(line, styles["Normal"]))

    doc.build(elements)
    return buf.getvalue()


def build_tu_request_package(
    *, fields: TuRequestFields, docx_bytes: bytes, excerpt_pdf_bytes: bytes,
) -> bytes:
    """Действие п. 3: «для порталов — комплект данных и вложений для
    ручной подачи» — zip с заполненным DOCX, PDF-выкопировкой и
    `fields.json` (структурированные поля — для ручного переноса в форму
    портала, где нет загрузки файла целиком, только поля ввода). Тот же
    приём упаковки, что `jobs.steps.package_outputs` (Шаг 2.10) —
    собирается один раз, не переизобретается."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("request.docx", docx_bytes)
        zf.writestr("attachment.pdf", excerpt_pdf_bytes)
        zf.writestr("fields.json", json.dumps({
            "заявитель": fields.applicant, "кадастровый_номер": fields.cadastral_number,
            "адрес": fields.address, "назначение_объекта": fields.object_purpose,
            "нагрузки": fields.loads, "сроки": fields.deadlines,
        }, ensure_ascii=False, indent=2))
    return buf.getvalue()
