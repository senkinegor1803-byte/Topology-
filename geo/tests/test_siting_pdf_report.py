"""Тесты Шага 3.9, п. 3: PDF-отчёт посадки — реальный разбор структуры PDF
(число страниц, наличие текста), не только магический заголовок."""

from __future__ import annotations

from pypdf import PdfReader
import io

from topology_geo.siting.checks import Collision
from topology_geo.siting.pdf_report import SourceEntry, build_placement_report_pdf


def _reader(pdf_bytes: bytes) -> PdfReader:
    return PdfReader(io.BytesIO(pdf_bytes))


def test_build_placement_report_pdf_has_valid_header():
    pdf_bytes = build_placement_report_pdf(
        site_name="Тестовый участок", design_elevation_m=155.5, footprint_area_m2=1200.0,
        cut_m3=100.0, fill_m3=50.0, collisions=[], sources=[],
    )
    assert pdf_bytes[:4] == b"%PDF"


def test_build_placement_report_pdf_contains_key_figures_in_text():
    pdf_bytes = build_placement_report_pdf(
        site_name="ЖК Рябиновый", design_elevation_m=155.50, footprint_area_m2=1234.5,
        cut_m3=321.0, fill_m3=654.0, collisions=[], sources=[],
    )

    reader = _reader(pdf_bytes)
    text = reader.pages[0].extract_text()

    assert "ЖК Рябиновый" in text
    assert "155.50" in text
    assert "1234.5" in text
    assert "321.0" in text
    assert "654.0" in text


def test_build_placement_report_pdf_lists_collisions():
    collisions = [
        Collision(check_type="ЗОУИТ", description="пересечение с водоохранной зоной", object_refs=["b1"]),
    ]
    pdf_bytes = build_placement_report_pdf(
        site_name="Участок", design_elevation_m=100.0, footprint_area_m2=500.0,
        cut_m3=0.0, fill_m3=0.0, collisions=collisions, sources=[],
    )

    text = "".join(page.extract_text() for page in _reader(pdf_bytes).pages)
    assert "ЗОУИТ" in text
    assert "водоохранной зоной" in text


def test_build_placement_report_pdf_lists_sources_with_dates():
    sources = [SourceEntry(name="НСПД, выгрузка ЗОУИТ", data_timestamp="2026-03-01")]
    pdf_bytes = build_placement_report_pdf(
        site_name="Участок", design_elevation_m=100.0, footprint_area_m2=500.0,
        cut_m3=0.0, fill_m3=0.0, collisions=[], sources=sources,
    )

    text = "".join(page.extract_text() for page in _reader(pdf_bytes).pages)
    assert "НСПД, выгрузка ЗОУИТ" in text
    assert "2026-03-01" in text


def test_build_placement_report_pdf_no_collisions_says_so():
    pdf_bytes = build_placement_report_pdf(
        site_name="Участок", design_elevation_m=100.0, footprint_area_m2=500.0,
        cut_m3=0.0, fill_m3=0.0, collisions=[], sources=[],
    )

    text = "".join(page.extract_text() for page in _reader(pdf_bytes).pages)
    assert "Коллизий не найдено" in text
