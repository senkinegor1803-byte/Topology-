"""Тесты Шага 3.4: импорт подземных сетей из DXF (`ezdxf`, уже зависимость
проекта с Шага 2.10 — там на запись, здесь на чтение). Фикстура — реальный
DXF, построенный тем же `ezdxf` (тот же приём, что `test_export_dxf.py`:
собрать через API библиотеки и прочитать обратно своим кодом), не запись
байтов вручную."""

from __future__ import annotations

import ezdxf
import pytest

from topology_geo.networks.dxf_import import (
    extract_diameter_mm,
    extract_elevation_m,
    extract_material,
    import_dxf,
    match_layer_to_network_type,
)


@pytest.mark.parametrize("layer_name,expected", [
    ("К1", "К"), ("к2", "К"), ("В1", "В"), ("Т1", "Т"), ("Г", "Г"), ("Кл", "Кл"), ("кл3", "Кл"),
    ("Канализация", "К"), ("Водопроводная сеть", "В"), ("Теплотрасса", "Т"),
    ("Газопровод низкого давления", "Г"), ("Кабельная линия 10кВ", "Кл"),
    ("СТЕНЫ", None), ("0", None), ("XREF_topo", None),
])
def test_match_layer_to_network_type(layer_name, expected):
    assert match_layer_to_network_type(layer_name) == expected


@pytest.mark.parametrize("text,expected_mm", [
    ("Ø300 чугун", 300.0), ("d=150 ПНД", 150.0), ("D=200", 200.0), ("труба d500", 500.0), ("без диаметра", None),
])
def test_extract_diameter_mm(text, expected_mm):
    assert extract_diameter_mm(text) == expected_mm


@pytest.mark.parametrize("text,expected", [
    ("Ø300 чугун", "чугун"), ("труба ПНД d110", "пнд"), ("железобетонный коллектор", "железобетон"), ("нет материала", None),
])
def test_extract_material(text, expected):
    assert extract_material(text) == expected


@pytest.mark.parametrize("text,expected_m", [
    ("отм.=95.50", 95.50), ("отм 12,30", 12.30), ("z=-2.5", -2.5), ("без отметки", None),
])
def test_extract_elevation_m(text, expected_m):
    assert extract_elevation_m(text) == expected_m


def _build_fixture_dxf(path) -> None:
    doc = ezdxf.new()
    msp = doc.modelspace()

    msp.add_lwpolyline([(0, 0), (100, 0)], dxfattribs={"layer": "К1"})
    msp.add_text("Ø300 чугун отм.=95.50", dxfattribs={"layer": "К1", "insert": (50, 0)}).dxf.insert = (50, 0)

    msp.add_line((0, 20), (100, 20), dxfattribs={"layer": "Водопровод"})
    msp.add_text("D=150 ПНД", dxfattribs={"layer": "Водопровод", "insert": (50, 20)}).dxf.insert = (50, 20)

    # Без подписи рядом - отметка должна стать нормативной (п. 4)
    msp.add_line((0, 40), (100, 40), dxfattribs={"layer": "Т1"})

    # Нестандартный слой - не должен сопоставиться ни с одним типом сети
    msp.add_line((0, 60), (100, 60), dxfattribs={"layer": "ПРОЧЕЕ"})

    doc.blocks.new(name="KOLODEC")
    msp.add_blockref("KOLODEC", insert=(50, 0), dxfattribs={"layer": "К1"})

    doc.saveas(path)


def test_import_dxf_real_fixture(tmp_path):
    dxf_path = tmp_path / "topoplan.dxf"
    _build_fixture_dxf(dxf_path)

    result = import_dxf(str(dxf_path))

    assert len(result.segments) == 4
    by_layer = {s.layer: s for s in result.segments}

    k1 = by_layer["К1"]
    assert k1.network_type == "К"
    assert k1.diameter_mm == 300.0
    assert k1.material == "чугун"
    assert k1.elevation_m == pytest.approx(95.50)
    assert k1.elevation_source == "факт"

    water = by_layer["Водопровод"]
    assert water.network_type == "В"
    assert water.diameter_mm == 150.0
    assert water.material == "пнд"
    # Подпись "D=150 ПНД" не несёт отметки - честная нормативная глубина (п. 4)
    assert water.elevation_source == "нормативная"
    assert water.elevation_m == pytest.approx(-1.8)  # NORMATIVE_DEPTH_M_BY_NETWORK_TYPE["В"]

    heat = by_layer["Т1"]
    assert heat.network_type == "Т"
    assert heat.elevation_source == "нормативная"
    assert heat.elevation_m == pytest.approx(-0.7)  # NORMATIVE_DEPTH_M_BY_NETWORK_TYPE["Т"]

    other = by_layer["ПРОЧЕЕ"]
    assert other.network_type is None

    assert result.unmatched_layers == {"ПРОЧЕЕ"}

    assert len(result.manholes) == 1
    assert result.manholes[0].block_name == "KOLODEC"
    assert result.manholes[0].x == 50
    assert result.manholes[0].y == 0
