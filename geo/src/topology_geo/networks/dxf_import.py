"""Импорт подземных сетей из DXF-топопланов (Шаг 3.4).

Читает DXF через `ezdxf` (уже зависимость проекта — Шаг 2.10, экспорт DXF;
здесь та же библиотека используется на ЧТЕНИЕ). Нативный DWG честно НЕ
читается — `ezdxf` понимает только DXF; конвертер DWG→DXF (ODA File
Converter / `libredwg`) в этой среде не установлен и не может быть
установлен без сетевого доступа к его дистрибутиву — план предполагает
получение топоплана от организации (Шаг 0.3, не реализован в этой
сессии), которая может отдать сразу DXF (большинство САПР умеют
экспортировать DXF из DWG) — тот же класс честного ограничения, что и у
остальных внешних материалов проекта.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

STANDARD_CODE_PATTERN = re.compile(r"^(кл|к|в|т|г)\d*$", re.IGNORECASE)

_DESCRIPTIVE_KEYWORDS: dict[str, set[str]] = {
    "К": {"канализация", "канализационная", "sewer", "sewage"},
    "В": {"водопровод", "водопроводная", "водоснабжение", "water"},
    "Т": {"теплосеть", "теплотрасса", "heating", "heat"},
    "Г": {"газопровод", "газоснабжение", "газ", "gas"},
    "Кл": {"кабель", "кабельная", "электрокабель", "cable"},
}

MATERIAL_KEYWORDS = ["сталь", "чугун", "пнд", "пвх", "полиэтилен", "железобетон", "асбестоцемент", "чугунная"]

_DIAMETER_RE = re.compile(r"[øØ⌀]\s*(\d+)|[dDдД]\s*=\s*(\d+)|\bd(\d+)\b", re.IGNORECASE)
_ELEVATION_RE = re.compile(r"отм\.?\s*=?\s*(-?\d+[.,]\d+)|[zZ]\s*=\s*(-?\d+[.,]\d+)")


def match_layer_to_network_type(layer_name: str) -> str | None:
    """Сопоставление слоя словарю сетей (п. 2) — сначала стандартный код
    (`К1`/`В2`/`Т1`/`Г`/`Кл`, с цифрой или без), затем по описательным
    ключевым словам целыми словами (не подстрокой — иначе однобуквенный
    код «Г» ловил бы что угодно). Несопоставленный слой — `None`, это
    честный сигнал «нестандартный, нужно подтверждение человека» (п. 2,
    вторая часть действия), не тихая догадка."""
    stripped = layer_name.strip()
    m = STANDARD_CODE_PATTERN.match(stripped)
    if m:
        code = m.group(1).lower()
        return "Кл" if code == "кл" else code.upper()

    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", stripped.lower())
    for network_type, keywords in _DESCRIPTIVE_KEYWORDS.items():
        if any(w in keywords for w in words):
            return network_type
    return None


def extract_diameter_mm(text: str) -> float | None:
    m = _DIAMETER_RE.search(text)
    if not m:
        return None
    value = next(g for g in m.groups() if g is not None)
    return float(value)


def extract_material(text: str) -> str | None:
    lowered = text.lower()
    for keyword in MATERIAL_KEYWORDS:
        if keyword in lowered:
            return keyword
    return None


def extract_elevation_m(text: str) -> float | None:
    m = _ELEVATION_RE.search(text)
    if not m:
        return None
    value = next(g for g in m.groups() if g is not None)
    return float(value.replace(",", "."))


@dataclass(frozen=True)
class Manhole:
    block_name: str
    x: float
    y: float
    layer: str


@dataclass(frozen=True)
class NetworkSegment:
    network_type: str | None  # None - нестандартный слой, не сопоставлен (п. 2)
    layer: str
    geometry: Any  # shapely LineString, локальные координаты чертежа
    diameter_mm: float | None
    material: str | None
    elevation_m: float | None
    elevation_source: str  # "факт" (из подписи) | "нормативная" (умолчание, п. 4)


@dataclass(frozen=True)
class DxfImportResult:
    segments: list[NetworkSegment]
    manholes: list[Manhole]
    unmatched_layers: set[str]  # слои без сопоставления сети - нужно подтверждение человека (п. 2)


NORMATIVE_DEPTH_M_BY_NETWORK_TYPE: dict[str, float] = {
    # Глубина заложения по умолчанию (СП 42.13330.2016, представительные
    # значения для средней полосы/Урала - глубина промерзания грунта) -
    # используется, когда отметка не извлечена из подписи (п. 4).
    "К": 1.5, "В": 1.8, "Т": 0.7, "Г": 0.8, "Кл": 0.7,
}


def _nearby_text_value(text_entities: list[tuple[float, float, str]], x: float, y: float, snap_distance: float):
    best = None
    best_dist_sq = snap_distance * snap_distance
    for tx, ty, text in text_entities:
        dist_sq = (tx - x) ** 2 + (ty - y) ** 2
        if dist_sq <= best_dist_sq:
            best_dist_sq = dist_sq
            best = text
    return best


def import_dxf(path: str, *, snap_distance: float = 5.0) -> DxfImportResult:
    """Читает DXF-топоплан (п. 1): слои → тип сети (п. 2), диаметр/
    материал/отметка из ближайших подписей (TEXT/MTEXT, п. 3), колодцы из
    блоков (INSERT, п. 3), нормативная глубина при отсутствии отметки
    (п. 4, `elevation_source="нормативная"`)."""
    import ezdxf
    from shapely.geometry import LineString

    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    text_entities: list[tuple[float, float, str]] = []
    for e in msp.query("TEXT MTEXT"):
        content = e.dxf.text if e.dxftype() == "TEXT" else e.text
        insert = e.dxf.insert
        text_entities.append((insert[0], insert[1], content))

    manholes: list[Manhole] = []
    for e in msp.query("INSERT"):
        manholes.append(Manhole(block_name=e.dxf.name, x=e.dxf.insert[0], y=e.dxf.insert[1], layer=e.dxf.layer))

    segments: list[NetworkSegment] = []
    unmatched_layers: set[str] = set()
    for e in msp.query("LINE LWPOLYLINE"):
        layer = e.dxf.layer
        network_type = match_layer_to_network_type(layer)
        if network_type is None:
            unmatched_layers.add(layer)

        if e.dxftype() == "LINE":
            coords = [(e.dxf.start[0], e.dxf.start[1]), (e.dxf.end[0], e.dxf.end[1])]
        else:
            coords = [(p[0], p[1]) for p in e.get_points()]
        if len(coords) < 2:
            continue
        geometry = LineString(coords)

        mid_x = sum(c[0] for c in coords) / len(coords)
        mid_y = sum(c[1] for c in coords) / len(coords)
        label = _nearby_text_value(text_entities, mid_x, mid_y, snap_distance)

        diameter_mm = extract_diameter_mm(label) if label else None
        material = extract_material(label) if label else None
        elevation_m = extract_elevation_m(label) if label else None

        if elevation_m is not None:
            elevation_source = "факт"
        else:
            elevation_m = -NORMATIVE_DEPTH_M_BY_NETWORK_TYPE.get(network_type, 1.0) if network_type else None
            elevation_source = "нормативная"

        segments.append(NetworkSegment(
            network_type=network_type, layer=layer, geometry=geometry,
            diameter_mm=diameter_mm, material=material,
            elevation_m=elevation_m, elevation_source=elevation_source,
        ))

    return DxfImportResult(segments=segments, manholes=manholes, unmatched_layers=unmatched_layers)
