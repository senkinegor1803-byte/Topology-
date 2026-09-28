"""LandXML экспорт (Шаг 2.10, п. 2: «LandXML (рельеф, оси дорог)»).

Пишется вручную через `xml.etree.ElementTree` — сам формат LandXML 1.2 это
обычный XML с открытой (не проприетарной) схемой, специальная библиотека не
нужна. Точки поверхности — из уже готового TIN участка (Шаг 1.5), оси дорог
— из `RoadRibbon.axis`/`RailRibbon.axis` (Шаг 2.4/2.5: только дороги/пути с
ОДНИМ цельным сегментом линии, тот же принцип, что и везде в проекте).

Координаты — локальные метры участка (центр = (0,0)), как и everywhere в
конвейере (IFC/GLB), НЕ полноценная гео-привязка LandXML `<CoordinateSystem>`
с формальным EPSG-кодом: у МСК-59 в этом проекте нет декларированного
глобального кода (см. `coords.py`), только собственная реализация
пересчёта — честно, не изобретаем формальный идентификатор, которого нет.
Открытие в целевой CAD/ГИС-программе (критерий проверки шага) не проверено
— нет доступа к такой программе в AI-сессии.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Protocol

LANDXML_NAMESPACE = "http://www.landxml.org/schema/LandXML-1.2"
LANDXML_VERSION = "1.2"


class _HasAxis(Protocol):
    osm_id: int
    axis: object | None  # shapely LineString | None


class _SiteTinLike(Protocol):
    vertices: object  # numpy (N, 3)
    triangles: object  # numpy (M, 3)


def _surface_element(tin: _SiteTinLike) -> ET.Element:
    surface = ET.Element("Surface", {"name": "Рельеф участка", "desc": "TIN, Шаг 1.5"})
    definition = ET.SubElement(surface, "Definition", {"surfType": "TIN"})
    pnts = ET.SubElement(definition, "Pnts")
    for i, (x, y, z) in enumerate(tin.vertices, start=1):
        p = ET.SubElement(pnts, "P", {"id": str(i)})
        p.text = f"{x:.3f} {y:.3f} {z:.3f}"
    faces = ET.SubElement(definition, "Faces")
    for tri in tin.triangles:
        face = ET.SubElement(faces, "F")
        face.text = " ".join(str(int(idx) + 1) for idx in tri)  # LandXML - индексы точек с 1
    return surface


def _alignment_element(osm_id: int, axis, *, name_prefix: str) -> ET.Element | None:
    if axis is None or axis.length <= 0:
        return None
    coords = list(axis.coords)
    if len(coords) < 2:
        return None
    alignment = ET.Element(
        "Alignment", {"name": f"{name_prefix} {osm_id}", "length": f"{axis.length:.3f}", "staStart": "0.000"}
    )
    coord_geom = ET.SubElement(alignment, "CoordGeom")
    for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
        # Только прямые сегменты - ось хранится ломаной (LineString), без
        # аппроксимации кривыми (Spiral/Curve LandXML не строятся: входных
        # данных о радиусе/переходной кривой нет).
        line = ET.SubElement(coord_geom, "Line")
        start = ET.SubElement(line, "Start")
        start.text = f"{x0:.3f} {y0:.3f}"
        end = ET.SubElement(line, "End")
        end.text = f"{x1:.3f} {y1:.3f}"
    return alignment


def build_landxml(
    tin: _SiteTinLike | None,
    roads: list[_HasAxis] | None = None,
    rail: list[_HasAxis] | None = None,
) -> bytes:
    """Собрать LandXML: поверхность рельефа (если есть TIN) + оси дорог/путей
    (только объекты с непустой `axis`, Шаг 2.4/2.5, п. 5/3 — параметрическая
    ось хранится лишь для одного цельного сегмента линии)."""
    root = ET.Element(
        "LandXML",
        {
            "xmlns": LANDXML_NAMESPACE,
            "version": LANDXML_VERSION,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        },
    )
    application = ET.SubElement(
        root, "Application", {"name": "Топология", "desc": "Автоматическая генерация BIM-окружения (Шаг 2.10)"}
    )
    ET.SubElement(application, "Author")

    if tin is not None:
        surfaces = ET.SubElement(root, "Surfaces")
        surfaces.append(_surface_element(tin))

    alignments_elements = []
    for road in roads or []:
        el = _alignment_element(road.osm_id, road.axis, name_prefix="Дорога")
        if el is not None:
            alignments_elements.append(el)
    for track in rail or []:
        el = _alignment_element(track.osm_id, track.axis, name_prefix="Путь")
        if el is not None:
            alignments_elements.append(el)
    if alignments_elements:
        alignments = ET.SubElement(root, "Alignments")
        for el in alignments_elements:
            alignments.append(el)

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes
