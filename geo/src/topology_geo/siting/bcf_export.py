"""Экспорт коллизий в BCF (Шаг 3.9, п. 2) — BIM Collaboration Format,
открытый стандарт buildingSMART для обмена коллизиями между BIM-
инструментами (Renga/Pilot-BIM его читают штатно — план сам называет
критерий «BCF открывается в Renga или Pilot-BIM»), а не изобретённый
формат: zip-архив, один подкаталог на GUID-тему коллизии, `markup.bcf`
(XML — заголовок/описание) + `viewpoint.bcfv` (XML — ссылки на объекты).

Снимок вида (упоминается в действии плана «запись BCF со снимком вида»)
ЧЕСТНО не генерируется — потребовал бы headless-рендер сцены (текущий
рендер — three.js во вьюере в браузере, Шаг 1.9, не доступен из Python-
бэкенда без отдельного headless-браузера); сама BCF-тема и ссылки на
объекты (`Component IfcGuid=...`) — содержательная часть коллизии,
которую Renga/Pilot-BIM использует для «перехода к объектам» (критерий
шага) — реализованы полностью."""

from __future__ import annotations

import io
import uuid
import zipfile
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

from topology_geo.siting.checks import Collision

BCF_VERSION = "2.1"


def _markup_xml(topic_guid: str, collision: Collision, viewpoint_guid: str) -> bytes:
    markup = ET.Element("Markup")
    topic = ET.SubElement(markup, "Topic", Guid=topic_guid, TopicType="Clash", TopicStatus="Open")
    ET.SubElement(topic, "Title").text = collision.check_type
    ET.SubElement(topic, "Description").text = collision.description
    ET.SubElement(topic, "CreationDate").text = datetime.now(timezone.utc).isoformat()
    viewpoints = ET.SubElement(markup, "Viewpoints", Guid=viewpoint_guid)
    ET.SubElement(viewpoints, "Viewpoint").text = "viewpoint.bcfv"
    return ET.tostring(markup, encoding="utf-8", xml_declaration=True)


def _viewpoint_xml(viewpoint_guid: str, collision: Collision) -> bytes:
    root = ET.Element("VisualizationInfo", Guid=viewpoint_guid)
    components = ET.SubElement(root, "Components")
    selection = ET.SubElement(components, "Selection")
    for ref in collision.object_refs:
        if ref:
            ET.SubElement(selection, "Component", IfcGuid=str(ref))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_bcf(collisions: list[Collision]) -> bytes:
    """Собрать `.bcf`-архив (zip) из списка коллизий. Каждая коллизия —
    отдельная тема (Topic) со своим GUID, реально открываемая структура
    (проверено обратным разбором в тестах: `zipfile`+`ElementTree`, не
    только «не упало»)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bcf.version", f'<?xml version="1.0" encoding="UTF-8"?><Version VersionId="{BCF_VERSION}"/>')
        for collision in collisions:
            topic_guid = str(uuid.uuid4())
            viewpoint_guid = str(uuid.uuid4())
            zf.writestr(f"{topic_guid}/markup.bcf", _markup_xml(topic_guid, collision, viewpoint_guid))
            zf.writestr(f"{topic_guid}/viewpoint.bcfv", _viewpoint_xml(viewpoint_guid, collision))
    return buf.getvalue()
