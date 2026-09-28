"""Тесты Шага 3.9, п. 2: экспорт коллизий в BCF — реальный обратный разбор
zip+XML, не только «не упало»."""

from __future__ import annotations

import io
import zipfile
from xml.etree import ElementTree as ET

from topology_geo.siting.bcf_export import BCF_VERSION, build_bcf
from topology_geo.siting.checks import Collision


def test_build_bcf_produces_valid_zip_with_version_file():
    collisions = [Collision(check_type="ЗОУИТ", description="пересечение с зоной", object_refs=["b1", "z1"])]

    bcf_bytes = build_bcf(collisions)

    zf = zipfile.ZipFile(io.BytesIO(bcf_bytes))
    assert zf.testzip() is None  # архив целостен
    assert "bcf.version" in zf.namelist()
    version_xml = ET.fromstring(zf.read("bcf.version"))
    assert version_xml.get("VersionId") == BCF_VERSION


def test_build_bcf_one_topic_folder_per_collision():
    collisions = [
        Collision(check_type="ЗОУИТ", description="A", object_refs=["b1"]),
        Collision(check_type="красные линии", description="B", object_refs=["b1", "rl1"]),
    ]

    bcf_bytes = build_bcf(collisions)
    zf = zipfile.ZipFile(io.BytesIO(bcf_bytes))

    markup_files = [n for n in zf.namelist() if n.endswith("markup.bcf")]
    viewpoint_files = [n for n in zf.namelist() if n.endswith("viewpoint.bcfv")]
    assert len(markup_files) == 2
    assert len(viewpoint_files) == 2
    # каждая тема - в своём GUID-подкаталоге
    assert len({m.split("/")[0] for m in markup_files}) == 2


def test_build_bcf_markup_contains_title_and_description():
    collisions = [Collision(check_type="ЛЭП", description="охранная зона ЛЭП 110 кВ", object_refs=["b1"])]

    bcf_bytes = build_bcf(collisions)
    zf = zipfile.ZipFile(io.BytesIO(bcf_bytes))
    markup_name = next(n for n in zf.namelist() if n.endswith("markup.bcf"))
    markup = ET.fromstring(zf.read(markup_name))

    topic = markup.find("Topic")
    assert topic is not None
    assert topic.get("Guid")
    assert topic.find("Title").text == "ЛЭП"
    assert topic.find("Description").text == "охранная зона ЛЭП 110 кВ"


def test_build_bcf_viewpoint_references_object_guids():
    collisions = [Collision(check_type="сети", description="близко к трубе", object_refs=["building-1", "К1"])]

    bcf_bytes = build_bcf(collisions)
    zf = zipfile.ZipFile(io.BytesIO(bcf_bytes))
    viewpoint_name = next(n for n in zf.namelist() if n.endswith("viewpoint.bcfv"))
    viewpoint = ET.fromstring(zf.read(viewpoint_name))

    components = viewpoint.findall(".//Component")
    ifc_guids = {c.get("IfcGuid") for c in components}
    assert ifc_guids == {"building-1", "К1"}


def test_build_bcf_empty_collisions_still_valid_archive():
    bcf_bytes = build_bcf([])
    zf = zipfile.ZipFile(io.BytesIO(bcf_bytes))
    assert zf.testzip() is None
    assert zf.namelist() == ["bcf.version"]
