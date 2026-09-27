"""Тесты Шага 2.8, п. 1: структура Asset Library (категория, тип, LOD,
лицензия, источник, версия) и критерий проверки шага по плану — «у каждого
элемента указана лицензия»."""

from __future__ import annotations

from pathlib import Path

import pytest

from topology_geo.assets.library import (
    KIND_GENERATOR,
    KIND_TEXTURE,
    AssetEntry,
    entries_by_category,
    find_entry,
    load_manifest,
    resolve_file,
)

ASSETS_ROOT = Path(__file__).resolve().parents[1] / "assets"


def test_load_manifest_returns_entries():
    entries = load_manifest()
    assert len(entries) > 0
    assert all(isinstance(e, AssetEntry) for e in entries)


def test_load_manifest_missing_file_returns_empty_list_not_exception():
    assert load_manifest(Path("/nonexistent/manifest.json")) == []


@pytest.mark.parametrize("field_name", ["key", "category", "type", "lod", "license", "source", "version"])
def test_every_entry_has_non_empty_required_fields(field_name):
    """Критерий проверки шага по плану: «у каждого элемента указана
    лицензия» - здесь же заодно все остальные обязательные поля структуры
    (п. 1: категория, тип, LOD, лицензия, источник, версия)."""
    for entry in load_manifest():
        assert getattr(entry, field_name).strip(), f"{entry.key}: пустое поле {field_name}"


def test_generator_entries_have_no_files():
    for entry in load_manifest():
        if entry.kind == KIND_GENERATOR:
            assert entry.files == ()


def test_texture_entries_reference_real_existing_files():
    """PBR-текстуры (п. 4) - не просто запись в каталоге, а реально
    скачанные файлы (Poly Haven, CC0)."""
    texture_entries = [e for e in load_manifest() if e.kind == KIND_TEXTURE]
    assert len(texture_entries) >= 5  # асфальт, плитка, грунт, газон, бетон
    for entry in texture_entries:
        assert entry.license == "CC0"
        assert len(entry.files) > 0
        for filename in entry.files:
            path = resolve_file(entry, filename, root=ASSETS_ROOT)
            assert path.is_file(), f"{entry.key}: файл {path} отсутствует"
            assert path.stat().st_size > 0


def test_find_entry_returns_none_for_unknown_key():
    assert find_entry("не-существует-такого-ключа") is None


def test_find_entry_returns_matching_entry():
    entry = find_entry("texture_asphalt")
    assert entry is not None
    assert entry.type == "асфальт"


def test_entries_by_category_filters():
    textures = entries_by_category("текстура")
    assert len(textures) >= 5
    assert all(e.category == "текстура" for e in textures)

    poles = entries_by_category("опора")
    assert len(poles) >= 2
    assert all(e.category == "опора" for e in poles)


def test_resolve_file_rejects_filename_not_in_entry():
    entry = find_entry("texture_asphalt")
    with pytest.raises(ValueError):
        resolve_file(entry, "textures/asphalt/does_not_exist.jpg", root=ASSETS_ROOT)


def test_all_material_categories_from_plan_are_covered():
    """Действие Шага 2.8, п. 4: «асфальт, плитка, грунт, газон, бетон»."""
    types = {e.type for e in load_manifest() if e.kind == KIND_TEXTURE}
    assert types == {"асфальт", "плитка (тротуарная)", "грунт", "газон", "бетон"}
