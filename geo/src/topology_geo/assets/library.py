"""Библиотека элементов (Шаг 2.8, п. 1): «структура Asset Library —
категория, тип, LOD, лицензия, источник, версия».

Каталог — `manifest.json` рядом с этим модулем, список записей `AssetEntry`.
Две разновидности записи (`kind`):

- `"generator"` — параметрический генератор кода (п. 2: опоры, столбы,
  бордюры, ограждения, пролёты, фонари, Шаги 2.3/2.6/2.7/2.8) — `files`
  пуст, `source` указывает на функцию-генератор в этом репозитории.
- `"texture"` — готовый файловый актив, PBR-текстура (п. 4) — `files`
  перечисляет реальные скачанные файлы под `geo/assets/`.

«Готовые модели с чистыми лицензиями» (деревья пород средней полосы, МАФ,
остановки — п. 3) в этот каталог не входят: интеграция реального licensed
3D-контента (поиск, проверка лицензии, загрузка детальных мешей) не
выполнялась в этом проходе, см. `docs/asset-library.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name("manifest.json")

KIND_GENERATOR = "generator"
KIND_TEXTURE = "texture"


@dataclass(frozen=True)
class AssetEntry:
    key: str
    category: str
    type: str
    lod: str
    license: str
    source: str
    version: str
    kind: str
    files: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


def load_manifest(path: Path = MANIFEST_PATH) -> list[AssetEntry]:
    """Загрузить каталог целиком. Пустой/отсутствующий файл — пустой список,
    не исключение (тот же принцип честного водопада, что и в остальном
    конвейере: нет каталога - библиотека пуста, не падение)."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [AssetEntry(**{**item, "files": tuple(item.get("files", ()))}) for item in raw]


def find_entry(key: str, *, path: Path = MANIFEST_PATH) -> AssetEntry | None:
    for entry in load_manifest(path):
        if entry.key == key:
            return entry
    return None


def entries_by_category(category: str, *, path: Path = MANIFEST_PATH) -> list[AssetEntry]:
    return [entry for entry in load_manifest(path) if entry.category == category]


def resolve_file(entry: AssetEntry, filename: str, *, root: Path | None = None) -> Path:
    """Абсолютный путь файла актива (`geo/assets/<путь из manifest>`) -
    «сцена использует элементы только ссылками» (критерий проверки шага):
    вызывающий код получает путь на файл библиотеки, а не копирует его
    содержимое к себе."""
    if filename not in entry.files:
        raise ValueError(f"{filename!r} не числится в files записи {entry.key!r}")
    base = root if root is not None else MANIFEST_PATH.parent.parent.parent.parent / "assets"
    return base / filename
