"""Интеграция с НСПД (Национальная система публичного кадастрового реестра)."""

from .nsdi_client import NSPDClient, search_cadastre_by_coords, CadastreData

__all__ = ["NSPDClient", "search_cadastre_by_coords", "CadastreData"]
