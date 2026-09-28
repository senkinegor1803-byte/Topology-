"""Тесты Шага 2.11, п. 1: `tileset.json` (3D Tiles 1.1, плоская иерархия тайлов)."""

from __future__ import annotations

from topology_geo.tiling.grid import LOD0, LOD1, LOD2, TileIndex
from topology_geo.tiling.tileset import RING_BACKGROUND_STEP_M, TileContentEntry, build_tileset_json


def _entry(tile, lod, key, minx=0.0, miny=0.0, maxx=250.0, maxy=250.0, zmin=100.0, zmax=110.0) -> TileContentEntry:
    return TileContentEntry(
        tile=tile, lod=lod, storage_key=key,
        local_minx=minx, local_miny=miny, local_maxx=maxx, local_maxy=maxy, z_min=zmin, z_max=zmax,
    )


def test_build_tileset_json_empty_entries():
    doc = build_tileset_json([])
    assert doc["asset"]["version"] == "1.1"
    assert doc["root"]["children"] == []
    assert doc["geometricError"] == 0.0


def test_build_tileset_json_has_one_child_per_entry():
    entries = [
        _entry(TileIndex(zone=2, tx=0, ty=0), LOD2, "jobs/1/tiles/a.glb"),
        _entry(TileIndex(zone=2, tx=1, ty=0), LOD1, "jobs/1/tiles/b.glb"),
    ]
    doc = build_tileset_json(entries)
    assert len(doc["root"]["children"]) == 2
    uris = {c["content"]["uri"] for c in doc["root"]["children"]}
    assert uris == {"jobs/1/tiles/a.glb", "jobs/1/tiles/b.glb"}


def test_build_tileset_json_geometric_error_matches_ring():
    entries = [
        _entry(TileIndex(zone=2, tx=0, ty=0), LOD2, "a.glb"),
        _entry(TileIndex(zone=2, tx=1, ty=0), LOD0, "b.glb"),
    ]
    doc = build_tileset_json(entries)
    errors_by_uri = {c["content"]["uri"]: c["geometricError"] for c in doc["root"]["children"]}
    assert errors_by_uri["a.glb"] == RING_BACKGROUND_STEP_M[LOD2]
    assert errors_by_uri["b.glb"] == RING_BACKGROUND_STEP_M[LOD0]
    assert errors_by_uri["a.glb"] < errors_by_uri["b.glb"]  # ближнее кольцо точнее


def test_build_tileset_json_root_bounding_volume_covers_all_children():
    entries = [
        _entry(TileIndex(zone=2, tx=0, ty=0), LOD2, "a.glb", minx=-100, miny=-100, maxx=0, maxy=0),
        _entry(TileIndex(zone=2, tx=1, ty=0), LOD0, "b.glb", minx=500, miny=500, maxx=750, maxy=750),
    ]
    doc = build_tileset_json(entries)
    cx, cy, cz, hx, _, _, _, hy, _, _, _, hz = doc["root"]["boundingVolume"]["box"]
    assert cx - hx <= -100.0
    assert cx + hx >= 750.0
    assert cy - hy <= -100.0
    assert cy + hy >= 750.0


def test_build_tileset_json_root_refine_is_add():
    doc = build_tileset_json([_entry(TileIndex(zone=2, tx=0, ty=0), LOD2, "a.glb")])
    assert doc["root"]["refine"] == "ADD"


def test_build_tileset_json_child_extras_identify_tile():
    tile = TileIndex(zone=2, tx=5, ty=-3)
    doc = build_tileset_json([_entry(tile, LOD1, "a.glb")])
    extras = doc["root"]["children"][0]["extras"]
    assert extras == {"lod": LOD1, "tx": 5, "ty": -3, "zone": 2}


def test_build_tileset_json_is_valid_box_bounding_volume_shape():
    doc = build_tileset_json([_entry(TileIndex(zone=2, tx=0, ty=0), LOD2, "a.glb")])
    box = doc["root"]["children"][0]["boundingVolume"]["box"]
    assert len(box) == 12
    assert all(isinstance(v, float) for v in box)
