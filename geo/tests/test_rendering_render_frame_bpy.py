"""Тест Шага 4.5: рендер-воркер, реальный Cycles CPU-рендер кадра
(`rendering.render_frame`). Собирает настоящую сцену тем же путём, что и
остальной пайплайн (Шаги 1.9/4.4), рендерит кадр и проверяет настоящий
пиксельный результат через Pillow (не только «файл создан»).

`bpy` — тяжёлая (374 МБ) опциональная зависимость, не входит в
`pyproject.toml` (см. `docs/rendering.md`); модуль целиком пропускается,
если пакет не установлен."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, box

bpy = pytest.importorskip("bpy")
PIL_Image = pytest.importorskip("PIL.Image")

from topology_geo.geometry.buildings import BuildingSolid
from topology_geo.ifc.assemble import BasePoint, SiteModel, build_site_ifc
from topology_geo.ifc.to_glb import convert_ifc_to_glb
from topology_geo.relief.tin import SiteTin
from topology_geo.rendering.camera_shots import bird_eye_cameras
from topology_geo.rendering.render_frame import (
    OUTPUT_FORMAT_EXR,
    QUALITY_DRAFT,
    QUALITY_FINAL,
    render_frame,
)

BASE_POINT = BasePoint(lon=56.25, lat=58.0, zone=2, x=2_310_450.0, y=-5_857_320.0, height=150.0)


def _make_flat_tin(half_extent: float = 40.0, n: int = 15) -> SiteTin:
    xs, ys = np.meshgrid(np.linspace(-half_extent, half_extent, n), np.linspace(-half_extent, half_extent, n))
    zs = np.zeros_like(xs)
    vertices = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    delaunay = Delaunay(vertices[:, :2])
    return SiteTin(vertices=vertices, triangles=delaunay.simplices, _delaunay=delaunay)


def _make_scene_glb(tmp_path: Path) -> Path:
    building = BuildingSolid(
        osm_id=1, footprint=Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)]),
        height_m=10.0, height_confidence="факт", height_source="OSM", base_z=0.0, building_type="жилой",
    )
    model_ifc, _ = build_site_ifc("IFC4", SiteModel(tin=_make_flat_tin(), buildings=[building]), BASE_POINT)
    glb_path = tmp_path / "scene.glb"
    glb_path.write_bytes(convert_ifc_to_glb(model_ifc))
    return glb_path


def _camera():
    footprint = box(-5, -5, 5, 5)
    return bird_eye_cameras(footprint, building_height_m=10.0, base_z=0.0)[0]


def test_render_frame_draft_produces_real_varied_image(tmp_path: Path):
    from PIL import Image

    glb_path = _make_scene_glb(tmp_path)
    output_path = tmp_path / "out.png"

    result = render_frame(glb_path, _camera(), output_path, quality="draft", sun_direction=(0.5, 0.0, 0.85))

    assert result.width, result.height == QUALITY_DRAFT.resolution
    assert result.samples == QUALITY_DRAFT.samples
    assert result.elapsed_s > 0
    assert output_path.exists()

    image = Image.open(output_path)
    assert image.size == QUALITY_DRAFT.resolution
    arr = np.array(image.convert("RGB"))
    assert arr.std() > 5.0, "кадр выглядит однородным/пустым - рендер не удался по существу"


def test_render_frame_resolution_override_ignores_preset_resolution(tmp_path: Path):
    from PIL import Image

    glb_path = _make_scene_glb(tmp_path)
    output_path = tmp_path / "out.png"

    result = render_frame(
        glb_path, _camera(), output_path, quality="draft",
        sun_direction=(0.5, 0.0, 0.85), resolution_override=(320, 240),
    )

    assert (result.width, result.height) == (320, 240)
    assert result.samples == QUALITY_DRAFT.samples  # сэмплы всё ещё от пресета
    image = Image.open(output_path)
    assert image.size == (320, 240)


def test_render_frame_final_quality_uses_denoising_and_more_samples(tmp_path: Path):
    glb_path = _make_scene_glb(tmp_path)
    output_path = tmp_path / "out.png"

    result = render_frame(
        glb_path, _camera(), output_path, quality="final",
        sun_direction=(0.5, 0.0, 0.85), resolution_override=(160, 90),  # маленькое разрешение - тест быстрый
    )

    assert result.samples == QUALITY_FINAL.samples
    assert result.samples > QUALITY_DRAFT.samples
    assert output_path.exists()


def test_render_frame_rejects_unknown_quality(tmp_path: Path):
    glb_path = _make_scene_glb(tmp_path)
    with pytest.raises(ValueError, match="неизвестный пресет"):
        render_frame(glb_path, _camera(), tmp_path / "out.png", quality="ultra")


def test_render_frame_supports_exr_output(tmp_path: Path):
    glb_path = _make_scene_glb(tmp_path)
    output_path = tmp_path / "out.exr"

    render_frame(
        glb_path, _camera(), output_path, quality="draft", output_format=OUTPUT_FORMAT_EXR,
        sun_direction=(0.5, 0.0, 0.85), resolution_override=(160, 90),
    )

    assert output_path.exists() and output_path.stat().st_size > 0


def test_render_frame_without_sun_direction_still_renders(tmp_path: Path):
    """Без `sun_direction` сцена освещается только фоновым мировым светом -
    должна рендериться (не падать), пусть и менее контрастно."""
    from PIL import Image

    glb_path = _make_scene_glb(tmp_path)
    output_path = tmp_path / "out.png"

    render_frame(glb_path, _camera(), output_path, quality="draft", sun_direction=None, resolution_override=(160, 90))

    image = Image.open(output_path)
    assert image.size == (160, 90)
