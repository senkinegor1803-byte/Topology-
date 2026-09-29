"""Тесты Шага 4.7: панорамы 360° и видео облёта. Собирает настоящую сцену
тем же путём, что и остальной пайплайн (Шаги 1.9/4.4/4.5), реально
рендерит через `bpy` Cycles и проверяет результат: панораму — пиксельно
через Pillow, видео — метаданными реального потока через `ffprobe`.

`bpy` и системный `ffmpeg` — опциональные зависимости; модуль целиком
пропускается, если их нет (см. `docs/rendering.md`)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, box

bpy = pytest.importorskip("bpy")
PIL_Image = pytest.importorskip("PIL.Image")

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="требуется системный ffmpeg")

from topology_geo.geometry.buildings import BuildingSolid
from topology_geo.ifc.assemble import BasePoint, SiteModel, build_site_ifc
from topology_geo.ifc.to_glb import convert_ifc_to_glb
from topology_geo.relief.tin import SiteTin
from topology_geo.rendering.camera_shots import flythrough_path
from topology_geo.rendering.panorama import (
    DEFAULT_PANORAMA_RESOLUTION,
    render_flythrough_video,
    render_panorama,
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


def _ffprobe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def test_render_panorama_produces_real_equirectangular_image(tmp_path: Path):
    from PIL import Image

    glb_path = _make_scene_glb(tmp_path)
    output_path = tmp_path / "pano.png"

    result = render_panorama(
        glb_path, (0.0, -20.0, 1.6), output_path, quality="draft",
        sun_direction=(0.5, 0.0, 0.85), resolution=(256, 128),
    )

    assert result == output_path
    assert output_path.exists()
    image = Image.open(output_path)
    assert image.size == (256, 128)
    assert image.size[0] == 2 * image.size[1]  # обязательное 2:1 для equirectangular
    arr = np.array(image.convert("RGB"))
    assert arr.std() > 5.0, "панорама выглядит однородной/пустой"


def test_render_panorama_rejects_non_2to1_resolution(tmp_path: Path):
    glb_path = _make_scene_glb(tmp_path)
    with pytest.raises(ValueError, match="2:1"):
        render_panorama(glb_path, (0.0, 0.0, 1.6), tmp_path / "out.png", resolution=(100, 100))


def test_render_panorama_default_resolution_is_2to1():
    assert DEFAULT_PANORAMA_RESOLUTION[0] == 2 * DEFAULT_PANORAMA_RESOLUTION[1]


def test_render_flythrough_video_produces_valid_mp4(tmp_path: Path):
    glb_path = _make_scene_glb(tmp_path)
    footprint = box(-5, -5, 5, 5)
    cameras = flythrough_path(footprint, building_height_m=10.0, base_z=0.0, duration_s=20.0, fps=2)[:5]
    output_path = tmp_path / "flythrough.mp4"

    result = render_flythrough_video(
        glb_path, cameras, output_path, quality="draft", fps=2,
        sun_direction=(0.5, 0.0, 0.85), resolution_override=(160, 90),
    )

    assert result == output_path
    assert output_path.exists() and output_path.stat().st_size > 0

    probe = _ffprobe(output_path)
    video_streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
    assert len(video_streams) == 1
    stream = video_streams[0]
    assert stream["codec_name"] == "h264"
    assert (int(stream["width"]), int(stream["height"])) == (160, 90)
    assert stream["nb_frames"] == "5"


def test_render_flythrough_video_rejects_empty_camera_list(tmp_path: Path):
    glb_path = _make_scene_glb(tmp_path)
    with pytest.raises(ValueError, match="пуст"):
        render_flythrough_video(glb_path, [], tmp_path / "out.mp4")


def test_render_flythrough_video_rejects_unknown_quality(tmp_path: Path):
    glb_path = _make_scene_glb(tmp_path)
    footprint = box(-5, -5, 5, 5)
    cameras = flythrough_path(footprint, building_height_m=10.0, base_z=0.0, duration_s=20.0, fps=1)[:2]
    with pytest.raises(ValueError, match="неизвестный пресет"):
        render_flythrough_video(glb_path, cameras, tmp_path / "out.mp4", quality="ultra")
