"""Тесты Шага 4.4: автоматические ракурсы (птичий полёт, пешеход, облёт)."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString, box

from topology_geo.rendering.camera_shots import (
    BIRD_EYE_ANGLE_DEG,
    EYE_HEIGHT_M,
    bird_eye_cameras,
    flythrough_path,
    pedestrian_view_points,
)


def test_bird_eye_cameras_returns_four_cameras_at_45_degrees():
    footprint = box(-10, -10, 10, 10)  # 20x20
    cameras = bird_eye_cameras(footprint, building_height_m=15.0, base_z=100.0)

    assert len(cameras) == 4
    for cam in cameras:
        cx, cy, cz = cam.position
        tx, ty, tz = cam.target
        horizontal = math.hypot(cx - tx, cy - ty)
        vertical = cz - 100.0  # относительно base_z (грубо - target z выше base_z на building_height/2, но camera z считается от base_z)
        elevation_angle = math.degrees(math.atan2(vertical, horizontal))
        assert elevation_angle == pytest.approx(BIRD_EYE_ANGLE_DEG, abs=0.5)


def test_bird_eye_cameras_four_compass_directions_are_90_degrees_apart():
    footprint = box(-10, -10, 10, 10)
    cameras = bird_eye_cameras(footprint, building_height_m=15.0)

    centroid = footprint.centroid
    bearings = []
    for cam in cameras:
        cx, cy, _ = cam.position
        bearing = math.degrees(math.atan2(cx - centroid.x, cy - centroid.y)) % 360
        bearings.append(bearing)
    bearings.sort()
    diffs = [(bearings[(i + 1) % 4] - bearings[i]) % 360 for i in range(4)]
    for diff in diffs:
        assert diff == pytest.approx(90.0, abs=0.1)


def test_bird_eye_cameras_target_is_building_centroid():
    footprint = box(0, 0, 20, 20)
    cameras = bird_eye_cameras(footprint, building_height_m=30.0, base_z=50.0)

    for cam in cameras:
        assert cam.target[0] == pytest.approx(10.0)
        assert cam.target[1] == pytest.approx(10.0)
        assert cam.target[2] == pytest.approx(50.0 + 15.0)  # base_z + height/2


def test_bird_eye_cameras_distance_scales_with_building_size():
    small = box(-5, -5, 5, 5)
    large = box(-50, -50, 50, 50)

    small_cam = bird_eye_cameras(small, building_height_m=10.0)[0]
    large_cam = bird_eye_cameras(large, building_height_m=10.0)[0]

    small_dist = math.hypot(small_cam.position[0], small_cam.position[1])
    large_dist = math.hypot(large_cam.position[0], large_cam.position[1])
    assert large_dist > small_dist


def test_pedestrian_view_points_at_eye_height():
    footprint = box(0, 0, 20, 20)
    sidewalk = LineString([(50, 10), (100, 10)])

    cameras = pedestrian_view_points(footprint, [sidewalk], obstacles=[])

    assert len(cameras) > 0
    for cam in cameras:
        assert cam.position[2] == pytest.approx(EYE_HEIGHT_M)


def test_pedestrian_view_points_excludes_blocked_sightlines():
    footprint = box(0, 0, 20, 20)
    sidewalk = LineString([(50, 10), (50, 10.1)])  # одна точка примерно
    obstacle = box(25, 5, 35, 15)  # прямо между тротуаром и ЖК

    cameras = pedestrian_view_points(footprint, [sidewalk], obstacles=[obstacle], sample_step_m=1.0)

    assert cameras == []


def test_pedestrian_view_points_excludes_too_far_points():
    footprint = box(0, 0, 20, 20)
    sidewalk = LineString([(1000, 10), (1010, 10)])

    cameras = pedestrian_view_points(footprint, [sidewalk], obstacles=[], max_distance_m=150.0)

    assert cameras == []


def test_flythrough_path_generates_correct_frame_count():
    footprint = box(-10, -10, 10, 10)
    cameras = flythrough_path(footprint, building_height_m=15.0, duration_s=25.0, fps=30)

    assert len(cameras) == 25 * 30


def test_flythrough_path_rejects_duration_outside_plan_range():
    footprint = box(-10, -10, 10, 10)
    with pytest.raises(ValueError, match="20-30"):
        flythrough_path(footprint, building_height_m=15.0, duration_s=10.0)


def test_flythrough_path_covers_full_circle():
    footprint = box(-10, -10, 10, 10)
    cameras = flythrough_path(footprint, building_height_m=15.0, duration_s=20.0, fps=4)  # 80 кадров, но берём немного для теста

    centroid = footprint.centroid
    first_bearing = math.atan2(cameras[0].position[0] - centroid.x, cameras[0].position[1] - centroid.y)
    last_bearing = math.atan2(cameras[-1].position[0] - centroid.x, cameras[-1].position[1] - centroid.y)
    # последний кадр - почти полный круг назад к первому (не ровно, т.к. дискретные шаги)
    assert abs(first_bearing - last_bearing) < math.radians(10.0) or abs(abs(first_bearing - last_bearing) - 2 * math.pi) < math.radians(10.0)
