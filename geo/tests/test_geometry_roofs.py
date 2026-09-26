"""Тесты форм крыш (Шаг 2.2, п. 1-2). Объёмы проверены точными замкнутыми
формулами (не приближённо), замкнутость меша - подсчётом рёбер (каждое ребро
входит ровно в две грани, в противоположных направлениях)."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from topology_geo.geometry.roofs import (
    CONFIDENCE_DEFAULT,
    CONFIDENCE_FACT,
    ROOF_FLAT,
    ROOF_GABLED,
    ROOF_HIPPED,
    ROOF_PYRAMIDAL,
    ROOF_SKILLION,
    OrientedBox,
    RoofParams,
    build_pitched_building_mesh,
    compute_roof_params,
    oriented_bounding_box,
)
from topology_geo.selection.service import SiteFeature


def _feature(raw_tags: dict[str, str]) -> SiteFeature:
    return SiteFeature(
        layer="osm_buildings", osm_id=1, osm_type="W",
        geometry=Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]),
        attributes={}, confidence={}, raw_tags=raw_tags,
    )


def _signed_volume(vertices, faces) -> float:
    vol = 0.0
    for face in faces:
        for i in range(1, len(face) - 1):
            a, b, c = face[0], face[i], face[i + 1]
            v0, v1, v2 = vertices[a], vertices[b], vertices[c]
            vol += (
                v0[0] * (v1[1] * v2[2] - v1[2] * v2[1])
                - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0])
                + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0])
            )
    return vol / 6.0


def _edge_count(faces):
    counts: dict[tuple[int, int], int] = {}
    for face in faces:
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i + 1) % n]
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def _assert_watertight(faces):
    counts = _edge_count(faces)
    for (a, b), c in counts.items():
        assert c == 1, f"ребро {(a, b)} встречается {c} раз"
        assert counts.get((b, a), 0) == 1, f"обратное ребро {(b, a)} отсутствует"


# --- oriented_bounding_box ----------------------------------------------------


def test_oriented_bounding_box_of_axis_aligned_rectangle():
    poly = Polygon([(0, 0), (20, 0), (20, 10), (0, 10)])
    obb = oriented_bounding_box(poly)
    assert obb.length == pytest.approx(20.0)
    assert obb.width == pytest.approx(10.0)
    assert obb.center == pytest.approx((10.0, 5.0))
    assert abs(obb.long_axis[0]) == pytest.approx(1.0, abs=1e-9)


def test_oriented_bounding_box_area_covers_rectangle_exactly():
    poly = Polygon([(0, 0), (20, 0), (20, 10), (0, 10)])
    obb = oriented_bounding_box(poly)
    assert obb.length * obb.width == pytest.approx(poly.area)


# --- compute_roof_params ------------------------------------------------------


def test_compute_roof_params_defaults_to_flat_when_no_tag():
    obb = oriented_bounding_box(Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]))
    params = compute_roof_params(_feature({}), obb)
    assert params.shape == ROOF_FLAT
    assert params.height_m == 0.0


def test_compute_roof_params_flat_explicit_is_fact():
    obb = oriented_bounding_box(Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]))
    params = compute_roof_params(_feature({"roof:shape": "flat"}), obb)
    assert params.shape == ROOF_FLAT
    assert params.shape_confidence == CONFIDENCE_FACT


def test_compute_roof_params_unsupported_shape_falls_back_to_flat_with_default_confidence():
    obb = oriented_bounding_box(Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]))
    params = compute_roof_params(_feature({"roof:shape": "dome"}), obb)
    assert params.shape == ROOF_FLAT
    assert params.shape_confidence == CONFIDENCE_DEFAULT


def test_compute_roof_params_uses_roof_height_tag_when_present():
    obb = oriented_bounding_box(Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]))
    params = compute_roof_params(_feature({"roof:shape": "gabled", "roof:height": "3.5"}), obb)
    assert params.shape == ROOF_GABLED
    assert params.height_m == pytest.approx(3.5)
    assert params.height_confidence == CONFIDENCE_FACT


def test_compute_roof_params_uses_roof_angle_when_height_missing():
    obb = oriented_bounding_box(Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]))
    params = compute_roof_params(_feature({"roof:shape": "gabled", "roof:angle": "45"}), obb)
    # уклон 45° -> высота = половина поперечника (tan(45)=1)
    assert params.height_m == pytest.approx(obb.width / 2.0)
    assert params.height_confidence == CONFIDENCE_FACT


def test_compute_roof_params_defaults_height_with_default_confidence():
    obb = oriented_bounding_box(Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]))
    params = compute_roof_params(_feature({"roof:shape": "gabled"}), obb)
    assert params.height_m > 0
    assert params.height_confidence == CONFIDENCE_DEFAULT


def test_compute_roof_params_gabled_orientation_across_uses_short_axis():
    obb = oriented_bounding_box(Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]))
    along = compute_roof_params(_feature({"roof:shape": "gabled", "roof:height": "2"}), obb)
    across = compute_roof_params(
        _feature({"roof:shape": "gabled", "roof:height": "2", "roof:orientation": "across"}), obb
    )
    assert along.ridge_along_long_axis is True
    assert across.ridge_along_long_axis is False


def test_compute_roof_params_direction_picks_nearest_obb_axis_for_gabled():
    obb = oriented_bounding_box(Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]))
    # длинная сторона обб вдоль x (восток); скат "на юг" (direction=180) означает
    # конёк вдоль запад-восток (long_axis) - ridge_along_long=True
    south = compute_roof_params(_feature({"roof:shape": "gabled", "roof:direction": "180"}), obb)
    assert south.ridge_along_long_axis is True
    # скат "на восток" (direction=90) означает конёк вдоль север-юг (short_axis)
    east = compute_roof_params(_feature({"roof:shape": "gabled", "roof:direction": "90"}), obb)
    assert east.ridge_along_long_axis is False


def test_compute_roof_params_hipped_and_pyramidal_ignore_orientation():
    obb = oriented_bounding_box(Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]))
    hipped = compute_roof_params(
        _feature({"roof:shape": "hipped", "roof:height": "2", "roof:orientation": "across"}), obb
    )
    pyramidal = compute_roof_params(_feature({"roof:shape": "pyramidal", "roof:height": "2"}), obb)
    assert hipped.ridge_along_long_axis is True
    assert pyramidal.ridge_along_long_axis is True


def test_compute_roof_params_skillion_direction_defaults_to_short_axis():
    obb = oriented_bounding_box(Polygon([(0, 0), (20, 0), (20, 10), (0, 10)]))
    params = compute_roof_params(_feature({"roof:shape": "skillion", "roof:height": "2"}), obb)
    assert params.direction == obb.short_axis


# --- build_pitched_building_mesh: объём + замкнутость -------------------------


L, W, H, BASE_Z, EAVE_Z = 20.0, 10.0, 4.0, 0.0, 6.0
OBB = OrientedBox(center=(0.0, 0.0), length=L, width=W, long_axis=(1.0, 0.0), short_axis=(0.0, 1.0))
WALL_VOL = L * W * (EAVE_Z - BASE_Z)


@pytest.mark.parametrize(
    "name,params,expected_volume",
    [
        ("gabled-along", RoofParams(ROOF_GABLED, CONFIDENCE_FACT, H, CONFIDENCE_FACT, True, None), WALL_VOL + 0.5 * L * W * H),
        ("gabled-across", RoofParams(ROOF_GABLED, CONFIDENCE_FACT, H, CONFIDENCE_FACT, False, None), WALL_VOL + 0.5 * L * W * H),
        ("hipped", RoofParams(ROOF_HIPPED, CONFIDENCE_FACT, H, CONFIDENCE_FACT, True, None), WALL_VOL + H * (L * W / 2 - W**2 / 6)),
        ("pyramidal", RoofParams(ROOF_PYRAMIDAL, CONFIDENCE_FACT, H, CONFIDENCE_FACT, True, None), WALL_VOL + L * W * H / 3.0),
        ("skillion-x", RoofParams(ROOF_SKILLION, CONFIDENCE_FACT, H, CONFIDENCE_FACT, True, (1.0, 0.0)), WALL_VOL + 0.5 * L * W * H),
        ("skillion-y", RoofParams(ROOF_SKILLION, CONFIDENCE_FACT, H, CONFIDENCE_FACT, True, (0.0, 1.0)), WALL_VOL + 0.5 * L * W * H),
    ],
)
def test_pitched_roof_volume_matches_closed_form(name, params, expected_volume):
    vertices, faces = build_pitched_building_mesh(OBB, params, BASE_Z, EAVE_Z)
    assert _signed_volume(vertices, faces) == pytest.approx(expected_volume, rel=1e-9)


@pytest.mark.parametrize(
    "params",
    [
        RoofParams(ROOF_GABLED, CONFIDENCE_FACT, H, CONFIDENCE_FACT, True, None),
        RoofParams(ROOF_GABLED, CONFIDENCE_FACT, H, CONFIDENCE_FACT, False, None),
        RoofParams(ROOF_HIPPED, CONFIDENCE_FACT, H, CONFIDENCE_FACT, True, None),
        RoofParams(ROOF_PYRAMIDAL, CONFIDENCE_FACT, H, CONFIDENCE_FACT, True, None),
        RoofParams(ROOF_SKILLION, CONFIDENCE_FACT, H, CONFIDENCE_FACT, True, (1.0, 0.0)),
    ],
)
def test_pitched_roof_mesh_is_watertight(params):
    _, faces = build_pitched_building_mesh(OBB, params, BASE_Z, EAVE_Z)
    _assert_watertight(faces)


def test_hipped_matches_pyramidal_when_square_base():
    square = OrientedBox(center=(0.0, 0.0), length=10.0, width=10.0, long_axis=(1.0, 0.0), short_axis=(0.0, 1.0))
    hipped = RoofParams(ROOF_HIPPED, CONFIDENCE_FACT, H, CONFIDENCE_FACT, True, None)
    pyramidal = RoofParams(ROOF_PYRAMIDAL, CONFIDENCE_FACT, H, CONFIDENCE_FACT, True, None)
    vol_hipped = _signed_volume(*build_pitched_building_mesh(square, hipped, BASE_Z, EAVE_Z))
    vol_pyramidal = _signed_volume(*build_pitched_building_mesh(square, pyramidal, BASE_Z, EAVE_Z))
    assert vol_hipped == pytest.approx(vol_pyramidal, rel=1e-9)


def test_gabled_along_and_across_same_volume_different_ridge_orientation():
    along_verts, _ = build_pitched_building_mesh(
        OBB, RoofParams(ROOF_GABLED, CONFIDENCE_FACT, H, CONFIDENCE_FACT, True, None), BASE_Z, EAVE_Z
    )
    across_verts, _ = build_pitched_building_mesh(
        OBB, RoofParams(ROOF_GABLED, CONFIDENCE_FACT, H, CONFIDENCE_FACT, False, None), BASE_Z, EAVE_Z
    )
    # конёк "along" тянется вдоль x (длинная сторона), "across" - вдоль y
    ridge_pts_along = [v for v in along_verts if abs(v[2] - (EAVE_Z + H)) < 1e-9]
    ridge_pts_across = [v for v in across_verts if abs(v[2] - (EAVE_Z + H)) < 1e-9]
    xs_along = {round(p[0], 6) for p in ridge_pts_along}
    ys_across = {round(p[1], 6) for p in ridge_pts_across}
    assert len(xs_along) == 2  # конёк вдоль x -> разные x, одна y=0
    assert len(ys_across) == 2  # конёк вдоль y -> разные y, одна x=0
