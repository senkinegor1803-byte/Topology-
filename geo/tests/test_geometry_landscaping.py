"""Тесты Шага 2.8, п. 2: ограждения/стены (с пролётами столбов) и фонари —
недостающие параметрические генераторы («опоры, столбы, бордюры,
ограждения, пролёты, фонари»; опоры/столбы/бордюр реализованы раньше)."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, Point

from topology_geo.geometry.landscaping import (
    DEFAULT_FENCE_HEIGHT_M,
    DEFAULT_WALL_HEIGHT_M,
    FENCE,
    STREETLAMP_POLE_HEIGHT_M,
    WALL,
    WALL_THICKNESS_M,
    FenceSegment,
    StreetLamp,
    build_fences,
    build_streetlamps,
)
from topology_geo.selection.service import SiteFeature


def _feature(layer, geometry, osm_id=1, raw_tags=None) -> SiteFeature:
    return SiteFeature(
        layer=layer, osm_id=osm_id, osm_type="W", geometry=geometry,
        attributes={}, confidence={}, raw_tags=raw_tags or {},
    )


# --- build_fences ------------------------------------------------------------


def test_build_fences_wall_is_solid_ribbon_without_posts():
    length = 20.0
    feature = _feature("osm_landscaping", LineString([(0, 0), (length, 0)]), raw_tags={"barrier": "wall"})
    fences = build_fences([feature])
    assert len(fences) == 1
    wall = fences[0]
    assert isinstance(wall, FenceSegment)
    assert wall.kind == WALL
    assert wall.height_m == pytest.approx(DEFAULT_WALL_HEIGHT_M)
    assert wall.posts == ()
    assert wall.ribbon.area == pytest.approx(length * WALL_THICKNESS_M, rel=1e-6)


def test_build_fences_fence_has_posts_at_regular_spans():
    length = 12.5
    feature = _feature("osm_landscaping", LineString([(0, 0), (length, 0)]), raw_tags={"barrier": "fence"})
    fence = build_fences([feature])[0]
    assert fence.kind == FENCE
    assert fence.height_m == pytest.approx(DEFAULT_FENCE_HEIGHT_M)
    assert len(fence.posts) >= 2
    xs = sorted(p[0] for p in fence.posts)
    assert xs[0] == pytest.approx(0.0)
    assert xs[-1] == pytest.approx(length)
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    assert max(gaps) - min(gaps) < 1e-6  # равномерно, без обрубка последнего пролёта


def test_build_fences_ignores_other_barrier_values_and_layers():
    other_barrier = _feature("osm_landscaping", LineString([(0, 0), (5, 0)]), raw_tags={"barrier": "hedge"})
    other_layer = _feature("osm_roads", LineString([(0, 0), (5, 0)]), raw_tags={"barrier": "fence"})
    assert build_fences([other_barrier, other_layer]) == []


def test_build_fences_respects_custom_spacing():
    length = 20.0
    feature = _feature("osm_landscaping", LineString([(0, 0), (length, 0)]), raw_tags={"barrier": "fence"})
    fence_wide = build_fences([feature], post_spacing_m=10.0)[0]
    fence_narrow = build_fences([feature], post_spacing_m=2.0)[0]
    assert len(fence_narrow.posts) > len(fence_wide.posts)


# --- build_streetlamps --------------------------------------------------------


def test_build_streetlamps_reads_point_with_defaults():
    feature = _feature("osm_landscaping", Point(5, 10), osm_id=7, raw_tags={"highway": "street_lamp"})
    lamps = build_streetlamps([feature])
    assert len(lamps) == 1
    lamp = lamps[0]
    assert isinstance(lamp, StreetLamp)
    assert lamp.osm_id == 7
    assert (lamp.x, lamp.y) == (5.0, 10.0)
    assert lamp.pole_height_m == pytest.approx(STREETLAMP_POLE_HEIGHT_M)


def test_build_streetlamps_ignores_non_point_and_other_highway_values():
    line_feature = _feature("osm_landscaping", LineString([(0, 0), (1, 1)]), raw_tags={"highway": "street_lamp"})
    other_highway = _feature("osm_landscaping", Point(0, 0), raw_tags={"highway": "residential"})
    assert build_streetlamps([line_feature, other_highway]) == []
