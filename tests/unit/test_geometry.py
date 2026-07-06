"""Tests for bounding-box geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tracker_system.utils.geometry import (
    BBox,
    bbox_from_center,
    clamp_bbox,
    clamp_point,
    extract_patch,
    frame_overlap_ratio,
    scale_bbox,
)


def test_bbox_from_center_is_centered():
    box = bbox_from_center(100, 50, 20)
    assert box.center == (100.0, 50.0)
    assert (box.w, box.h) == (20.0, 20.0)
    assert (box.x, box.y) == (90.0, 40.0)


def test_bbox_properties():
    box = BBox(10, 20, 30, 40)
    assert box.x2 == 40
    assert box.y2 == 60
    assert box.area == 1200
    assert box.as_int_xywh() == (10, 20, 30, 40)


def test_scale_bbox_roundtrip():
    box = BBox(100, 200, 40, 60)
    down = scale_bbox(box, 0.5)
    assert (down.x, down.y, down.w, down.h) == (50, 100, 20, 30)
    up = down.scaled(2.0)
    assert (up.x, up.y, up.w, up.h) == (100, 200, 40, 60)


def test_clamp_bbox_keeps_box_inside_frame():
    # Box hanging off the top-left corner.
    clamped = clamp_bbox(BBox(-10, -10, 50, 50), frame_w=100, frame_h=100)
    assert clamped.x >= 0 and clamped.y >= 0
    assert clamped.x2 <= 100 and clamped.y2 <= 100

    # Box hanging off the bottom-right corner.
    clamped = clamp_bbox(BBox(90, 90, 50, 50), frame_w=100, frame_h=100)
    assert clamped.x2 <= 100 and clamped.y2 <= 100
    assert clamped.w >= 1 and clamped.h >= 1


def test_clamp_point():
    assert clamp_point(-5, 200, 100, 100) == (0, 99)
    assert clamp_point(50, 50, 100, 100) == (50, 50)


def test_extract_patch_shape():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    patch = extract_patch(image, BBox(10, 20, 30, 40))
    assert patch.shape == (40, 30, 3)


def test_extract_patch_clamped_at_edge():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Box partially outside the frame -> patch is trimmed, never empty.
    patch = extract_patch(image, BBox(90, 90, 40, 40))
    assert patch.size > 0
    assert patch.shape[0] <= 40 and patch.shape[1] <= 40


def test_frame_overlap_ratio():
    # Fully inside.
    assert frame_overlap_ratio(BBox(10, 10, 20, 20), 100, 100) == 1.0
    # Half off the right edge.
    assert frame_overlap_ratio(BBox(90, 10, 20, 20), 100, 100) == pytest.approx(0.5)
    # Fully outside.
    assert frame_overlap_ratio(BBox(200, 200, 20, 20), 100, 100) == 0.0
