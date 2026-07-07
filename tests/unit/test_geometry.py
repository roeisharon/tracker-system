"""Tests for BBox geometry + frame resize helpers."""

from __future__ import annotations

import numpy as np
import pytest

from geometry import (
    BBox, bbox_from_center, bbox_from_center_wh, clamp_bbox, clamp_point,
    extract_patch, frame_overlap_ratio, resize_frame,
)


def test_bbox_center_edges_area():
    b = BBox(10, 20, 30, 40)
    assert b.center == (25.0, 40.0)
    assert (b.x2, b.y2) == (40, 60)
    assert b.area == 1200


def test_bbox_from_center_variants():
    assert bbox_from_center(50, 50, 20) == BBox(40, 40, 20, 20)
    assert bbox_from_center_wh(50, 50, 20, 40) == BBox(40, 30, 20, 40)


def test_clamp_bbox_keeps_box_inside_frame():
    c = clamp_bbox(BBox(-10, -10, 50, 50), 100, 100)
    assert c.x >= 0 and c.y >= 0 and c.x2 <= 100 and c.y2 <= 100


def test_clamp_bbox_minimum_size():
    c = clamp_bbox(BBox(200, 200, 5, 5), 100, 100)
    assert c.w >= 1 and c.h >= 1


def test_clamp_point():
    assert clamp_point(-5, 500, 100, 100) == (0, 99)


def test_frame_overlap_ratio():
    assert frame_overlap_ratio(BBox(10, 10, 20, 20), 100, 100) == pytest.approx(1.0)
    assert frame_overlap_ratio(BBox(-20, 0, 20, 20), 100, 100) == pytest.approx(0.0)
    assert frame_overlap_ratio(BBox(-10, 0, 20, 20), 100, 100) == pytest.approx(0.5)


def test_extract_patch_is_safe_and_copies():
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    patch = extract_patch(img, BBox(90, 90, 40, 40))  # spills past the edge
    assert patch.size > 0 and patch.base is None


def test_resize_frame_identity_and_downscale():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    assert resize_frame(img, 1.0) is img
    small = resize_frame(img, 0.5)
    assert small.shape[:2] == (50, 100)
