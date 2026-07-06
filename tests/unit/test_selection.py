"""Tests for selectors and burned-in overlay detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tracker_system.config import SelectionConfig
from tracker_system.geometry import BBox
from tracker_system.selection import (
    CvClickSelector, ManualPixelSelector, SelectionError,
    box_overlaps_overlay, detect_static_overlay,
)

FRAME_H, FRAME_W = 480, 640


@pytest.fixture
def frame():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def test_manual_selector_maps_ij_to_centered_bbox(frame):
    result = ManualPixelSelector(row=200, col=300, bbox_size=80).select(frame)
    assert result.source == "manual" and result.seed_point == (300, 200)
    assert result.bbox.center == pytest.approx((300.0, 200.0))


def test_manual_selector_outside_frame_raises(frame):
    with pytest.raises(SelectionError):
        ManualPixelSelector(row=1000, col=300, bbox_size=80).select(frame)


def test_mouse_click_matches_manual(frame):
    x, y = 300, 200
    sel = CvClickSelector(bbox_size=80)
    sel._frame_shape = frame.shape[:2]
    sel._on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    manual = ManualPixelSelector(row=y, col=x, bbox_size=80).select(frame)
    assert sel._result.source == "mouse"
    assert sel._result.bbox == manual.bbox


def test_overlay_detection_flags_static_structural_pixels():
    # Moving background + a fixed white cross => the cross is the overlay.
    cfg = SelectionConfig(overlay_static_std=12.0, overlay_min_motion_frac=0.4)
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(8):
        f = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        cv2.line(f, (80, 0), (80, 119), (255, 255, 255), 2)
        cv2.line(f, (0, 60), (159, 60), (255, 255, 255), 2)
        frames.append(f)
    mask = detect_static_overlay(frames, cfg)
    assert mask is not None and mask.sum() > 0
    assert box_overlaps_overlay(mask, BBox(70, 50, 20, 20))


def test_overlay_detection_noop_on_static_scene():
    cfg = SelectionConfig()
    frames = [np.full((100, 100, 3), 100, np.uint8) for _ in range(8)]
    assert detect_static_overlay(frames, cfg) is None
