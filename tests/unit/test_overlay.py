"""Tests for burned-in overlay detection and overlay-safe initialisation."""

from __future__ import annotations

import cv2
import numpy as np

from tracker_system.config.settings import SelectionConfig
from tracker_system.selection.overlay import (
    box_overlaps_overlay,
    clean_overlay,
    detect_static_overlay,
    overlay_free_first_frame,
)
from tracker_system.utils.geometry import BBox

SIZE = 200
CROSS = (100, 100)


def _draw_cross(img, at=CROSS):
    cv2.line(img, (at[0] - 15, at[1]), (at[0] + 15, at[1]), (255, 255, 255), 2)
    cv2.line(img, (at[0], at[1] - 15), (at[0], at[1] + 15), (255, 255, 255), 2)


def _moving_bg_with_overlay(n=14):
    """Frames whose textured background *shifts* (camera motion) under a fixed cross."""
    rng = np.random.default_rng(0)
    bg = rng.integers(0, 255, (SIZE + 80, SIZE + 80, 3), dtype=np.uint8)
    frames = []
    for i in range(n):
        dx, dy = i * 4, i * 3
        crop = bg[dy : dy + SIZE, dx : dx + SIZE].copy()
        _draw_cross(crop)
        frames.append(crop)
    return frames


def _static_scene(n=14):
    """Identical frames (no motion) — overlay cannot be told from real content."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    _draw_cross(base)
    return [base.copy() for _ in range(n)]


def test_detects_fixed_overlay_over_moving_background():
    mask = detect_static_overlay(_moving_bg_with_overlay(), SelectionConfig())
    assert mask is not None
    # The fixed cross region is flagged...
    assert mask[90:110, 90:110].any()
    # ...and it is a small fraction of the frame (not the whole thing).
    assert 0.0 < (mask > 0).mean() < 0.15


def test_no_detection_without_camera_motion():
    # A static scene has no majority-dynamic frame, so detection abstains (no-op).
    assert detect_static_overlay(_static_scene(), SelectionConfig()) is None


def test_no_detection_with_too_few_frames():
    assert detect_static_overlay(_moving_bg_with_overlay(n=3), SelectionConfig()) is None


def test_box_overlap_detection():
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[90:110, 90:110] = 255
    assert box_overlaps_overlay(mask, BBox(80, 80, 40, 40)) is True
    assert box_overlaps_overlay(mask, BBox(0, 0, 30, 30)) is False


def test_clean_overlay_removes_bright_overlay_pixels():
    frame = np.full((SIZE, SIZE, 3), 60, np.uint8)
    _draw_cross(frame)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    _draw_cross(mask)  # single-channel mask: cv2.line writes 255 along the cross
    cleaned = clean_overlay(frame, mask)
    # The bright cross centre should be inpainted down toward the background level.
    assert int(cleaned[CROSS[1], CROSS[0]].mean()) < 150


def test_overlay_free_first_frame_is_noop_when_disabled():
    frame = np.full((SIZE, SIZE, 3), 60, np.uint8)
    _draw_cross(frame)
    out = overlay_free_first_frame(
        "unused", frame, BBox(80, 80, 40, 40), SelectionConfig(handle_overlay=False)
    )
    assert out is frame
