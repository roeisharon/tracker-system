"""Tests for target selection (manual + mouse-click mapping)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tracker_system.selection.cv_click_selector import CvClickSelector
from tracker_system.selection.target_selector import (
    ManualPixelSelector,
    SelectionError,
)

FRAME_H, FRAME_W = 480, 640


@pytest.fixture
def frame():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def test_manual_selector_maps_ij_to_centered_bbox(frame):
    # [i, j] = (row, col) = (200, 300) -> centre at (x=300, y=200).
    result = ManualPixelSelector(row=200, col=300, bbox_size=80).select(frame)
    assert result.source == "manual"
    assert result.seed_point == (300, 200)
    cx, cy = result.bbox.center
    assert cx == pytest.approx(300.0)
    assert cy == pytest.approx(200.0)
    assert (result.bbox.w, result.bbox.h) == (80.0, 80.0)


def test_manual_selector_outside_frame_raises(frame):
    with pytest.raises(SelectionError):
        ManualPixelSelector(row=1000, col=300, bbox_size=80).select(frame)


def test_manual_selector_clamps_bbox_near_edge(frame):
    # Point in the corner: the box must stay within the frame.
    result = ManualPixelSelector(row=5, col=5, bbox_size=80).select(frame)
    assert result.bbox.x >= 0 and result.bbox.y >= 0
    assert result.bbox.x2 <= FRAME_W and result.bbox.y2 <= FRAME_H


def test_mouse_click_maps_to_same_result_as_manual(frame):
    """A click at (x, y) must yield the same bbox as manual [i=y, j=x]."""
    x, y = 300, 200
    selector = CvClickSelector(bbox_size=80)
    selector._frame_shape = frame.shape[:2]  # what select() would set

    # Simulate the OpenCV mouse callback firing a left-button-down event.
    selector._on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    mouse_result = selector._result

    manual_result = ManualPixelSelector(row=y, col=x, bbox_size=80).select(frame)

    assert mouse_result is not None
    assert mouse_result.source == "mouse"
    assert mouse_result.seed_point == manual_result.seed_point
    assert mouse_result.bbox == manual_result.bbox


def test_mouse_non_click_event_is_ignored(frame):
    selector = CvClickSelector(bbox_size=80)
    selector._frame_shape = frame.shape[:2]
    selector._on_mouse(cv2.EVENT_MOUSEMOVE, 10, 10, 0, None)
    assert selector._result is None
