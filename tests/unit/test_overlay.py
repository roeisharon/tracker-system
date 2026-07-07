"""Smoke tests for overlay drawing (in place, no exceptions)."""

from __future__ import annotations

import numpy as np

from geometry import BBox
from overlay import draw_debug_search, render_overlay, state_color


def test_state_color_known_and_default():
    assert state_color("TRACKING") == (0, 200, 0)
    assert state_color("LOST") == (0, 0, 255)
    assert state_color("???") == (0, 0, 255)


def test_render_overlay_runs():
    frame = np.zeros((200, 300, 3), np.uint8)
    out = render_overlay(frame, BBox(10, 10, 40, 40), [(30, 30), (35, 32)],
                         "TRACKING", 42.0, "HYBRID", confidence=0.8, reason="ok")
    assert out.shape == frame.shape


def test_draw_debug_search_runs():
    frame = np.zeros((200, 300, 3), np.uint8)
    cands = [(BBox(10, 10, 20, 20), 0.5), (BBox(40, 40, 20, 20), 0.3)]
    out = draw_debug_search(frame, BBox(0, 0, 100, 100), cands, BBox(10, 10, 20, 20), (50, 50))
    assert out.shape == frame.shape
