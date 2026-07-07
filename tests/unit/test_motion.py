"""Tests for global ego-motion estimation."""

from __future__ import annotations

import cv2
import numpy as np

from config import MotionConfig
from geometry import BBox
from motion import GlobalMotionEstimator, Transform


def _textured(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (240, 320, 3), dtype=np.uint8)


def test_identity_has_zero_confidence():
    t = Transform.identity()
    assert t.confidence == 0.0 and t.scale == 1.0
    assert t.apply_point((5, 7)) == (5.0, 7.0)


def test_first_frame_returns_identity():
    est = GlobalMotionEstimator(MotionConfig())
    assert est.update(_textured()).confidence == 0.0


def test_translation_is_recovered():
    est = GlobalMotionEstimator(MotionConfig())
    base = _textured(1)
    est.update(base)
    shifted = np.roll(base, 6, axis=1)  # shift right by 6 px
    t = est.update(shifted)
    assert t.confidence > 0
    dx, _ = t.apply_point((160, 120))
    assert dx > 160  # motion carried the point to the right


def test_apply_bbox_scales_and_translates():
    m = np.array([[1.1, 0.0, 5.0], [0.0, 1.1, 0.0]], dtype=np.float64)
    out = Transform(m, 1.0).apply_bbox(BBox(0, 0, 10, 10))
    assert out.w > 10 and out.h > 10
