"""Tests for the global (camera) motion estimator."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tracker_system.config.settings import MotionConfig
from tracker_system.motion.ego_motion import GlobalMotionEstimator, Transform
from tracker_system.utils.geometry import BBox


def _textured_frame(w=640, h=480, seed=0):
    """A richly-textured frame so optical flow has features to track."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _translate(frame, dx, dy):
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]))


# -- Transform value type ------------------------------------------------


def test_identity_transform_is_low_confidence_noop():
    t = Transform.identity()
    assert t.confidence == 0.0
    assert t.scale == pytest.approx(1.0)
    assert t.apply_point((37.0, 12.0)) == pytest.approx((37.0, 12.0))


def test_transform_applies_translation_and_scale_to_bbox():
    # scale 2, translate (+10, -5): matrix [[2,0,10],[0,2,-5]].
    m = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, -5.0]])
    t = Transform(m, confidence=0.8)
    assert t.scale == pytest.approx(2.0)
    box = BBox(0.0, 0.0, 20.0, 20.0)  # centre (10, 10)
    out = t.apply_bbox(box)
    assert out.w == pytest.approx(40.0) and out.h == pytest.approx(40.0)
    # new centre = (2*10+10, 2*10-5) = (30, 15)
    assert out.center == pytest.approx((30.0, 15.0))


# -- Estimator behaviour -------------------------------------------------


def test_first_frame_returns_identity():
    est = GlobalMotionEstimator(MotionConfig())
    t = est.update(_textured_frame())
    assert t.confidence == 0.0  # no previous frame yet


def test_recovers_pure_translation():
    cfg = MotionConfig(flow_scale=1.0)  # full-res for a crisp synthetic check
    est = GlobalMotionEstimator(cfg)
    frame = _textured_frame(seed=3)
    est.update(frame)
    shifted = _translate(frame, dx=15, dy=-8)
    t = est.update(shifted)
    assert t.confidence > 0.0
    # Background moved (+15, -8); the transform (prev->cur) should report it.
    x, y = t.apply_point((100.0, 100.0))
    assert x == pytest.approx(115.0, abs=1.5)
    assert y == pytest.approx(92.0, abs=1.5)
    assert t.scale == pytest.approx(1.0, abs=0.05)


def test_disabled_estimator_is_noop():
    est = GlobalMotionEstimator(MotionConfig(enabled=False))
    est.update(_textured_frame())
    t = est.update(_translate(_textured_frame(), 15, -8))
    assert t.confidence == 0.0


def test_featureless_frame_returns_identity():
    cfg = MotionConfig(flow_scale=1.0)
    est = GlobalMotionEstimator(cfg)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    est.update(blank)
    t = est.update(np.zeros((480, 640, 3), dtype=np.uint8))
    assert t.confidence == 0.0  # nothing to track -> graceful fallback


def test_target_box_is_masked_out():
    # A frame whose ONLY texture is inside the target box: masking it out leaves
    # no background features, so the estimate must fall back to identity rather
    # than tracking the target's own (independent) motion.
    cfg = MotionConfig(flow_scale=1.0, min_features=10)
    est = GlobalMotionEstimator(cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[200:280, 300:380] = _textured_frame(80, 80, seed=5)
    target = BBox(300, 200, 80, 80)
    est.update(frame, target)
    moved = np.zeros((480, 640, 3), dtype=np.uint8)
    moved[210:290, 320:400] = frame[200:280, 300:380]  # target moved, bg empty
    t = est.update(moved, target)
    assert t.confidence == 0.0
