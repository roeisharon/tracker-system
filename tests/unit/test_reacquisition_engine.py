"""Tests for the re-acquisition engine (predict, expand, match)."""

from __future__ import annotations

import numpy as np
import pytest

from tracker_system.config.settings import ReacquireConfig
from tracker_system.reacquisition.engine import ReacquisitionEngine
from tracker_system.reacquisition.matcher import Matcher
from tracker_system.target.profile import TargetProfile
from tracker_system.utils.geometry import BBox

FRAME_W, FRAME_H = 640, 480


def _texture(size, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def _engine(config=None):
    config = config or ReacquireConfig(scales=(1.0,))
    return ReacquisitionEngine(config, Matcher(config), FRAME_W, FRAME_H)


def _profile(center, velocity, template):
    cx, cy = center
    box = BBox(cx - 20, cy - 20, 40, 40)
    return TargetProfile(
        initial_bbox=box, current_bbox=box, template=template,
        velocity=velocity, history=[box],
    )


def test_predicted_center_extrapolates_and_clamps():
    # With no ego transform the prediction falls back to a per-step constant
    # velocity carry (advanced once per step), capped at max_prediction_frames.
    engine = _engine()
    engine.begin(_profile((100, 240), (10, 0), _texture(40)))
    for _ in range(5):
        cx, cy = engine._advance_prediction(None)
    assert cx == pytest.approx(150)  # 100 + 10 * 5
    assert cy == pytest.approx(240)

    # Runaway velocity is clamped to the frame.
    engine = _engine()
    engine.begin(_profile((600, 240), (1000, 0), _texture(40)))
    for _ in range(5):
        cx, _ = engine._advance_prediction(None)
    assert cx <= FRAME_W - 1


def test_predicted_center_follows_ego_transform():
    # A confident camera transform carries the prediction (world-fixed target).
    from tracker_system.motion.ego_motion import Transform
    import numpy as np

    engine = _engine()
    engine.begin(_profile((100, 240), (0, 0), _texture(40)))
    shift = Transform(np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -4.0]]), confidence=0.9)
    cx, cy = engine._advance_prediction(shift)
    assert cx == pytest.approx(112)
    assert cy == pytest.approx(236)


def test_search_radius_grows_then_caps():
    cfg = ReacquireConfig(
        scales=(1.0,), search_radius_frac=0.1,
        search_expansion_frac=0.05, max_search_radius_frac=0.5,
    )
    engine = _engine(cfg)
    engine.begin(_profile((100, 100), (0, 0), _texture(40)))
    engine.frames_since_lost = 0
    r0 = engine._search_radius()
    engine.frames_since_lost = 3
    r3 = engine._search_radius()
    engine.frames_since_lost = 1000
    r_big = engine._search_radius()
    assert r3 > r0
    assert r_big == pytest.approx(0.5 * engine.diagonal)  # capped


def test_step_reacquires_target_near_prediction():
    template = _texture(40, seed=1)
    engine = _engine()
    engine.begin(_profile((110, 240), (0, 0), template))

    # Target reappears at the predicted position.
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[220:260, 90:130] = template  # centre (110, 240)

    found = engine.step(frame)
    assert found is not None
    assert abs(found.center[0] - 110) <= 3
    assert abs(found.center[1] - 240) <= 3


def test_step_returns_none_when_absent():
    engine = _engine()
    engine.begin(_profile((110, 240), (0, 0), _texture(40)))
    blank = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    assert engine.step(blank) is None


def test_search_region_stays_bounded():
    """Even after a long absence the region must not cover the whole frame."""
    cfg = ReacquireConfig(scales=(1.0,), max_region_frac=0.6)
    engine = _engine(cfg)
    engine.begin(_profile((100, 240), (0, 0), _texture(40)))
    blank = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    for _ in range(500):  # let the radius grow as long as possible
        engine.step(blank)
    region = engine.last_search_region
    assert region.w <= cfg.max_region_frac * FRAME_W + 1
    assert region.h <= cfg.max_region_frac * FRAME_H + 1
