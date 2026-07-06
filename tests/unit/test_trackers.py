"""Tests for tracker backends and the flow scale/translation estimator."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tracker_system.config import TrackerConfig
from tracker_system.geometry import BBox
from tracker_system.trackers import (
    FlowTracker, TrackerNotAvailableError, create_backend, probe_backends,
)


def _textured_frame(cx, cy, seed=0):
    rng = np.random.default_rng(seed)
    frame = np.full((300, 400, 3), 30, np.uint8)
    tex = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    x, y = int(cx - 40), int(cy - 40)
    frame[y:y + 80, x:x + 80] = tex
    return frame


def test_probe_reports_csrt_available():
    assert probe_backends(TrackerConfig())["csrt"] is True


def test_csrt_backend_tracks_a_shift():
    trk = create_backend(TrackerConfig(backend="csrt"))
    trk.init(_textured_frame(200, 150), BBox(160, 110, 80, 80))
    found, box, score = trk.update(_textured_frame(210, 150))
    assert found and box.center[0] > 190


def test_unknown_backend_raises():
    with pytest.raises(TrackerNotAvailableError):
        create_backend(TrackerConfig(backend="nope"))


def test_flow_tracker_follows_translation():
    cfg = TrackerConfig()
    ft = FlowTracker(cfg)
    ft.init(_textured_frame(200, 150), BBox(160, 110, 80, 80))
    result = ft.update(_textured_frame(212, 150))
    assert result is not None
    ncx, ncy, scale, score = result
    assert ncx > 200  # centre carried to the right
    assert cfg.flow_scale_min <= scale <= cfg.flow_scale_max
