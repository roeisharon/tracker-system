"""Tests for the FPS meter."""

from __future__ import annotations

import pytest

from tracker_system.metrics.fps import FpsMeter


def test_average_over_uniform_frames():
    meter = FpsMeter()
    for _ in range(10):
        meter.update(0.01)  # 100 fps each
    assert meter.average == pytest.approx(100.0)
    assert meter.frames == 10
    assert meter.min == pytest.approx(100.0)
    assert meter.max == pytest.approx(100.0)


def test_min_max_track_extremes():
    meter = FpsMeter()
    meter.update(0.01)  # 100 fps
    meter.update(0.10)  # 10 fps
    meter.update(0.02)  # 50 fps
    assert meter.max == 100.0
    assert meter.min == 10.0


def test_zero_dt_does_not_crash():
    meter = FpsMeter()
    fps = meter.update(0.0)  # guarded internally
    assert fps > 0


def test_summary_keys():
    meter = FpsMeter()
    meter.update(0.01)
    summary = meter.summary()
    assert set(summary) == {"frames", "avg_fps", "min_fps", "max_fps"}
