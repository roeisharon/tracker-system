"""Tests for fused-confidence loss detection with hysteresis."""

from __future__ import annotations

from config import LossConfig
from geometry import BBox
from loss import (
    LossDetector, REASON_LEFT_FRAME, REASON_LOW_CONFIDENCE, REASON_SCALE_CHANGE,
)

BOX = BBox(40, 40, 20, 20)


def _det(**kw):
    return LossDetector(LossConfig(**kw), 100, 100)


def test_healthy_frame_resets_counter():
    d = _det(t_lost=0.35, lost_patience=3)
    a = d.assess(0.9, BOX, BOX)
    assert a.healthy and not a.confirmed_lost and a.consecutive_bad == 0


def test_low_confidence_confirms_after_patience():
    d = _det(t_lost=0.35, lost_patience=3)
    assert not d.assess(0.1, BOX, BOX).confirmed_lost   # bad 1
    assert not d.assess(0.1, BOX, BOX).confirmed_lost   # bad 2
    a = d.assess(0.1, BOX, BOX)                          # bad 3 -> confirmed
    assert a.confirmed_lost and a.reason == REASON_LOW_CONFIDENCE


def test_one_healthy_frame_breaks_the_streak():
    d = _det(t_lost=0.35, lost_patience=3)
    d.assess(0.1, BOX, BOX)
    d.assess(0.1, BOX, BOX)
    d.assess(0.9, BOX, BOX)  # reset
    assert d.assess(0.1, BOX, BOX).consecutive_bad == 1


def test_left_frame_detected():
    d = _det(min_frame_overlap=0.3)
    a = d.assess(0.9, BBox(-18, 40, 20, 20), BOX)
    assert a.reason == REASON_LEFT_FRAME


def test_scale_change_detected():
    d = _det(max_scale_ratio=4.0)
    a = d.assess(0.9, BBox(40, 40, 100, 100), BOX)  # ~25x area jump
    assert a.reason == REASON_SCALE_CHANGE


def test_reset_clears_counter():
    d = _det(lost_patience=3)
    d.assess(0.1, BOX, BOX)
    d.reset()
    assert d.assess(0.1, BOX, BOX).consecutive_bad == 1
