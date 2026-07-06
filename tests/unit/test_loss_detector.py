"""Tests for the loss detector's individual signals and hysteresis."""

from __future__ import annotations

import numpy as np
import pytest

from tracker_system.config.settings import LossConfig
from tracker_system.loss.detector import (
    REASON_LEFT_FRAME,
    REASON_LOW_SIMILARITY,
    REASON_MOTION_JUMP,
    REASON_SCALE_CHANGE,
    REASON_TRACKER_FAILURE,
    LossDetector,
    appearance_similarity,
)
from tracker_system.utils.geometry import BBox, bbox_from_center, extract_patch

FRAME_W, FRAME_H = 640, 480


def _detector(**overrides):
    cfg = LossConfig(**overrides)
    return LossDetector(cfg, FRAME_W, FRAME_H)


def _assess(detector, **kwargs):
    base = dict(
        tracker_ok=True,
        bbox=bbox_from_center(320, 240, 80),
        prev_bbox=None,
        frame=None,
        reference_template=None,  # no reference -> similarity check inert
        frame_index=1,
    )
    base.update(kwargs)
    return detector.assess(**base)


def test_healthy_frame():
    result = _assess(_detector())
    assert result.healthy is True
    assert result.reason is None
    assert result.consecutive_bad == 0


def test_tracker_failure_confirmed_after_window():
    detector = _detector(max_lost_frames=3)
    r1 = _assess(detector, tracker_ok=False, bbox=None)
    r2 = _assess(detector, tracker_ok=False, bbox=None)
    r3 = _assess(detector, tracker_ok=False, bbox=None)
    assert r1.reason == REASON_TRACKER_FAILURE
    assert (r1.confirmed_lost, r2.confirmed_lost) == (False, False)
    assert r3.confirmed_lost is True


def test_left_frame_detected():
    detector = _detector()
    result = _assess(detector, bbox=BBox(625, 0, 80, 80))  # mostly outside right edge
    assert result.reason == REASON_LEFT_FRAME
    assert result.healthy is False


def test_motion_jump_detected():
    detector = _detector()
    result = _assess(
        detector,
        prev_bbox=bbox_from_center(100, 100, 80),
        bbox=bbox_from_center(400, 100, 80),  # 300px jump > 0.25 * 800 diag
    )
    assert result.reason == REASON_MOTION_JUMP


def test_scale_change_detected():
    detector = _detector()
    result = _assess(
        detector,
        prev_bbox=bbox_from_center(320, 240, 40),  # area 1600
        bbox=bbox_from_center(320, 240, 80),  # area 6400 -> ratio 4 > 1.6
    )
    assert result.reason == REASON_SCALE_CHANGE


def test_low_similarity_detected():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
    bbox = bbox_from_center(320, 240, 80)
    different = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)

    detector = _detector(min_similarity=0.5)
    result = detector.assess(
        tracker_ok=True,
        bbox=bbox,
        prev_bbox=None,
        frame=frame,
        reference_template=different,
        frame_index=1,
    )
    assert result.reason == REASON_LOW_SIMILARITY


def test_matching_appearance_stays_healthy():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
    bbox = bbox_from_center(320, 240, 80)
    same = extract_patch(frame, bbox)

    detector = _detector(min_similarity=0.5)
    result = detector.assess(
        tracker_ok=True,
        bbox=bbox,
        prev_bbox=None,
        frame=frame,
        reference_template=same,
        frame_index=1,
    )
    assert result.healthy is True


def test_hysteresis_healthy_frame_resets_counter():
    detector = _detector(max_lost_frames=3)
    _assess(detector, tracker_ok=False, bbox=None)  # bad 1
    _assess(detector, tracker_ok=False, bbox=None)  # bad 2
    healthy = _assess(detector)  # resets
    assert healthy.consecutive_bad == 0
    # Now three fresh bad frames are needed to confirm.
    _assess(detector, tracker_ok=False, bbox=None)
    _assess(detector, tracker_ok=False, bbox=None)
    third = _assess(detector, tracker_ok=False, bbox=None)
    assert third.confirmed_lost is True


def test_isolated_similarity_dip_does_not_trigger_loss():
    """De-chatter: a single low-similarity frame amid healthy ones is ignored.

    The appearance signal is smoothed (EMA), so one dip cannot drag it below the
    threshold. Uses real patches: a matching reference (healthy) with one frame
    swapped for a mismatching patch.
    """
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
    bbox = bbox_from_center(320, 240, 80)
    reference = extract_patch(frame, bbox)  # perfect match -> high similarity
    mismatch_frame = rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)

    detector = _detector(min_similarity=0.5, max_lost_frames=5)

    def step(f):
        return detector.assess(
            tracker_ok=True, bbox=bbox, prev_bbox=None, frame=f,
            reference_template=reference, frame_index=1,
        )

    # Warm up the EMA on healthy frames, then a single bad frame, then healthy.
    for _ in range(5):
        assert step(frame).healthy
    dip = step(mismatch_frame)
    after = step(frame)
    # The isolated dip must not confirm a loss (EMA stays above threshold).
    assert dip.confirmed_lost is False
    assert after.healthy is True


def test_sustained_low_similarity_confirms_loss():
    """A sustained appearance drop still confirms LOST within the window."""
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
    bbox = bbox_from_center(320, 240, 80)
    reference = extract_patch(frame, bbox)
    bad_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    detector = _detector(min_similarity=0.5, max_lost_frames=5)
    detector.assess(tracker_ok=True, bbox=bbox, prev_bbox=None, frame=frame,
                    reference_template=reference, frame_index=1)  # seed healthy

    confirmed = False
    for _ in range(20):
        r = detector.assess(tracker_ok=True, bbox=bbox, prev_bbox=None,
                            frame=bad_frame, reference_template=reference, frame_index=1)
        confirmed = confirmed or r.confirmed_lost
    assert confirmed is True


def test_appearance_similarity_missing_patch_is_safe():
    assert appearance_similarity(None, np.zeros((4, 4, 3), np.uint8)) == 1.0
    assert appearance_similarity(np.zeros((0, 0, 3), np.uint8), np.zeros((4, 4, 3), np.uint8)) == 1.0


def test_adaptive_threshold_is_strict_for_stable_targets():
    """A tight (low-noise) distribution yields a gate near its median."""
    from tracker_system.loss.detector import robust_identity_threshold

    stable = [0.95, 0.96, 0.94, 0.95, 0.93, 0.96, 0.95, 0.94]  # bush-like
    threshold, center = robust_identity_threshold(stable, k=5.0)
    assert center == pytest.approx(0.95, abs=0.02)
    # Strict: a drift down to ~0.72 (background) is well below the gate.
    assert 0.72 < threshold < 0.95


def test_adaptive_threshold_is_tolerant_for_noisy_targets():
    """A wide (high-noise) distribution yields a very low/negative gate."""
    from tracker_system.loss.detector import robust_identity_threshold

    noisy = [0.98, 0.28, 0.87, 0.35, 0.92, 0.59, 0.81, 0.30, 0.64, 0.90]  # bottle-like
    threshold, _ = robust_identity_threshold(noisy, k=5.0)
    # Tolerant: even a genuine dip to ~0.3 stays above the gate.
    assert threshold < 0.3
