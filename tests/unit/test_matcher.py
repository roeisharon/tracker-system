"""Tests for the re-acquisition matcher: scoring, multi-scale, distractors."""

from __future__ import annotations

import numpy as np
import pytest

from tracker_system.config.settings import ReacquireConfig
from tracker_system.reacquisition.matcher import (
    Matcher,
    motion_prior,
    weighted_score,
)
from tracker_system.utils.geometry import BBox, extract_patch

FRAME_W, FRAME_H = 640, 480
DIAG = (FRAME_W**2 + FRAME_H**2) ** 0.5


def _blank_frame():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def _texture(size, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def _structured(size):
    """A four-colour patch whose histogram survives resizing (unlike noise)."""
    p = np.zeros((size, size, 3), dtype=np.uint8)
    h = size // 2
    p[:h, :h] = (200, 40, 40)
    p[:h, h:] = (40, 200, 40)
    p[h:, :h] = (40, 40, 200)
    p[h:, h:] = (200, 200, 40)
    return p


def _place(frame, patch, top_left):
    x, y = top_left
    h, w = patch.shape[:2]
    frame[y : y + h, x : x + w] = patch


# -- pure scoring helpers --------------------------------------------------


def test_weighted_score_monotonic_in_each_component():
    cfg = ReacquireConfig()
    base = weighted_score(0.5, 0.5, 0.5, cfg)
    assert weighted_score(0.9, 0.5, 0.5, cfg) > base  # better template
    assert weighted_score(0.5, 0.9, 0.5, cfg) > base  # better histogram
    assert weighted_score(0.5, 0.5, 0.9, cfg) > base  # better motion


def test_motion_prior_decays_with_distance():
    at = motion_prior((100, 100), (100, 100), DIAG, 0.25)
    near = motion_prior((110, 100), (100, 100), DIAG, 0.25)
    far = motion_prior((400, 400), (100, 100), DIAG, 0.25)
    assert at == pytest.approx(1.0)
    assert at > near > far
    # No prediction -> neutral.
    assert motion_prior((0, 0), None, DIAG, 0.25) == 1.0


# -- matching --------------------------------------------------------------


def test_finds_target_in_search_region():
    template = _texture(40, seed=1)
    frame = _blank_frame()
    _place(frame, template, (300, 200))  # target centre ~ (320, 220)

    matcher = Matcher(ReacquireConfig(scales=(1.0,)))
    region = BBox(240, 140, 200, 200)
    result = matcher.find(frame, region, template, (320, 220), DIAG)

    assert result is not None
    cx, cy = result.bbox.center
    assert abs(cx - 320) <= 3 and abs(cy - 220) <= 3


@pytest.mark.parametrize("search_scale", [1.0, 0.5])
def test_downscaled_matching_returns_fullres_coords(search_scale):
    """Matching on a downscaled ROI must still report full-resolution coords.

    (Phase 4 · Increment 2: SEARCHING runs matchTemplate on a downscaled frame for
    speed; candidate boxes must map back to native coordinates.)
    """
    template = _texture(40, seed=8)
    frame = _blank_frame()
    _place(frame, template, (300, 200))  # centre ~ (320, 220) in full-res

    matcher = Matcher(ReacquireConfig(scales=(1.0,), search_scale=search_scale))
    result = matcher.find(frame, BBox(240, 140, 200, 200), template, (320, 220), DIAG)

    assert result is not None
    cx, cy = result.bbox.center
    # Localisation tolerance scales with 1/search_scale (down-then-up rounding).
    tol = max(3, int(round(1.0 / search_scale)) + 2)
    assert abs(cx - 320) <= tol and abs(cy - 220) <= tol
    assert result.bbox.w == pytest.approx(40, abs=2)  # full-res template size


def test_multiscale_finds_rescaled_patch():
    import cv2

    template = _structured(40)
    scaled = cv2.resize(template, (50, 50))  # 1.25x
    frame = _blank_frame()
    _place(frame, scaled, (300, 200))

    matcher = Matcher(ReacquireConfig(scales=(0.8, 1.0, 1.25)))
    region = BBox(240, 140, 220, 220)
    result = matcher.find(frame, region, template, (325, 225), DIAG)

    assert result is not None
    # The winning box should be about the rescaled (50px) size, not 40px.
    assert result.bbox.w == pytest.approx(50, abs=2)


def test_motion_prior_disambiguates_identical_distractor():
    """Two identical patches: the one near the predicted position must win."""
    template = _texture(40, seed=3)
    frame = _blank_frame()
    _place(frame, template, (100, 200))  # true target, near prediction
    _place(frame, template, (520, 200))  # identical distractor, far away

    matcher = Matcher(ReacquireConfig(scales=(1.0,)))
    region = BBox(0, 120, FRAME_W, 200)  # covers both patches
    predicted = (120, 220)  # near the first patch

    result = matcher.find(frame, region, template, predicted, DIAG)
    assert result is not None
    assert result.bbox.center[0] < FRAME_W / 2  # chose the left (true) target


def test_no_match_below_min_score_returns_none():
    template = _texture(40, seed=4)
    frame = _blank_frame()  # target absent

    matcher = Matcher(ReacquireConfig(scales=(1.0,), min_score=0.5))
    region = BBox(240, 140, 200, 200)
    assert matcher.find(frame, region, template, (320, 220), DIAG) is None


def test_spatial_gate_rejects_far_lookalike():
    """A perfect-appearance match far from the prediction must be rejected.

    This is the identity constraint: a look-alike elsewhere in the frame (e.g. a
    different but identical bottle) is not the tracked instance.
    """
    template = _texture(40, seed=5)
    frame = _blank_frame()
    _place(frame, template, (560, 200))  # identical patch, far right

    matcher = Matcher(ReacquireConfig(scales=(1.0,)))
    region = BBox(0, 120, FRAME_W, 200)  # region spans the whole width
    predicted = (80, 220)  # target predicted at the far LEFT

    # The far patch has tmpl~1, hist~1, but its motion prior is tiny -> gated out.
    result = matcher.find(frame, template=template, search_region=region,
                          predicted_center=predicted, diagonal=DIAG)
    assert result is None
    # And it is present as a candidate that fails only the spatial gate.
    best = matcher.last_best
    assert best is not None and best.hist_score >= 0.3
    assert best.motion_score < 0.3


def test_spatial_gate_accepts_lookalike_at_prediction():
    """The same patch, now near the prediction, is accepted."""
    template = _texture(40, seed=5)
    frame = _blank_frame()
    _place(frame, template, (60, 200))  # centre ~ (80, 220)

    matcher = Matcher(ReacquireConfig(scales=(1.0,)))
    region = BBox(0, 120, FRAME_W, 200)
    result = matcher.find(frame, template=template, search_region=region,
                          predicted_center=(80, 220), diagonal=DIAG)
    assert result is not None
    assert abs(result.bbox.center[0] - 80) <= 4
