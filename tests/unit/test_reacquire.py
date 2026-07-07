"""Tests for appearance-confirmed re-acquisition."""

from __future__ import annotations

import numpy as np

from appearance import AppearanceMemory, Verifier
from config import ReacquireConfig, VerifierConfig
from geometry import BBox
from reacquire import Reacquirer


def _frame_with_texture_at(x, y, tex):
    frame = np.full((300, 400, 3), 30, np.uint8)
    frame[y:y + 80, x:x + 80] = tex
    return frame


def test_reacquire_finds_returned_target():
    tex = np.random.default_rng(5).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    box = BBox(60, 60, 80, 80)
    frame_a = _frame_with_texture_at(60, 60, tex)
    mem = AppearanceMemory(VerifierConfig())
    mem.initialise(frame_a, box)
    ver = Verifier(VerifierConfig(), mem)
    reacq = Reacquirer(ReacquireConfig(t_reacq=0.35, reacq_scales=(1.0,)), mem, ver)

    frame_b = _frame_with_texture_at(240, 180, tex)  # same target, moved
    found = reacq.search(frame_b)
    assert found is not None
    result_box, conf = found
    cx, cy = result_box.center
    assert abs(cx - 280) < 60 and abs(cy - 220) < 60


def test_reacquire_rejects_when_absent():
    tex = np.random.default_rng(6).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    mem = AppearanceMemory(VerifierConfig())
    mem.initialise(_frame_with_texture_at(60, 60, tex), BBox(60, 60, 80, 80))
    ver = Verifier(VerifierConfig(), mem)
    reacq = Reacquirer(ReacquireConfig(t_reacq=0.9, reacq_scales=(1.0,)), mem, ver)
    # A blank frame -> nothing clears the strict gate.
    assert reacq.search(np.full((300, 400, 3), 30, np.uint8)) is None


def test_top_k_distinct_dedups_and_caps():
    from config import ReacquireConfig
    from appearance import AppearanceMemory, Verifier
    cfg = ReacquireConfig(confirm_topk=3)
    reac = Reacquirer(cfg, AppearanceMemory(VerifierConfig()), None)
    # Two boxes at the same centre (different scale) + three distinct centres.
    cands = [
        (0.9, BBox(100, 100, 40, 40), None, None, 0, 0),   # centre (120,120)
        (0.8, BBox(90, 90, 60, 60), None, None, 0, 0),      # centre (120,120) - dup
        (0.7, BBox(300, 300, 40, 40), None, None, 0, 0),    # distinct
        (0.6, BBox(10, 10, 40, 40), None, None, 0, 0),      # distinct
        (0.5, BBox(200, 10, 40, 40), None, None, 0, 0),     # distinct (beyond K)
    ]
    picked = reac._top_k_distinct(cands)
    centres = [c[1].center for c in picked]
    assert len(picked) == 3                      # capped at K
    assert (120.0, 120.0) in centres             # kept the higher-score of the dup pair
    assert centres.count((120.0, 120.0)) == 1    # the duplicate centre was dropped
