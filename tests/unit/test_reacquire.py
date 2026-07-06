"""Tests for appearance-confirmed re-acquisition."""

from __future__ import annotations

import numpy as np

from tracker_system.appearance import AppearanceMemory, Verifier
from tracker_system.config import ReacquireConfig, VerifierConfig
from tracker_system.geometry import BBox
from tracker_system.reacquire import Reacquirer


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
