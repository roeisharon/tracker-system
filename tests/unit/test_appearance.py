"""Tests for appearance memory + multi-cue verifier."""

from __future__ import annotations

import numpy as np
import pytest

from tracker_system.appearance import AppearanceMemory, Verifier, hist_similarity
from tracker_system.config import VerifierConfig
from tracker_system.geometry import BBox


def _frame_with_patch(seed=1):
    rng = np.random.default_rng(seed)
    frame = np.full((200, 200, 3), 40, np.uint8)
    frame[60:140, 60:140] = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    return frame


def test_memory_initialises_anchor_and_recent():
    mem = AppearanceMemory(VerifierConfig())
    mem.initialise(_frame_with_patch(), BBox(60, 60, 80, 80))
    assert mem.anchor is not None and mem.recent is mem.anchor
    assert len(mem.templates()) == 1


def test_verifier_high_on_same_patch_low_on_background():
    frame = _frame_with_patch()
    box = BBox(60, 60, 80, 80)
    mem = AppearanceMemory(VerifierConfig())
    mem.initialise(frame, box)
    ver = Verifier(VerifierConfig(), mem)
    same, _ = ver.appearance_confidence(frame, box, force_orb=True)
    bg, _ = ver.appearance_confidence(frame, BBox(0, 0, 40, 40), force_orb=True)
    assert same > bg


def test_recent_updates_only_when_confident():
    frame = _frame_with_patch()
    box = BBox(60, 60, 80, 80)
    mem = AppearanceMemory(VerifierConfig(ema_update_conf=0.6, tmpl_update_score=0.7))
    mem.initialise(frame, box)
    anchor = mem.recent
    mem.update(frame, BBox(0, 0, 40, 40), None, confidence=0.1, tracker_score=0.1)
    assert mem.recent is anchor  # low confidence -> no drift
    mem.update(_frame_with_patch(2), box, None, confidence=0.9, tracker_score=0.9)
    assert mem.recent is not anchor  # high confidence -> learns a recent template


def test_fuse_with_tracker_blend():
    ver = Verifier(VerifierConfig(w_tracker=0.5), AppearanceMemory(VerifierConfig()))
    assert ver.fuse_with_tracker(0.8, 0.4) == pytest.approx(0.6)


def test_hist_similarity_identity():
    frame = _frame_with_patch()
    mem = AppearanceMemory(VerifierConfig())
    t = mem.extract(frame, BBox(60, 60, 80, 80))
    assert hist_similarity(t.hist, t.hist) == pytest.approx(1.0)


def test_max_patch_bounds_descriptor_size():
    cfg = VerifierConfig(max_patch=64)
    mem = AppearanceMemory(cfg)
    big = np.random.default_rng(3).integers(0, 255, (600, 600, 3), dtype=np.uint8)
    t = mem.extract(big, BBox(0, 0, 600, 600))
    assert t.gray.shape[0] <= 64 and t.gray.shape[1] <= 64
    assert t.size == (600, 600)  # original box size is preserved
