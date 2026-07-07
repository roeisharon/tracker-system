"""Tests for configuration defaults and validation (validated on construction)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tracker_system.config import ConfigError, Settings, TrackerConfig, VerifierConfig


def test_defaults_are_valid():
    s = Settings()
    assert s.tracker.backend == "hybrid"
    assert s.loss.lost_patience >= 1
    assert s.reacquire.t_reacq >= s.loss.t_lost


def test_bad_backend_rejected():
    with pytest.raises(ConfigError):
        replace(Settings(), tracker=replace(TrackerConfig(), backend="yolo"))


def test_verifier_weights_not_all_zero_rejected():
    with pytest.raises(ConfigError):
        replace(Settings(), verifier=replace(VerifierConfig(), w_ncc=0, w_hist=0, w_orb=0))


def test_bad_processing_scale_rejected():
    from tracker_system.config import VideoConfig
    with pytest.raises(ConfigError):
        replace(Settings(), video=replace(VideoConfig(), processing_scale=0.0))


def test_reacq_scales_default_is_tuple():
    assert isinstance(Settings().reacquire.reacq_scales, tuple)
