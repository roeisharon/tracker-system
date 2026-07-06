"""Tests for configuration defaults, validation, and YAML loading."""

from __future__ import annotations

import pytest

from tracker_system.config import ConfigError, Settings, load_settings


def test_defaults_are_valid():
    s = Settings.from_mapping({})
    assert s.tracker.backend == "hybrid"
    assert s.loss.lost_patience >= 1
    assert s.reacquire.t_reacq >= s.loss.t_lost


def test_unknown_section_rejected():
    with pytest.raises(ConfigError):
        Settings.from_mapping({"nope": {}})


def test_unknown_key_rejected():
    with pytest.raises(ConfigError):
        Settings.from_mapping({"tracker": {"type": "CSRT"}})  # old key


def test_bad_backend_rejected():
    with pytest.raises(ConfigError):
        Settings.from_mapping({"tracker": {"backend": "yolo"}})


def test_verifier_weights_not_all_zero():
    with pytest.raises(ConfigError):
        Settings.from_mapping({"verifier": {"w_ncc": 0, "w_hist": 0, "w_orb": 0}})


def test_reacq_scales_coerced_from_list():
    s = Settings.from_mapping({"reacquire": {"reacq_scales": [1.0, 2.0]}})
    assert s.reacquire.reacq_scales == (1.0, 2.0)


def test_load_settings_defaults_when_missing(tmp_path):
    missing = tmp_path / "none.yaml"
    with pytest.raises(ConfigError):
        load_settings(missing)


def test_load_settings_from_file(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("tracker:\n  backend: csrt\n")
    s = load_settings(p)
    assert s.tracker.backend == "csrt"
