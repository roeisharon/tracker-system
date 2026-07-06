"""Tests for configuration loading and validation."""

from __future__ import annotations

import textwrap

import pytest

from tracker_system.config.settings import (
    ConfigError,
    Settings,
    load_settings,
)


def test_defaults_are_valid():
    settings = Settings.from_mapping({})
    assert settings.tracker.type == "CSRT"
    assert settings.video.processing_scale == 1.0
    assert settings.video.url_open_timeout_ms == 10000
    assert settings.selection.default_bbox_size == 80


def test_partial_mapping_merges_over_defaults():
    settings = Settings.from_mapping({"tracker": {"type": "KCF"}})
    assert settings.tracker.type == "KCF"
    # Untouched sections keep their defaults.
    assert settings.video.processing_scale == 1.0


def test_bundled_default_yaml_loads():
    # No path -> loads configs/default.yaml from the repo.
    settings = load_settings()
    assert settings.tracker.type == "CSRT"


def test_load_settings_from_file(tmp_path):
    cfg = tmp_path / "custom.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            video:
              processing_scale: 0.5
            selection:
              default_bbox_size: 120
            tracker:
              type: MOSSE
            """
        )
    )
    settings = load_settings(cfg)
    assert settings.video.processing_scale == 0.5
    assert settings.selection.default_bbox_size == 120
    assert settings.tracker.type == "MOSSE"


def test_missing_explicit_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "does_not_exist.yaml")


@pytest.mark.parametrize(
    "data",
    [
        {"video": {"processing_scale": 0.0}},
        {"video": {"processing_scale": 1.5}},
        {"video": {"url_open_timeout_ms": 0}},
        {"selection": {"default_bbox_size": 0}},
        {"tracker": {"type": "YOLO"}},
        {"loss": {"max_lost_frames": 0}},
        {"loss": {"max_scale_ratio": 1.0}},
        {"loss": {"min_similarity": 2.0}},
        {"loss": {"min_frame_overlap": -0.1}},
        {"loss": {"similarity_ema_alpha": 0.0}},
        {"loss": {"identity_window": 0}},
        {"loss": {"identity_k": -1.0}},
        {"reacquire": {"min_template_score": 2.0}},
        {"reacquire": {"min_hist_score": 2.0}},
        {"reacquire": {"min_motion_score": 1.5}},
        {"reacquire": {"max_region_frac": 0.0}},
        {"reacquire": {"search_scale": 0.0}},
        {"reacquire": {"search_scale": 1.5}},
    ],
)
def test_invalid_values_raise(data):
    with pytest.raises(ConfigError):
        Settings.from_mapping(data)


def test_loss_and_reacquire_defaults():
    settings = Settings.from_mapping({})
    assert settings.loss.max_lost_frames == 8
    assert settings.loss.similarity_ema_alpha == 0.3
    assert settings.reacquire.min_hist_score == 0.3
    assert settings.reacquire.min_motion_score == 0.3
    assert settings.reacquire.weight_motion == 0.5


def test_unknown_section_raises():
    with pytest.raises(ConfigError):
        Settings.from_mapping({"bogus": {}})


def test_unknown_key_raises():
    with pytest.raises(ConfigError):
        Settings.from_mapping({"tracker": {"typo": "CSRT"}})


def test_non_mapping_root_raises(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        load_settings(cfg)
