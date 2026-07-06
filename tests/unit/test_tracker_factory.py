"""Tests for the tracker capability probe and factory."""

from __future__ import annotations

import pytest

from tracker_system.tracking.factory import (
    TrackerNotAvailableError,
    available_trackers,
    create_tracker,
    known_trackers,
    probe_trackers,
)


def test_known_trackers():
    assert known_trackers() == ["CSRT", "KCF", "MOSSE"]


def test_probe_returns_bool_map():
    probed = probe_trackers()
    assert set(probed) == {"CSRT", "KCF", "MOSSE"}
    assert all(isinstance(v, bool) for v in probed.values())


def test_csrt_is_available():
    # The environment gate: the contrib build must provide CSRT.
    assert probe_trackers().get("CSRT") is True, (
        "CSRT missing — ensure opencv-contrib-python is installed, "
        "not opencv-python-headless"
    )


def test_create_available_tracker_returns_object():
    available = available_trackers()
    assert available, "no trackers available in this OpenCV build"
    tracker = create_tracker(available[0])
    assert tracker is not None
    # OpenCV trackers expose an init() method.
    assert hasattr(tracker, "init")


def test_create_unknown_tracker_raises():
    with pytest.raises(TrackerNotAvailableError):
        create_tracker("NOPE")


def test_available_is_subset_of_known():
    assert set(available_trackers()).issubset(set(known_trackers()))
