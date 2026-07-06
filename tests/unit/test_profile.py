"""Tests for TargetProfile."""

from __future__ import annotations

import numpy as np

from tracker_system.target.profile import TargetProfile
from tracker_system.utils.geometry import BBox


def _frame():
    return np.full((100, 100, 3), 128, dtype=np.uint8)


def test_create_captures_template_and_history():
    frame = _frame()
    box = BBox(10, 10, 20, 20)
    profile = TargetProfile.create(frame, box)
    assert profile.template.shape == (20, 20, 3)
    assert profile.current_bbox == box
    assert profile.initial_bbox == box
    assert profile.history == [box]
    assert profile.velocity == (0.0, 0.0)


def test_update_tracks_velocity_and_history():
    frame = _frame()
    profile = TargetProfile.create(frame, BBox(10, 10, 20, 20))  # centre (20, 20)
    profile.update(BBox(15, 10, 20, 20))  # centre (25, 20) -> vx=5, vy=0
    assert profile.velocity == (5.0, 0.0)
    assert profile.current_bbox == BBox(15, 10, 20, 20)
    assert len(profile.history) == 2


def test_trajectory_returns_centers():
    frame = _frame()
    profile = TargetProfile.create(frame, BBox(0, 0, 10, 10))  # centre (5, 5)
    profile.update(BBox(10, 10, 10, 10))  # centre (15, 15)
    assert profile.trajectory == [(5, 5), (15, 15)]


def test_refresh_template_updates_patch():
    frame = _frame()
    profile = TargetProfile.create(frame, BBox(0, 0, 10, 10))
    new_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    profile.update(BBox(0, 0, 30, 30), frame=new_frame, refresh_template=True)
    assert profile.template.shape == (30, 30, 3)
    assert int(profile.template.mean()) == 0  # taken from the new (black) frame
