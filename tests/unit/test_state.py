"""Tests for the tracking state machine."""

from __future__ import annotations

import pytest

from tracker_system.app.state import (
    InvalidTransition,
    StateMachine,
    TrackerState,
)


def test_starts_in_init_with_empty_timeline():
    machine = StateMachine()
    assert machine.state == TrackerState.INIT
    assert machine.timeline == []


def test_happy_path_transitions_are_recorded():
    machine = StateMachine()
    machine.to(TrackerState.READY, 0, "selected")
    machine.to(TrackerState.TRACKING, 0, "started")
    machine.to(TrackerState.LOST, 42, "left frame")
    assert machine.state == TrackerState.LOST
    assert [e.state for e in machine.timeline] == [
        TrackerState.READY,
        TrackerState.TRACKING,
        TrackerState.LOST,
    ]
    assert machine.timeline[-1].frame_index == 42
    assert machine.timeline[-1].reason == "left frame"


def test_same_state_transition_is_noop():
    machine = StateMachine()
    machine.to(TrackerState.READY, 0)
    machine.to(TrackerState.TRACKING, 1)
    machine.to(TrackerState.TRACKING, 2)  # no-op
    assert len(machine.timeline) == 2


def test_illegal_transition_raises():
    machine = StateMachine()
    with pytest.raises(InvalidTransition):
        machine.to(TrackerState.LOST, 0)  # INIT -> LOST is illegal


def test_recovery_transitions_allowed():
    # Phase 3 path is permitted by the table already.
    machine = StateMachine()
    machine.to(TrackerState.READY, 0)
    machine.to(TrackerState.TRACKING, 0)
    machine.to(TrackerState.LOST, 10, "gone")
    machine.to(TrackerState.SEARCHING, 11, "searching")
    machine.to(TrackerState.REACQUIRED, 20, "found")
    machine.to(TrackerState.TRACKING, 21, "resumed")
    assert machine.state == TrackerState.TRACKING


def test_count_counts_entries():
    machine = StateMachine()
    machine.to(TrackerState.READY, 0)
    machine.to(TrackerState.TRACKING, 0)
    machine.to(TrackerState.LOST, 5)
    assert machine.count(TrackerState.LOST) == 1
    assert machine.count(TrackerState.SEARCHING) == 0
