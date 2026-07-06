"""The tracking state machine and its event timeline.

Tracking is an explicit state machine, not a pile of booleans. This makes the
system's behaviour legible ("why did it do that?") and gives the overlay/debug
panel and the performance report a single source of truth.

States follow ``project-overview.md``::

    INIT -> READY -> TRACKING -> LOST -> SEARCHING -> REACQUIRED -> TRACKING

Phase 2 exercises ``INIT -> READY -> TRACKING -> LOST``. The recovery states
(SEARCHING/REACQUIRED) are defined here and permitted by the transition table so
Phase 3 can plug in without reworking this module.

Every accepted transition is appended to an in-memory timeline as a
:class:`TimelineEvent` (frame index + new state + human-readable reason), which
is exactly the event log the report and future timeline widget consume.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set

class TrackerState(Enum): #Tracking status of the state machine.
    INIT = "INIT"
    READY = "READY"
    TRACKING = "TRACKING"
    LOST = "LOST"
    SEARCHING = "SEARCHING"
    REACQUIRED = "REACQUIRED"

# Allowed forward transitions. A transition to the same state is always a no-op
# and never recorded (this is part of what prevents timeline/state chatter).
_ALLOWED: Dict[TrackerState, Set[TrackerState]] = {
    TrackerState.INIT: {TrackerState.READY},
    TrackerState.READY: {TrackerState.TRACKING},
    TrackerState.TRACKING: {TrackerState.LOST},
    TrackerState.LOST: {TrackerState.SEARCHING, TrackerState.TRACKING},
    TrackerState.SEARCHING: {TrackerState.REACQUIRED, TrackerState.LOST},
    TrackerState.REACQUIRED: {TrackerState.TRACKING, TrackerState.LOST},
}

class InvalidTransition(RuntimeError):
    """Raised when an illegal state transition is attempted."""

@dataclass(frozen=True)
class TimelineEvent:
    """A single recorded state change."""

    frame_index: int
    state: TrackerState
    reason: str

@dataclass
class StateMachine:
    """Holds the current state and records the transition timeline."""

    state: TrackerState = TrackerState.INIT
    timeline: List[TimelineEvent] = field(default_factory=list)

    def to(self, new_state: TrackerState, frame_index: int, reason: str = "") -> None:
        """Transition to ``new_state``, recording a timeline event.

        A transition to the current state is a silent no-op. An illegal
        transition raises :class:`InvalidTransition`.
        """
        if new_state == self.state:
            return
        if new_state not in _ALLOWED[self.state]:
            raise InvalidTransition(
                f"Illegal transition {self.state.value} -> {new_state.value}"
            )
        self.state = new_state
        self.timeline.append(TimelineEvent(frame_index, new_state, reason))

    def count(self, state: TrackerState) -> int:
        """Number of times ``state`` was entered over the run."""
        return sum(1 for event in self.timeline if event.state == state)
