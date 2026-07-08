"""Loss detection: fused-confidence hysteresis + cheap box sanity.

Each TRACKING frame is judged healthy or given a reason it looks bad:

  1. left frame    - too little of the box remains inside the frame
  2. scale change  - box area exploded/collapsed in one step (tracker glitch)
  3. low confidence - fused confidence (tracker score + appearance) below ``t_lost``

A single bad frame is not a loss: consecutive bad frames are counted and LOST is
confirmed only at ``lost_patience`` (hysteresis stops state chatter). One healthy
frame resets the counter.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from config import LossConfig
from geometry import BBox, frame_overlap_ratio

REASON_LEFT_FRAME = "left frame"
REASON_SCALE_CHANGE = "scale change"
REASON_LOW_CONFIDENCE = "low confidence"


@dataclass(frozen=True)
class LossAssessment:
    """One frame's verdict: healthy vs. bad, whether loss is confirmed, and why."""

    healthy: bool              # True = this frame looks on-target
    confirmed_lost: bool       # True = enough bad frames in a row to declare LOST
    reason: Optional[str]      # why it looked bad (None when healthy)
    consecutive_bad: int       # how many bad frames have accumulated so far


class LossDetector:
    """Turns a per-frame verdict into a debounced LOST decision (hysteresis)."""

    def __init__(self, config: LossConfig, frame_w: int, frame_h: int) -> None:
        self.config = config
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._bad = 0            # running count of consecutive bad frames

    def reset(self) -> None:
        """Clear the bad-frame counter (called after a successful re-acquisition)."""
        self._bad = 0

    def assess(self, confidence: float, bbox: Optional[BBox],
               prev_bbox: Optional[BBox]) -> LossAssessment:
        """Judge one frame; a healthy frame resets the counter, a bad one grows it."""
        reason = self._instantaneous_reason(confidence, bbox, prev_bbox)
        if reason is None:
            self._bad = 0
            return LossAssessment(True, False, None, 0)
        # Bad frame: LOST is confirmed only once the streak reaches lost_patience.
        self._bad += 1
        return LossAssessment(False, self._bad >= self.config.lost_patience, reason, self._bad)

    def _instantaneous_reason(self, confidence, bbox, prev_bbox) -> Optional[str]:
        """Why this single frame looks bad, or None if it looks fine."""
        # Cheapest/most-conclusive checks first; the first one that trips wins.
        cfg = self.config
        if bbox is None:
            return REASON_LOW_CONFIDENCE
        if frame_overlap_ratio(bbox, self.frame_w, self.frame_h) < cfg.min_frame_overlap:
            return REASON_LEFT_FRAME
        if prev_bbox is not None and prev_bbox.area > 0 and bbox.area > 0:
            # Bigger-over-smaller either way, so one threshold catches a sudden
            # explosion or collapse of the box (a tracker glitch).
            ratio = max(bbox.area / prev_bbox.area, prev_bbox.area / bbox.area)
            if ratio > cfg.max_scale_ratio:
                return REASON_SCALE_CHANGE
        if confidence < cfg.t_lost:
            return REASON_LOW_CONFIDENCE
        return None
