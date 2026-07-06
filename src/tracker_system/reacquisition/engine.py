"""The re-acquisition engine that drives SEARCHING.

When the target is lost, the engine remembers its last appearance, position, and
velocity, then each frame:

1. **predicts** where the target should be (last position + velocity, capped so
   a stale velocity can't fling the guess off-screen),
2. defines a **search region** around that prediction that **expands** the longer
   the target stays missing (adaptive search), and
3. asks the :class:`Matcher` for the best candidate in that region.

If a good-enough candidate is found its box is returned (the pipeline then
reinitialises the tracker and resumes); otherwise the search continues, wider,
next frame. All of this runs only while SEARCHING, keeping normal tracking cheap.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ..config.settings import ReacquireConfig
from ..motion.ego_motion import Transform
from ..target.profile import TargetProfile
from ..utils.geometry import BBox, clamp_bbox
from .matcher import Matcher


class ReacquisitionEngine:
    """Stateful SEARCHING controller: predict, expand, match."""

    def __init__(
        self, config: ReacquireConfig, matcher: Matcher, frame_w: int, frame_h: int
    ) -> None:
        self.config = config
        self.matcher = matcher
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.diagonal = math.hypot(frame_w, frame_h)

        self._template: Optional[np.ndarray] = None
        self._appearance_floor: Optional[float] = None
        self._velocity: Tuple[float, float] = (0.0, 0.0)
        # Running prediction of the target centre and its accumulated scale drift
        # since SEARCHING began. When a confident ego transform is available the
        # centre is carried forward by the camera motion each frame (so it tracks
        # where a world-fixed target actually is); otherwise it falls back to a
        # capped constant-velocity extrapolation.
        self._predicted_point: Tuple[float, float] = (0.0, 0.0)
        self._scale_accum: float = 1.0
        self._base_scale0: float = 1.0
        self.frames_since_lost = 0
        self.last_search_region: Optional[BBox] = None
        self.last_predicted_center: Optional[Tuple[float, float]] = None
        self.last_radius: float = 0.0

    def begin(
        self, profile: TargetProfile, appearance_floor: Optional[float] = None
    ) -> None:
        """Capture the last valid target state when SEARCHING starts.

        ``appearance_floor`` is the target's learned identity threshold from loss
        detection; it raises the matcher's colour gate so re-acquisition demands
        the same identity confidence required to keep tracking (no re-lock onto
        background the loss detector would immediately reject).
        """
        self._template = profile.match_template
        self._appearance_floor = appearance_floor
        self._velocity = profile.velocity
        self._predicted_point = profile.current_bbox.center
        self._scale_accum = 1.0
        # Scale of the target at loss relative to the match-template capture box, so
        # template matching is centred on the target's current size, not 1.0.
        self._base_scale0 = profile.match_scale
        self.frames_since_lost = 0
        self.last_search_region = None

    def step(
        self, frame: np.ndarray, transform: Optional[Transform] = None
    ) -> Optional[BBox]:
        """Run one SEARCHING frame; return a box if the target is re-acquired.

        ``transform`` is the camera motion (previous->current frame). When it is
        confident the running prediction is carried forward by it; the scale
        accumulator follows the camera zoom so a target that keeps growing while
        lost is still matched at the right size.
        """
        self.frames_since_lost += 1
        predicted = self._advance_prediction(transform)
        radius = self._search_radius()
        self.last_predicted_center = predicted
        self.last_radius = radius

        # Anchor the region on the (edge-clamped) prediction and BOUND its size so
        # it stays a locality: it must never grow to cover the whole frame, or it
        # would stop constraining where the target can be. clamp_bbox then keeps
        # it inside the frame, so an edge exit yields a region hugging that edge.
        max_w = self.config.max_region_frac * self.frame_w
        max_h = self.config.max_region_frac * self.frame_h
        w = min(2.0 * radius, max_w)
        h = min(2.0 * radius, max_h)
        region = BBox(predicted[0] - w / 2.0, predicted[1] - h / 2.0, w, h)
        self.last_search_region = clamp_bbox(region, self.frame_w, self.frame_h)

        candidate = self.matcher.find(
            frame, self.last_search_region, self._template, predicted, self.diagonal,
            appearance_floor=self._appearance_floor,
            base_scale=self._base_scale0 * self._scale_accum,
        )
        return candidate.bbox if candidate is not None else None

    # -- internals ---------------------------------------------------------

    def _advance_prediction(
        self, transform: Optional[Transform]
    ) -> Tuple[float, float]:
        """Advance the running prediction one frame and return it (edge-clamped)."""
        if transform is not None and transform.confidence > 0:
            self._predicted_point = transform.apply_point(self._predicted_point)
            self._scale_accum *= transform.scale
        elif self.frames_since_lost <= self.config.max_prediction_frames:
            # No ego estimate: fall back to a (capped) constant-velocity carry.
            self._predicted_point = (
                self._predicted_point[0] + self._velocity[0],
                self._predicted_point[1] + self._velocity[1],
            )
        cx = min(max(self._predicted_point[0], 0.0), self.frame_w - 1.0)
        cy = min(max(self._predicted_point[1], 0.0), self.frame_h - 1.0)
        return (cx, cy)

    def _search_radius(self) -> float:
        base = self.config.search_radius_frac * self.diagonal
        grow = self.config.search_expansion_frac * self.diagonal * self.frames_since_lost
        cap = self.config.max_search_radius_frac * self.diagonal
        return min(base + grow, cap)
