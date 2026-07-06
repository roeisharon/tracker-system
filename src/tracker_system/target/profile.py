"""The target's persistent profile.

``TargetProfile`` is the memory of what we are tracking: where it is now, where
it has been, how fast it is moving, and what it looks like (a template patch).
In this phase it feeds the overlay (box + trajectory); in later phases the same
profile becomes the reference for loss detection and appearance-based
re-acquisition, which is why it is a first-class object rather than a few loose
variables.

All coordinates are stored in **full-resolution** frame space so the overlay and
any future matching operate in a single, consistent coordinate system.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from ..utils.geometry import BBox, extract_patch


@dataclass
class TargetProfile:
    """Latest known state and short history of the tracked target.

    Two appearance references are kept:

    - ``template`` — the immutable anchor captured at selection. It is the trusted
      identity reference for loss detection (via the scale-invariant HSV histogram
      similarity), and is never overwritten while tracking, so a slow drift can
      never poison it.
    - ``match_template`` — a *structural* reference for the re-acquisition template
      match (``cv2.matchTemplate``), which is NOT scale-invariant. Under the
      descending drone's zoom the anchor's pixel structure goes stale, so this one
      is refreshed (under guard, only from a confident healthy track) to the
      target's current scale. ``match_bbox`` records the box it was captured at so
      the accumulated scale drift since the last refresh can be measured.
    """

    initial_bbox: BBox
    current_bbox: BBox
    template: np.ndarray
    match_template: Optional[np.ndarray] = None
    match_bbox: Optional[BBox] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    history: List[BBox] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Default the structural match template to the anchor template so a profile
        # constructed directly (e.g. in tests) is always fully initialised.
        if self.match_template is None:
            self.match_template = self.template
        if self.match_bbox is None:
            self.match_bbox = self.current_bbox

    @classmethod
    def create(cls, frame: np.ndarray, bbox: BBox) -> "TargetProfile":
        """Build a profile from the initial selection on the first frame."""
        patch = extract_patch(frame, bbox)
        return cls(
            initial_bbox=bbox,
            current_bbox=bbox,
            template=patch,
            match_template=patch,
            match_bbox=bbox,
            velocity=(0.0, 0.0),
            history=[bbox],
        )

    @property
    def match_scale(self) -> float:
        """Scale of the current box relative to the match-template capture box.

        ``> 1`` means the target has grown since ``match_template`` was captured;
        used to centre the re-acquisition template search on the right scale.
        """
        cur = math.hypot(self.current_bbox.w, self.current_bbox.h)
        ref = math.hypot(self.match_bbox.w, self.match_bbox.h)
        return cur / ref if ref > 0 else 1.0

    def refresh_match_template(self, frame: np.ndarray, bbox: BBox) -> None:
        """Recapture the structural match template at the current scale."""
        self.match_template = extract_patch(frame, bbox)
        self.match_bbox = bbox

    def update(
        self,
        bbox: BBox,
        frame: Optional[np.ndarray] = None,
        refresh_template: bool = False,
    ) -> None:
        """Record a new confirmed position.

        Velocity is the centre displacement since the previous box. The template
        is only refreshed when asked (and a frame is provided) — this is the hook
        for "track the latest appearance, not just the initial patch" that later
        phases rely on; it stays off by default to keep the tracking loop cheap.
        """
        prev_cx, prev_cy = self.current_bbox.center
        cur_cx, cur_cy = bbox.center
        self.velocity = (cur_cx - prev_cx, cur_cy - prev_cy)
        self.current_bbox = bbox
        self.history.append(bbox)
        if refresh_template and frame is not None:
            self.template = extract_patch(frame, bbox)

    @property
    def trajectory(self) -> List[Tuple[int, int]]:
        """Centre points of the history as integer ``(x, y)`` pixels."""
        return [(int(b.center[0]), int(b.center[1])) for b in self.history]
