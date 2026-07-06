"""Candidate generation and scoring for re-acquisition.

Given a search region, a reference template, and a motion-predicted position,
the matcher proposes candidate boxes and scores each by a weighted blend of:

- **multi-scale template match** — how strongly the appearance matches, tried at
  several scales so a target that grew/shrank on re-entry is still found;
- **colour-histogram similarity** — a scale-invariant appearance cross-check;
- **motion prior** — a Gaussian bonus for candidates near where the target was
  predicted to be, which is what disambiguates the true target from identical
  distractors in repetitive terrain.

The best candidate is returned only if its combined score clears ``min_score``;
otherwise re-acquisition holds off (the false-lock guard). This work is only ever
run while SEARCHING.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config.settings import ReacquireConfig
from ..utils.appearance import appearance_similarity
from ..utils.geometry import BBox, clamp_bbox, extract_patch


@dataclass(frozen=True)
class Candidate:
    """A scored re-acquisition candidate (box in full-frame coordinates)."""

    bbox: BBox
    template_score: float
    hist_score: float
    motion_score: float
    score: float


def motion_prior(
    center: Tuple[float, float],
    predicted_center: Optional[Tuple[float, float]],
    diagonal: float,
    sigma_frac: float,
) -> float:
    """Gaussian motion prior in ``[0, 1]`` (1.0 at the predicted position)."""
    if predicted_center is None:
        return 1.0
    sigma = max(sigma_frac * diagonal, 1.0)
    dist = math.hypot(center[0] - predicted_center[0], center[1] - predicted_center[1])
    return math.exp(-0.5 * (dist / sigma) ** 2)


def weighted_score(
    template_score: float,
    hist_score: float,
    motion_score: float,
    config: ReacquireConfig,
) -> float:
    """Normalised weighted blend of the three identity signals."""
    total = config.weight_template + config.weight_histogram + config.weight_motion
    return (
        config.weight_template * template_score
        + config.weight_histogram * hist_score
        + config.weight_motion * motion_score
    ) / total


class Matcher:
    """Generates and scores re-acquisition candidates within a search region."""

    def __init__(self, config: ReacquireConfig) -> None:
        self.config = config
        # Diagnostics from the most recent find() (for explainability/debug).
        self.last_candidates: List[Candidate] = []
        self.last_best: Optional[Candidate] = None
        self.last_accepted: Optional[Candidate] = None
        self.last_hist_gate: float = config.min_hist_score

    def find(
        self,
        frame: np.ndarray,
        search_region: BBox,
        template: np.ndarray,
        predicted_center: Optional[Tuple[float, float]],
        diagonal: float,
        appearance_floor: Optional[float] = None,
        base_scale: float = 1.0,
    ) -> Optional[Candidate]:
        """Return the best identity-matched candidate, or ``None``.

        This is the explicit *identity-matching* stage. A candidate is not merely
        "something that looks like the target" (detection); it must be the *same
        instance*, which for a moving object is defined by both:

        - **appearance validity** — it looks like the target in structure
          (``min_template_score``) AND colour (at least ``min_hist_score``), and
        - **spatial validity** — it is where the target is predicted to be
          (``min_motion_score``).

        Both are hard gates: a candidate failing either is rejected outright, no
        matter how strong the other signals are. This is what prevents locking
        onto an identical look-alike elsewhere in the frame. Survivors are ranked
        by the weighted score (in which spatial continuity is the primary term)
        and the best is accepted only above ``min_score``. If nothing qualifies,
        the caller keeps SEARCHING.

        ``appearance_floor`` raises the colour gate to the target's *own* learned
        identity level (from loss detection), so re-acquisition applies the **same
        identity standard used to keep tracking**. Without it, a distractor that is
        good enough to re-lock (``hist >= min_hist_score``) but not good enough to
        keep (``hist < learned threshold``) causes a re-lock/lose thrash onto
        background — the drone's sand being the canonical case.

        ``base_scale`` centres the multi-scale template search on the target's
        current size (the accumulated camera zoom since the template was captured),
        so a target that grew manyfold during the drone's descent is still matched.
        """
        candidates = self._score_candidates(
            frame, search_region, template, predicted_center, diagonal, base_scale
        )
        self.last_candidates = candidates
        self.last_best = max(candidates, key=lambda c: c.score) if candidates else None
        self.last_hist_gate = self._hist_gate(appearance_floor)
        self.last_accepted = None

        eligible = self.identity_matches(candidates, appearance_floor)
        if not eligible:
            return None
        best = max(eligible, key=lambda c: c.score)
        if best.score >= self.config.min_score:
            self.last_accepted = best
            return best
        return None

    def _hist_gate(self, appearance_floor: Optional[float]) -> float:
        """Effective colour gate: the config floor, raised to the learned level."""
        if appearance_floor is None:
            return self.config.min_hist_score
        return max(self.config.min_hist_score, appearance_floor)

    def identity_matches(
        self, candidates: List[Candidate], appearance_floor: Optional[float] = None
    ) -> List[Candidate]:
        """Candidates passing the appearance (template+hist) and spatial gates."""
        cfg = self.config
        hist_gate = self._hist_gate(appearance_floor)
        return [
            c
            for c in candidates
            if c.template_score >= cfg.min_template_score
            and c.hist_score >= hist_gate
            and c.motion_score >= cfg.min_motion_score
        ]

    # -- internals ---------------------------------------------------------

    def _score_candidates(
        self, frame, search_region, template, predicted_center, diagonal, base_scale=1.0
    ) -> List[Candidate]:
        scored: List[Candidate] = []
        for bbox, template_score in self._raw_candidates(
            frame, search_region, template, base_scale
        ):
            patch = extract_patch(frame, bbox)
            hist_score = appearance_similarity(patch, template)
            m_score = motion_prior(
                bbox.center, predicted_center, diagonal, self.config.motion_sigma_frac
            )
            total = weighted_score(template_score, hist_score, m_score, self.config)
            scored.append(Candidate(bbox, template_score, hist_score, m_score, total))
        return scored

    def _raw_candidates(
        self, frame, search_region, template, base_scale=1.0
    ) -> List[Tuple[BBox, float]]:
        """Multi-scale template matching -> (bbox, template_score) proposals.

        The expensive ``matchTemplate`` runs on a **downscaled** ROI/template
        (``search_scale``) for speed, but every proposed box is reported back in
        **full-resolution** frame coordinates, so predicted-position, motion, and
        histogram scoring — and all overlays/diagnostics — stay in native space.

        ``base_scale`` multiplies every configured scale, so the search is centred
        on the target's current size rather than the template-capture size.
        """
        frame_h, frame_w = frame.shape[:2]
        region = clamp_bbox(search_region, frame_w, frame_h)
        rx, ry, rw, rh = region.as_int_xywh()
        roi = frame[ry : ry + rh, rx : rx + rw]
        if roi.size == 0:
            return []

        ss = self.config.search_scale
        if ss < 1.0:
            roi_m = cv2.resize(
                roi,
                (max(1, int(round(rw * ss))), max(1, int(round(rh * ss)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            roi_m = roi

        t_h, t_w = template.shape[:2]
        out: List[Tuple[BBox, float]] = []
        for cfg_scale in self.config.scales:
            scale = cfg_scale * base_scale                 # centre on current size
            tw = max(1, int(round(t_w * scale)))          # full-res template size
            th = max(1, int(round(t_h * scale)))
            tw_m = max(1, int(round(tw * ss)))            # matching (downscaled) size
            th_m = max(1, int(round(th * ss)))
            if tw_m > roi_m.shape[1] or th_m > roi_m.shape[0]:
                continue  # template larger than the (downscaled) search ROI
            scaled = cv2.resize(template, (tw_m, th_m))
            result = cv2.matchTemplate(roi_m, scaled, cv2.TM_CCOEFF_NORMED)
            min_distance = max(4, min(tw_m, th_m) // 2)
            for (lx, ly), value in _peaks(result, self.config.max_candidates, min_distance):
                # Map the downscaled-ROI peak back to full-resolution frame coords.
                fx = rx + int(round(lx / ss))
                fy = ry + int(round(ly / ss))
                out.append((BBox(float(fx), float(fy), float(tw), float(th)), value))
        return out


def _peaks(result: np.ndarray, k: int, min_distance: int) -> List[Tuple[Tuple[int, int], float]]:
    """Top-``k`` local maxima of a match map with simple non-max suppression."""
    peaks: List[Tuple[Tuple[int, int], float]] = []
    work = result.copy()
    for _ in range(k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= -1.0:
            break
        peaks.append((max_loc, float(max_val)))
        x, y = max_loc
        x1, y1 = max(0, x - min_distance), max(0, y - min_distance)
        x2 = min(work.shape[1], x + min_distance + 1)
        y2 = min(work.shape[0], y + min_distance + 1)
        work[y1:y2, x1:x2] = -1.0
    return peaks
