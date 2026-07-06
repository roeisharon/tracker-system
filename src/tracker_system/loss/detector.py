"""Detects when tracking has failed.

The detector runs *cheap* per-frame checks during normal tracking (the roadmap's
rule: keep TRACKING light, reserve heavy work for SEARCHING). Each frame is
judged healthy or given a reason it looks bad:

1. tracker failure       — the tracker itself reported no box
2. left frame            — too little of the box remains inside the frame
3. motion jump           — the centre teleported (likely latched onto a distractor)
4. scale change          — the box area exploded/collapsed in one step
5. low similarity        — periodic, cheap appearance check vs the reference

A single bad frame is not a loss. The detector counts *consecutive* bad frames
and only confirms LOST once the count reaches ``max_lost_frames``. This
hysteresis window is what keeps momentary tracker wobble or ego-motion from
causing false positives and state chatter; a single healthy frame resets it.
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import List, Optional, Tuple
import numpy as np

# Scale factor making the median-absolute-deviation a consistent estimator of the
# standard deviation for normally distributed data.
_MAD_TO_STD = 1.4826


def robust_identity_threshold(samples: List[float], k: float) -> Tuple[float, float]:
    """Return ``(threshold, center)`` for the target's similarity distribution.

    ``center`` is the median; the threshold is ``median - k * robust_std`` where
    ``robust_std`` is derived from the median absolute deviation (MAD). A tight
    distribution (stable target) yields a threshold close to the median (strict);
    a wide distribution (noisy target) yields a low/negative threshold (tolerant),
    which is what makes appearance a weak signal for noisy targets.
    """
    center = float(median(samples))
    mad = float(median([abs(x - center) for x in samples]))
    robust_std = _MAD_TO_STD * mad
    return center - k * robust_std, center

from ..config.settings import LossConfig
from ..utils.appearance import appearance_similarity
from ..utils.geometry import BBox, extract_patch, frame_overlap_ratio

# Reasons (kept as short constants so overlay/report strings stay consistent).
REASON_TRACKER_FAILURE = "tracker failure"
REASON_LEFT_FRAME = "left frame"
REASON_MOTION_JUMP = "motion jump"
REASON_SCALE_CHANGE = "scale change"
REASON_LOW_SIMILARITY = "low similarity"
REASON_IDENTITY_LOW = "identity confidence low"


@dataclass(frozen=True)
class LossAssessment:
    """The verdict for one frame."""

    healthy: bool
    confirmed_lost: bool
    reason: Optional[str]
    consecutive_bad: int


class LossDetector:
    """Per-frame loss assessment with a consecutive-bad-frame confirmation window."""

    def __init__(self, config: LossConfig, frame_w: int, frame_h: int) -> None:
        self.config = config
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._diagonal = math.hypot(frame_w, frame_h)
        self._bad = 0
        self._sim_ema: Optional[float] = None
        # Adaptive identity distribution — collected during sustained genuine
        # tracking and PERSISTED across re-acquisitions (never re-seeded onto the
        # background). The gate is derived from this distribution's centre+spread.
        self._id_window: deque = deque(maxlen=max(1, config.identity_window))
        self._consecutive_healthy = 0
        self._identity_threshold: Optional[float] = None
        self._identity_center: Optional[float] = None
        self._identity_frozen = False
        self._last_similarity: Optional[float] = None
        # Diagnostics from the most recent assess() (for explainability/debug).
        self.last_metrics: dict = {}

    @property
    def identity_threshold(self) -> Optional[float]:
        """The target's learned identity gate (``median - k*MAD``), or ``None``.

        Exposed so re-acquisition can apply the *same* per-target identity
        standard used to keep tracking. ``None`` until enough genuine-track samples
        have been collected (then re-acquisition falls back to the config gate).
        """
        return self._identity_threshold

    def reset(self) -> None:
        """Reset per-segment loss state on re-acquisition.

        The confirmation counter and the fast EMA are reset (the EMA is seeded to
        the learned distribution centre so a fresh re-acquisition gets a brief
        grace period), and the consecutive-healthy counter restarts so a short
        false lock does not contribute background samples. The learned identity
        distribution itself PERSISTS: the re-acquired track must still match the
        original target's own confidence level.
        """
        self._bad = 0
        self._sim_ema = self._identity_center
        self._consecutive_healthy = 0

    def assess(
        self,
        *,
        tracker_ok: bool,
        bbox: Optional[BBox],
        prev_bbox: Optional[BBox],
        frame: Optional[np.ndarray],
        reference_template: Optional[np.ndarray],
        frame_index: int,
        ego_bbox: Optional[BBox] = None,
    ) -> LossAssessment:
        """Judge one frame and update the confirmation counter.

        ``ego_bbox`` (when given) is where the camera-motion estimate predicts the
        box should be this frame; the motion-jump and scale-change checks measure
        the tracker's *residual* deviation from it, so a legitimate camera pan/zoom
        no longer looks like a jump. It falls back to ``prev_bbox`` when no
        confident ego estimate is available.
        """
        reason = self._instantaneous_reason(
            tracker_ok=tracker_ok,
            bbox=bbox,
            prev_bbox=prev_bbox,
            frame=frame,
            reference_template=reference_template,
            frame_index=frame_index,
            ego_bbox=ego_bbox,
        )

        # Learn the target's similarity distribution, but ONLY from a sustained
        # healthy track (so a brief false lock cannot inject background samples).
        self._update_identity_distribution(healthy=reason is None)

        if reason is None:
            self._bad = 0
            return LossAssessment(True, False, None, 0)

        self._bad += 1
        confirmed = self._bad >= self.config.max_lost_frames
        return LossAssessment(False, confirmed, reason, self._bad)

    def _instantaneous_reason(
        self,
        *,
        tracker_ok: bool,
        bbox: Optional[BBox],
        prev_bbox: Optional[BBox],
        frame: Optional[np.ndarray],
        reference_template: Optional[np.ndarray],
        frame_index: int,
        ego_bbox: Optional[BBox] = None,
    ) -> Optional[str]:
        cfg = self.config

        # Always compute the cheap diagnostic metrics so they are available for
        # explainability, independent of which check (if any) trips first.
        overlap = (
            frame_overlap_ratio(bbox, self.frame_w, self.frame_h)
            if bbox is not None
            else 0.0
        )
        # Measure the jump/scale RESIDUAL against the ego-motion prediction when
        # available (so camera pan/zoom is not mistaken for a tracker jump), else
        # against the previous box (the pre-ego-compensation behaviour).
        reference = ego_bbox if ego_bbox is not None else prev_bbox
        center_jump = None
        scale_ratio = None
        if bbox is not None and reference is not None:
            (px, py), (cx, cy) = reference.center, bbox.center
            center_jump = math.hypot(cx - px, cy - py)
            if reference.area > 0 and bbox.area > 0:
                scale_ratio = max(bbox.area / reference.area, reference.area / bbox.area)
        similarity = None
        similarity_fast = None
        below_floor = False
        identity_low = False
        if bbox is not None and frame is not None and reference_template is not None:
            similarity = appearance_similarity(extract_patch(frame, bbox), reference_template)
            # Fast EMA of the similarity (smoothed current level).
            if self._sim_ema is None:
                self._sim_ema = similarity
            else:
                a = cfg.similarity_ema_alpha
                self._sim_ema = a * similarity + (1.0 - a) * self._sim_ema
            similarity_fast = self._sim_ema

            below_floor = similarity_fast < cfg.min_similarity
            # Adaptive identity gate learned from THIS target's own distribution.
            identity_low = (
                self._identity_threshold is not None
                and similarity_fast < self._identity_threshold
            )

        self._last_similarity = similarity
        self.last_metrics = {
            "overlap": overlap,
            "center_jump": center_jump,
            "scale_ratio": scale_ratio,
            "similarity": similarity,
            "similarity_fast": similarity_fast,
            "identity_center": self._identity_center,
            "identity_threshold": self._identity_threshold,
        }

        if not tracker_ok or bbox is None:
            return REASON_TRACKER_FAILURE

        if overlap < cfg.min_frame_overlap:
            return REASON_LEFT_FRAME

        if center_jump is not None and center_jump > cfg.max_center_jump_frac * self._diagonal:
            return REASON_MOTION_JUMP

        if scale_ratio is not None and scale_ratio > cfg.max_scale_ratio:
            return REASON_SCALE_CHANGE

        if below_floor:
            return REASON_LOW_SIMILARITY

        # Identity confidence has fallen below the target's own learned level:
        # the track is no longer convincingly the original object -> back to LOST.
        if identity_low:
            return REASON_IDENTITY_LOW

        return None

    def freeze_identity(self) -> None:
        """Stop learning the identity distribution (called on the first loss).

        The distribution is learned only from the target's initial genuine
        tracking period; freezing it at the first confirmed loss prevents any
        later (possibly wrong) tracking on the background from poisoning the
        reference and slowly turning a strict gate into a permissive one.
        """
        self._identity_frozen = True

    def _update_identity_distribution(self, *, healthy: bool) -> None:
        """Feed the target's own similarity into the learned distribution.

        Only samples from a SUSTAINED healthy track during the INITIAL tracking
        period (before the first loss freezes it) are collected. Once enough
        samples exist the gate is (re)derived from the distribution's centre and
        spread.
        """
        cfg = self.config
        if self._identity_frozen:
            return
        if not healthy:
            self._consecutive_healthy = 0
            return

        self._consecutive_healthy += 1
        if (
            self._last_similarity is not None
            and self._consecutive_healthy >= cfg.identity_stable_frames
        ):
            self._id_window.append(self._last_similarity)
            if len(self._id_window) >= cfg.identity_min_samples:
                self._identity_threshold, self._identity_center = (
                    robust_identity_threshold(list(self._id_window), cfg.identity_k)
                )


# Re-exported for callers/tests that import it from this module.
__all__ = [
    "LossAssessment",
    "LossDetector",
    "appearance_similarity",
    "REASON_TRACKER_FAILURE",
    "REASON_LEFT_FRAME",
    "REASON_MOTION_JUMP",
    "REASON_SCALE_CHANGE",
    "REASON_LOW_SIMILARITY",
    "REASON_IDENTITY_LOW",
]
