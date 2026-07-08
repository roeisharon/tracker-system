"""Appearance memory + independent multi-cue verifier.

A deep tracker can drift onto a distractor while still reporting a high score, so
every box is cross-checked against an appearance memory with three cues that fail
in different ways:

  * NCC  - grayscale normalised cross-correlation (structure/texture, scale-normalised)
  * HIST - HSV hue-saturation histogram          (colour)
  * ORB  - feature matches + RANSAC inliers      (distinctive geometry; the cue
           the old colour+coarse-NCC model lacked)

The memory keeps an **anchor** (the original selection, never updated -> can't
drift) plus a **recent** template (EMA, refreshed only from confident frames ->
follows gradual appearance/scale change). Cues are scored as the best match over
both templates and fused into a confidence in ``[0, 1]``.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
from config import VerifierConfig
from geometry import BBox, clamp_bbox

_CANON = (64, 64)          # canonical size for scale-invariant NCC
# ORB builds an 8-level 1.2x pyramid; on a near-collapsed box a level rounds to
# 0 px and cv2.resize asserts. Skip the cue below this side length (too small to
# yield meaningful features anyway).
_ORB_MIN_SIDE = 8
_HIST_BINS = [50, 60]
_HIST_RANGES = [0, 180, 0, 256]
_HIST_CHANNELS = [0, 1]


@dataclass
class Template:
    """A stored snapshot of the target's look — the inputs the three cues compare against."""

    gray: np.ndarray                    # grayscale patch (for NCC)
    hist: np.ndarray                    # HSV colour histogram
    keypoints: tuple                    # ORB keypoints
    descriptors: Optional[np.ndarray]   # ORB descriptors (None if too small/blank)
    size: Tuple[int, int]               # original box size in px (w, h)
    # Canonical-size copies of ``gray`` rotated at every step (anchor only) — makes
    # the NCC identity cue rotation-tolerant. None = score gray directly.
    variants: Optional[List[np.ndarray]] = None


def _rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a grayscale patch about its centre (edges extended, not blacked out)."""
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _rot_variants(gray: np.ndarray, step: int) -> Optional[List[np.ndarray]]:
    """Pre-rotate a canonical patch at every ``step`` degrees (for rotation-tolerant NCC)."""
    if step <= 0 or gray.size == 0:
        return None
    base = cv2.resize(gray, _CANON)
    return [_rotate(base, a) for a in range(0, 360, step)]


def orb_matcher() -> cv2.BFMatcher:
    """Brute-force Hamming matcher for ORB descriptors (cross-checked matches)."""
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 - Bhattacharyya distance, in [0, 1] (higher = more similar)."""
    return float(max(0.0, 1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)))


def _sub_mask(hud_mask: Optional[np.ndarray], x, y, w, h) -> Optional[np.ndarray]:
    # Usable = scene pixels (HUD excluded) inside the box.
    if hud_mask is None:
        return None
    return cv2.bitwise_not(hud_mask[y:y + h, x:x + w])


class AppearanceMemory:
    """Immutable anchor + confidence-gated EMA recent template."""

    def __init__(self, cfg: VerifierConfig) -> None:
        """Set up the ORB detector and empty template slots."""
        self.cfg = cfg
        self._orb = cv2.ORB_create(nfeatures=cfg.orb_nfeatures, scaleFactor=1.2, nlevels=8)
        self.anchor: Optional[Template] = None
        self.recent: Optional[Template] = None
        # Gallery of past-appearance snapshots (spans scale change) for re-acquire.
        self.snapshots: List[Template] = []
        self._snap_diag: Optional[float] = None

    def extract(self, frame: np.ndarray, bbox: BBox, hud_mask=None) -> Template:
        """Crop the box and precompute its identity fingerprints (gray, hist, ORB)."""
        fh, fw = frame.shape[:2]
        x, y, w, h = clamp_bbox(bbox, fw, fh).as_int_xywh()
        patch = frame[y:y + h, x:x + w]
        sm = _sub_mask(hud_mask, x, y, w, h)  # per-patch mask that excludes HUD pixels
        # Cap the descriptor patch size so cost stays bounded once the box fills
        # the frame — identity doesn't need full resolution of a huge box.
        mx = self.cfg.max_patch
        if max(w, h) > mx:
            f = mx / max(w, h)
            patch = cv2.resize(patch, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
            if sm is not None:
                sm = cv2.resize(sm, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], _HIST_CHANNELS, sm, _HIST_BINS, _HIST_RANGES)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        if min(gray.shape[:2]) >= _ORB_MIN_SIDE:
            kp, desc = self._orb.detectAndCompute(gray, sm)
        else:
            kp, desc = (), None
        return Template(gray, hist, tuple(kp), desc, (w, h))

    def initialise(self, frame, bbox: BBox, hud_mask=None) -> None:
        """Set the frozen anchor template from the user's first selection."""
        self.anchor = self.extract(frame, bbox, hud_mask)
        # Pre-rotate the anchor so the NCC cue can match a rotated view of the target.
        self.anchor.variants = _rot_variants(self.anchor.gray, self.cfg.rot_ncc_step)
        self.recent = self.anchor
        self.snapshots = []
        self._snap_diag = float(np.hypot(bbox.w, bbox.h))  # box diagonal at last snapshot

    def update(self, frame, bbox: BBox, hud_mask, confidence: float, tracker_score: float = 0.0) -> None:
        """Refresh the recent template when the frame is confidently on-target.

        Gating on EITHER the fused confidence OR the raw tracker score breaks the
        deadlock (confidence needs a fresh template to stay high, but was the gate
        for refreshing it). Also snapshots the current appearance into the gallery
        on a significant scale change (for re-acquisition).
        """
        if confidence < self.cfg.ema_update_conf and tracker_score < self.cfg.tmpl_update_score:
            return
        fresh = self.extract(frame, bbox, hud_mask)
        self._maybe_snapshot(fresh, bbox)
        new = fresh
        # Ease the recent template toward the new look (EMA) instead of snapping to it,
        # so one odd frame can't hijack identity. Only possible when sizes match.
        if self.recent is not None and self.recent.gray.shape == new.gray.shape:
            blended = cv2.addWeighted(self.recent.gray, 1 - self.cfg.ema_alpha,
                                      new.gray, self.cfg.ema_alpha, 0)
            new = Template(blended, new.hist, new.keypoints, new.descriptors, new.size)
        self.recent = new

    def _maybe_snapshot(self, tmpl: Template, bbox: BBox) -> None:
        """Add a gallery snapshot once the box has grown/shrunk by a scale step.

        Tracks the box diagonal since the last snapshot; when it changes by
        ``snapshot_scale_step`` (either way), keep this template. The gallery is
        FIFO-capped, so it always spans the target's recent range of appearances.
        """
        if self.cfg.max_snapshots <= 0:
            return
        diag = float(np.hypot(bbox.w, bbox.h))
        if self._snap_diag is None:
            self._snap_diag = diag
            return
        step = self.cfg.snapshot_scale_step
        if diag / self._snap_diag >= step or diag / self._snap_diag <= 1.0 / step:
            self.snapshots.append(tmpl)
            if len(self.snapshots) > self.cfg.max_snapshots:
                self.snapshots.pop(0)  # drop the oldest to stay within max_snapshots
            self._snap_diag = diag

    def templates(self) -> List[Template]:
        """Anchor + recent — the cheap per-frame set for loss detection."""
        out = []
        if self.anchor is not None:
            out.append(self.anchor)
        if self.recent is not None and self.recent is not self.anchor:
            out.append(self.recent)
        return out

    def reacq_templates(self) -> List[Template]:
        """Anchor + gallery snapshots + recent — the fuller set for re-acquisition,
        so a target that changed appearance before loss is still proposed on return."""
        out = self.templates()[:1]  # anchor
        out.extend(self.snapshots)
        if self.recent is not None and all(self.recent is not t for t in out):
            out.append(self.recent)
        return out


def _ncc(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    """Structural similarity of two patches in [0, 1], size-normalised to _CANON.

    Both patches are squashed to the same canonical size first, so a big box and a
    small box of the same thing still score high (scale-invariant).
    """
    if a_gray.size == 0 or b_gray.size == 0:
        return 0.0
    a = cv2.resize(a_gray, _CANON)
    b = cv2.resize(b_gray, _CANON)
    return float(max(0.0, cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0]))


# Upright-NCC above this is already a good match -> skip the rotation sweep (keeps
# the common upright case at one NCC; only rotated/poor matches pay for the sweep).
_ROT_SKIP = 0.55


def _ncc_template(query_gray: np.ndarray, tmpl: Template) -> float:
    """NCC of a query patch against a template; rotation-tolerant if the template
    carries rotated variants (the anchor). The rotation sweep is lazy — only run
    when the upright match is weak."""
    if not tmpl.variants or query_gray.size == 0:
        return _ncc(tmpl.gray, query_gray)
    q = cv2.resize(query_gray, _CANON)
    upright = float(cv2.matchTemplate(tmpl.variants[0], q, cv2.TM_CCOEFF_NORMED)[0, 0])
    if upright >= _ROT_SKIP:
        return max(0.0, upright)
    return float(max(0.0, max(cv2.matchTemplate(v, q, cv2.TM_CCOEFF_NORMED)[0, 0]
                              for v in tmpl.variants)))


def _orb_inlier_ratio(query: Template, ref: Template, matcher) -> float:
    """Fraction of ORB matches that survive a RANSAC homography, in [0, 1].

    A geometry cue: high only when query and ref share distinctive features in a
    consistent spatial layout (0 when either lacks descriptors).
    """
    if query.descriptors is None or ref.descriptors is None:
        return 0.0
    if len(query.descriptors) < 4 or len(ref.descriptors) < 4:
        return 0.0
    matches = matcher.match(query.descriptors, ref.descriptors)
    if len(matches) < 4:
        return 0.0
    src = np.float32([query.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([ref.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return 0.0
    denom = min(len(query.keypoints), len(ref.keypoints)) or 1
    return float(min(1.0, int(mask.sum()) / denom))


class Verifier:
    """Fuses NCC + HSV + ORB into an appearance confidence, best over the memory."""

    def __init__(self, cfg: VerifierConfig, memory: AppearanceMemory) -> None:
        self.cfg = cfg
        self.memory = memory
        self.matcher = orb_matcher()
        self._orb_counter = 0     # frames since ORB last ran (it runs every orb_every)
        self._last_orb = 0.0      # cached ORB score reused between runs

    def appearance_confidence(self, frame, bbox: BBox, hud_mask=None,
                              force_orb: bool = False, templates=None) -> Tuple[float, dict]:
        """Fused identity confidence for ``bbox`` — best NCC/HSV/ORB over the memory.

        ORB (the costly cue) runs every ``orb_every`` calls unless ``force_orb``;
        between runs its last value is reused. Returns ``(confidence, per-cue dict)``.
        """
        templates = self.memory.templates() if templates is None else templates
        if not templates:
            return 0.0, {}
        query = self.memory.extract(frame, bbox, hud_mask)
        # Each cue scores the query against its best-matching stored template.
        s_ncc = max(_ncc_template(query.gray, t) for t in templates)
        s_hist = max(hist_similarity(query.hist, t.hist) for t in templates)
        # ORB is expensive, so run it periodically and reuse its last score in between.
        run_orb = force_orb or (self._orb_counter % self.cfg.orb_every == 0)
        self._orb_counter += 1
        if run_orb:
            s_orb = max(_orb_inlier_ratio(query, t, self.matcher) for t in templates)
            self._last_orb = s_orb
        else:
            s_orb = self._last_orb
        # Combine the three cues as a weighted average (weights need not sum to 1).
        w = self.cfg.w_ncc + self.cfg.w_hist + self.cfg.w_orb
        conf = (self.cfg.w_ncc * s_ncc + self.cfg.w_hist * s_hist + self.cfg.w_orb * s_orb) / w
        return float(conf), {"ncc": s_ncc, "hist": s_hist, "orb": s_orb}

    def fuse_with_tracker(self, tracker_score: float, appearance_conf: float) -> float:
        """Blend the tracker's native score with appearance confidence (``w_tracker``)."""
        wt = self.cfg.w_tracker
        return float(wt * tracker_score + (1.0 - wt) * appearance_conf)
