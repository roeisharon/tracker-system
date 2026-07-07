"""Appearance memory + independent multi-cue verifier.

A deep tracker can drift onto a distractor while still reporting a high score, so
every box is cross-checked against an appearance memory with three cues that fail
in different ways:

  * NCC  - grayscale normalised cross-correlation (structure/texture, scale-normalised)
  * HIST - HSV hue-saturation histogram          (colour)
  * ORB  - feature matches + RANSAC inliers       (distinctive geometry; the cue
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

from .config import VerifierConfig
from .geometry import BBox, clamp_bbox

_CANON = (64, 64)          # canonical size for scale-invariant NCC
_HIST_BINS = [50, 60]
_HIST_RANGES = [0, 180, 0, 256]
_HIST_CHANNELS = [0, 1]


@dataclass
class Template:
    gray: np.ndarray
    hist: np.ndarray
    keypoints: tuple
    descriptors: Optional[np.ndarray]
    size: Tuple[int, int]


def orb_matcher() -> cv2.BFMatcher:
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
        self.cfg = cfg
        self._orb = cv2.ORB_create(nfeatures=cfg.orb_nfeatures, scaleFactor=1.2, nlevels=8)
        self.anchor: Optional[Template] = None
        self.recent: Optional[Template] = None

    def extract(self, frame: np.ndarray, bbox: BBox, hud_mask=None) -> Template:
        fh, fw = frame.shape[:2]
        x, y, w, h = clamp_bbox(bbox, fw, fh).as_int_xywh()
        patch = frame[y:y + h, x:x + w]
        sm = _sub_mask(hud_mask, x, y, w, h)
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
        kp, desc = self._orb.detectAndCompute(gray, sm)
        return Template(gray, hist, tuple(kp), desc, (w, h))

    def initialise(self, frame, bbox: BBox, hud_mask=None) -> None:
        self.anchor = self.extract(frame, bbox, hud_mask)
        self.recent = self.anchor

    def update(self, frame, bbox: BBox, hud_mask, confidence: float, tracker_score: float = 0.0) -> None:
        """Refresh the recent template when the frame is confidently on-target.

        Gating on EITHER the fused confidence OR the raw tracker score breaks the
        deadlock (confidence needs a fresh template to stay high, but was the gate
        for refreshing it).
        """
        if confidence < self.cfg.ema_update_conf and tracker_score < self.cfg.tmpl_update_score:
            return
        new = self.extract(frame, bbox, hud_mask)
        if self.recent is not None and self.recent.gray.shape == new.gray.shape:
            blended = cv2.addWeighted(self.recent.gray, 1 - self.cfg.ema_alpha,
                                      new.gray, self.cfg.ema_alpha, 0)
            new = Template(blended, new.hist, new.keypoints, new.descriptors, new.size)
        self.recent = new

    def templates(self) -> List[Template]:
        out = []
        if self.anchor is not None:
            out.append(self.anchor)
        if self.recent is not None and self.recent is not self.anchor:
            out.append(self.recent)
        return out


def _ncc(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    if a_gray.size == 0 or b_gray.size == 0:
        return 0.0
    a = cv2.resize(a_gray, _CANON)
    b = cv2.resize(b_gray, _CANON)
    return float(max(0.0, cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0]))


def _orb_inlier_ratio(query: Template, ref: Template, matcher) -> float:
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
        self._orb_counter = 0
        self._last_orb = 0.0

    def appearance_confidence(self, frame, bbox: BBox, hud_mask=None,
                              force_orb: bool = False) -> Tuple[float, dict]:
        templates = self.memory.templates()
        if not templates:
            return 0.0, {}
        query = self.memory.extract(frame, bbox, hud_mask)
        s_ncc = max(_ncc(t.gray, query.gray) for t in templates)
        s_hist = max(hist_similarity(query.hist, t.hist) for t in templates)
        run_orb = force_orb or (self._orb_counter % self.cfg.orb_every == 0)
        self._orb_counter += 1
        if run_orb:
            s_orb = max(_orb_inlier_ratio(query, t, self.matcher) for t in templates)
            self._last_orb = s_orb
        else:
            s_orb = self._last_orb
        w = self.cfg.w_ncc + self.cfg.w_hist + self.cfg.w_orb
        conf = (self.cfg.w_ncc * s_ncc + self.cfg.w_hist * s_hist + self.cfg.w_orb * s_orb) / w
        return float(conf), {"ncc": s_ncc, "hist": s_hist, "orb": s_orb}

    def fuse_with_tracker(self, tracker_score: float, appearance_conf: float) -> float:
        wt = self.cfg.w_tracker
        return float(wt * tracker_score + (1.0 - wt) * appearance_conf)
