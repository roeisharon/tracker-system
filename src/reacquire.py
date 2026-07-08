"""Appearance-confirmed re-acquisition while LOST.

Coarse multi-scale template match over a downscaled full frame proposes the best
location; it is then confirmed at full resolution by the appearance
:class:`~appearance.Verifier` (including ORB/RANSAC) and accepted
only above ``t_reacq`` (stricter than the tracking gate). This is deliberately
appearance-first — unlike the old motion-prior-dominated matcher it will not snap
onto whatever blob sits nearest the predicted point.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from appearance import AppearanceMemory, Verifier, _rotate
from config import ReacquireConfig
from geometry import BBox, clamp_bbox


def _is_ambiguous(res: np.ndarray, loc, tw: int, th: int, ratio: float) -> bool:
    """True when a second, non-overlapping peak rivals the best in ``res``.

    In repetitive scenes (identical bushes, roof tiles) the winning peak carries
    no identity — suppress a window around it and see if another peak reaches
    ``ratio`` of the best.
    """
    best = float(res[loc[1], loc[0]])
    if best <= 0:
        return True
    work = res.copy()
    x0, y0 = max(0, loc[0] - tw), max(0, loc[1] - th)
    work[y0:loc[1] + th + 1, x0:loc[0] + tw + 1] = -1.0
    return work.size > 0 and float(work.max()) >= ratio * best


class Reacquirer:
    """Coarse multi-scale template search + full-identity confirm, run while LOST."""

    def __init__(self, cfg: ReacquireConfig, memory: AppearanceMemory, verifier: Verifier) -> None:
        self.cfg = cfg
        self.memory = memory
        self.verifier = verifier
        self._calls = 0
        # Diagnostics for the debug overlay.
        self.last_candidates: List[Tuple[BBox, float]] = []
        self.last_accepted: Optional[BBox] = None
        self.last_predicted: Optional[Tuple[int, int]] = None

    def search(self, frame, hud_mask=None,
               predicted_center: Optional[Tuple[float, float]] = None) -> Optional[Tuple[BBox, float]]:
        """Return ``(box, confidence)`` if the target is confidently re-found."""
        self.last_candidates = []
        self.last_accepted = None
        self.last_predicted = (int(predicted_center[0]), int(predicted_center[1])) if predicted_center else None
        self._calls += 1
        # The gallery (anchor + scale snapshots + recent) so a target that changed
        # appearance before loss is still proposed on return.
        templates = self.memory.reacq_templates()
        if not templates:
            return None

        H, W = frame.shape[:2]
        ds = self.cfg.reacq_downscale
        # Correlate on a downscaled gray frame — the coarse stage only needs a location.
        small_gray = cv2.cvtColor(cv2.resize(frame, (int(W * ds), int(H * ds))), cv2.COLOR_BGR2GRAY)
        # Throttled rotation sweep: try the anchor rotated so a rotated returning
        # target is localized (the upright search would miss it).
        sweep = self.cfg.rot_step > 0 and (self._calls % self.cfg.rot_every == 0)

        # Slide every (template, scale, angle) over the frame; keep the strongest peak.
        best = None  # (match_value, box, response_map, loc, sw, sh) of the coarse-best peak
        for i, tmpl in enumerate(templates):
            tw, th = tmpl.size
            angles = range(0, 360, self.cfg.rot_step) if (sweep and i == 0) else (0,)
            for s in self.cfg.reacq_scales:
                sw, sh = max(8, int(tw * s * ds)), max(8, int(th * s * ds))
                if sw >= small_gray.shape[1] or sh >= small_gray.shape[0]:
                    continue  # template bigger than the frame at this scale — can't match
                base = cv2.resize(tmpl.gray, (sw, sh))
                for ang in angles:
                    g = base if ang == 0 else _rotate(base, ang)
                    res = cv2.matchTemplate(small_gray, g, cv2.TM_CCOEFF_NORMED)
                    _, maxv, _, maxloc = cv2.minMaxLoc(res)
                    # Map the peak back to full-res coordinates.
                    box = clamp_bbox(BBox(maxloc[0] / ds, maxloc[1] / ds, sw / ds, sh / ds), W, H)
                    self.last_candidates.append((box, float(maxv)))
                    if best is None or maxv > best[0]:
                        best = (float(maxv), box, res, maxloc, sw, sh)

        if best is None:
            return None

        # Confirm the single best coarse peak with the full identity verifier.
        _, box, res, loc, sw, sh = best
        if hud_mask is not None:  # reject a centre sitting on the HUD overlay
            cx, cy = (int(v) for v in box.center)
            if 0 <= cy < H and 0 <= cx < W and hud_mask[cy, cx] > 0:
                return None
        # Ambiguity: a rival peak means the location carries little identity, so
        # demand a higher confidence before re-locking.
        ambiguous = _is_ambiguous(res, loc, sw, sh, self.cfg.ambiguity_ratio)
        bar = self.cfg.t_reacq_ambiguous if ambiguous else self.cfg.t_reacq
        conf, _ = self.verifier.appearance_confidence(frame, box, hud_mask,
                                                      force_orb=True, templates=templates)
        if conf >= bar:
            self.last_accepted = box
            return box, conf
        return None
