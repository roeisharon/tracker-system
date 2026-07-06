"""Appearance-confirmed re-acquisition while LOST.

Coarse multi-scale template match over a downscaled full frame proposes the best
location; it is then confirmed at full resolution by the appearance
:class:`~tracker_system.appearance.Verifier` (including ORB/RANSAC) and accepted
only above ``t_reacq`` (stricter than the tracking gate). This is deliberately
appearance-first — unlike the old motion-prior-dominated matcher it will not snap
onto whatever blob sits nearest the predicted point.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .appearance import AppearanceMemory, Verifier
from .config import ReacquireConfig
from .geometry import BBox, clamp_bbox


class Reacquirer:
    def __init__(self, cfg: ReacquireConfig, memory: AppearanceMemory, verifier: Verifier) -> None:
        self.cfg = cfg
        self.memory = memory
        self.verifier = verifier
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
        templates = self.memory.templates()
        if not templates:
            return None

        H, W = frame.shape[:2]
        ds = self.cfg.reacq_downscale
        small_gray = cv2.cvtColor(cv2.resize(frame, (int(W * ds), int(H * ds))), cv2.COLOR_BGR2GRAY)

        best = None  # (match_value, full_res_box)
        # Search with every remembered template (recent first for the current
        # scale, anchor as the drift-free fallback).
        for tmpl in reversed(templates):
            tw, th = tmpl.size
            for s in self.cfg.reacq_scales:
                sw, sh = max(8, int(tw * s * ds)), max(8, int(th * s * ds))
                if sw >= small_gray.shape[1] or sh >= small_gray.shape[0]:
                    continue
                res = cv2.matchTemplate(small_gray, cv2.resize(tmpl.gray, (sw, sh)),
                                        cv2.TM_CCOEFF_NORMED)
                _, maxv, _, maxloc = cv2.minMaxLoc(res)
                box = clamp_bbox(BBox(maxloc[0] / ds, maxloc[1] / ds, sw / ds, sh / ds), W, H)
                self.last_candidates.append((box, float(maxv)))
                if best is None or maxv > best[0]:
                    best = (maxv, box)

        if best is None:
            return None
        cand = best[1]
        if hud_mask is not None:  # reject a centre sitting on the HUD overlay
            cx, cy = (int(v) for v in cand.center)
            if 0 <= cy < H and 0 <= cx < W and hud_mask[cy, cx] > 0:
                return None
        conf, _ = self.verifier.appearance_confidence(frame, cand, hud_mask, force_orb=True)
        if conf >= self.cfg.t_reacq:
            self.last_accepted = cand
            return cand, conf
        return None
