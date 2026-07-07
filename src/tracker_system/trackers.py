"""Single-object tracker backends behind one interface.

The decisive problem on the drone clip is **scale**: the target grows ~15x as the
drone descends. CSRT freezes its box (undershoots) and a raw ViT balloons it to
the whole frame (overshoots). So the default ``hybrid`` backend decouples the two
signals — it takes the **centre** from a deep Siamese tracker (ViT, robust and
class-agnostic, with a native confidence score) and the **size** from an
optical-flow similarity transform (the true per-frame zoom), optionally damped by
the global ego-motion scale. ``vit`` / ``nano`` / ``csrt`` are single-backend
fallbacks.

Every backend implements: ``init(frame, bbox, usable_mask)`` /
``update(frame) -> (found, BBox, score)`` / ``reinit`` / ``name``, plus a no-op
``set_scale_hint(scale, confidence)`` the pipeline uses to feed the ego scale.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import REPO_ROOT, TrackerConfig
from .geometry import BBox, bbox_from_center_wh, clamp_bbox

# CSRT constructor may live under cv2 or cv2.legacy depending on the build.
_CSRT_PATHS = ("TrackerCSRT_create", "legacy.TrackerCSRT_create")


class TrackerNotAvailableError(RuntimeError):
    """Raised when a requested tracker cannot be created in this OpenCV build."""


def _resolve(path: str) -> Optional[Callable]:
    obj = cv2
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if callable(obj) else None


def _model_path(path: str) -> str:
    """Resolve a model path against CWD then the repo root."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return str(REPO_ROOT / path)


def _to_bbox(box) -> BBox:
    x, y, w, h = box
    return BBox(float(x), float(y), float(w), float(h))


# -- flow-based scale estimator ---------------------------------------------

_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


class FlowTracker:
    """Local feature-flow similarity tracker: translation + uniform scale.

    Tracks a cloud of point features **inside the box** with LK optical flow and
    fits a RANSAC similarity transform between frames. This is the scale backbone
    under the drone's extreme zoom (the box grows with the target) and also gives
    a translation the deep tracker can defer to when it is unconfident on a small,
    low-texture target. Features are replenished as they spread out and sampled
    only from the usable (HUD-masked) region.

    ``update`` returns ``(cx, cy, scale, score)`` for the *current* box centre, or
    ``None`` when no reliable estimate is available. ``score`` is the inlier count
    normalised to ``[0, 1]`` — a "the local region is still coherent" signal (not
    a target-identity signal; identity is the verifier's job).
    """

    def __init__(self, cfg: TrackerConfig, usable_mask: Optional[np.ndarray] = None) -> None:
        self.cfg = cfg
        self.usable = usable_mask
        self.prev_gray: Optional[np.ndarray] = None
        self.pts: Optional[np.ndarray] = None
        self.box: Optional[list] = None

    def init(self, frame, box: BBox, usable_mask=None) -> None:
        if usable_mask is not None:
            self.usable = usable_mask
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.box = [box.x, box.y, box.w, box.h]
        self.pts = self._detect(self.prev_gray, self.box)

    def set_box(self, box: BBox) -> None:
        """Re-anchor the feature-cloud box (to the fused centre/size each frame)."""
        self.box = [box.x, box.y, box.w, box.h]

    def update(self, frame) -> Optional[Tuple[float, float, float, float]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_f = self.cfg.flow_min_features
        if self.pts is None or len(self.pts) < min_f:
            self.pts = self._detect(self.prev_gray, self.box)
        if self.pts is None or len(self.pts) < min_f:
            self.prev_gray = gray
            return None

        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.pts, None, **_LK)
        result = None
        if nxt is not None and st is not None:
            st = st.reshape(-1).astype(bool)
            p0, p1 = self.pts.reshape(-1, 2)[st], nxt.reshape(-1, 2)[st]
            if len(p0) >= min_f:
                M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                                     ransacReprojThreshold=3)
                if M is not None:
                    s = float(np.clip(np.hypot(M[0, 0], M[0, 1]),
                                      self.cfg.flow_scale_min, self.cfg.flow_scale_max))
                    cx, cy = self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] / 2
                    ncx = float(M[0, 0] * cx + M[0, 1] * cy + M[0, 2])
                    ncy = float(M[1, 0] * cx + M[1, 1] * cy + M[1, 2])
                    inl = inl.reshape(-1).astype(bool)
                    n_in = int(inl.sum())
                    self.pts = p1[inl].reshape(-1, 1, 2)
                    result = (ncx, ncy, s, min(1.0, n_in / self.cfg.flow_score_norm))
        if self.pts is not None and len(self.pts) < self.cfg.flow_replenish_below:
            fresh = self._detect(gray, self.box)
            if fresh is not None and (self.pts is None or len(fresh) > len(self.pts)):
                self.pts = fresh
        self.prev_gray = gray
        return result

    def _detect(self, gray, box) -> Optional[np.ndarray]:
        if gray is None or box is None:
            return None
        x, y, w, h = (int(v) for v in box)
        x, y = max(0, x), max(0, y)
        m = np.zeros(gray.shape, np.uint8)
        m[y:y + h, x:x + w] = 255
        if self.usable is not None:
            m = cv2.bitwise_and(m, self.usable)
        return cv2.goodFeaturesToTrack(gray, maxCorners=self.cfg.flow_max_features,
                                       qualityLevel=0.01, minDistance=5, mask=m, blockSize=7)


# -- backends ---------------------------------------------------------------

class _DeepTracker:
    """Wraps cv2.TrackerVit / TrackerNano -> (found, BBox, native score)."""

    def __init__(self, cfg: TrackerConfig, kind: str) -> None:
        self.cfg = cfg
        self.kind = kind
        self._impl = None
        self.last_score = 0.0   # identity-ish score, for the template-update gate

    def _create(self):
        if self.kind == "nano":
            bb, hd = _model_path(self.cfg.nano_backbone), _model_path(self.cfg.nano_head)
            if not (os.path.exists(bb) and os.path.exists(hd)):
                raise TrackerNotAvailableError(
                    "NanoTrack models missing. Run: python download_models.py --nano")
            p = cv2.TrackerNano_Params()
            p.backbone, p.neckhead = bb, hd
            return cv2.TrackerNano_create(p)
        net = _model_path(self.cfg.vit_model)
        if not os.path.exists(net):
            raise TrackerNotAvailableError(
                f"ViT model missing at {net}. Run: python download_models.py")
        p = cv2.TrackerVit_Params()
        p.net = net
        return cv2.TrackerVit_create(p)

    def init(self, frame, bbox: BBox, usable_mask=None) -> None:
        self._impl = self._create()
        self._impl.init(frame, tuple(int(v) for v in bbox.as_int_xywh()))

    def update(self, frame) -> Tuple[bool, BBox, float]:
        found, box = self._impl.update(frame)
        try:
            score = float(self._impl.getTrackingScore())
        except Exception:
            score = 1.0 if found else 0.0
        self.last_score = score
        return bool(found), _to_bbox(box), score

    def reinit(self, frame, bbox: BBox, usable_mask=None) -> None:
        self.init(frame, bbox, usable_mask)

    def set_scale_hint(self, scale: float, confidence: float) -> None:
        pass

    @property
    def name(self) -> str:
        return self.kind.upper()


class _CsrtTracker:
    """Classical CSRT backend (its own box; no native score -> found=1.0/0.0)."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self._impl = None
        self.last_score = 1.0

    def init(self, frame, bbox: BBox, usable_mask=None) -> None:
        ctor = next((c for c in (_resolve(p) for p in _CSRT_PATHS) if c), None)
        if ctor is None:
            raise TrackerNotAvailableError(
                "CSRT unavailable. Install 'opencv-contrib-python' (not headless).")
        self._impl = ctor()
        self._impl.init(frame, bbox.as_int_xywh())

    def update(self, frame) -> Tuple[bool, BBox, float]:
        ok, box = self._impl.update(frame)
        return (True, _to_bbox(box), 1.0) if ok else (False, BBox(0, 0, 1, 1), 0.0)

    def reinit(self, frame, bbox: BBox, usable_mask=None) -> None:
        self.init(frame, bbox, usable_mask)

    def set_scale_hint(self, scale: float, confidence: float) -> None:
        pass

    @property
    def name(self) -> str:
        return "CSRT"


class HybridTracker:
    """Deep-tracker centre (when confident) + local-flow translation/scale.

    The drone target is small and low-texture early (ViT can't lock it) and huge
    late. So the box motion+size come from the local flow similarity transform,
    and the ViT centre overrides it only when the ViT score clears
    ``vit_trust_score`` (distinct targets, or the target once it has grown). The
    returned score is a presence proxy (max of the two) that keeps the fused
    confidence meaningful; identity is enforced separately by the verifier.
    """

    def __init__(self, cfg: TrackerConfig, deep_kind: str = "vit") -> None:
        self.cfg = cfg
        self.deep = _DeepTracker(cfg, deep_kind)
        self.flow = FlowTracker(cfg)
        self.cx = self.cy = 0.0
        self.w = self.h = 0.0
        self._ego_scale = 1.0
        self._ego_conf = 0.0
        self.last_score = 0.0   # ViT identity score (NOT the flow presence)

    def init(self, frame, bbox: BBox, usable_mask=None) -> None:
        self.deep.init(frame, bbox)
        self.flow.init(frame, bbox, usable_mask)
        self.cx, self.cy = bbox.center
        self.w, self.h = float(bbox.w), float(bbox.h)

    def reinit(self, frame, bbox: BBox, usable_mask=None) -> None:
        self.init(frame, bbox, usable_mask)

    def set_scale_hint(self, scale: float, confidence: float) -> None:
        self._ego_scale, self._ego_conf = scale, confidence

    def update(self, frame) -> Tuple[bool, BBox, float]:
        H, W = frame.shape[:2]
        flow = self.flow.update(frame)              # (cx, cy, scale, score) or None
        found_v, dbox, vscore = self.deep.update(frame)
        confident = found_v and vscore >= self.cfg.vit_trust_score

        # Centre: trust the ViT when it is confident, else follow the flow cloud —
        # but while unconfident, cap the per-frame shift so a bad flow fit on
        # low-texture desert can't run the box to the edge before the ViT locks.
        if confident:
            self.cx, self.cy = dbox.center
        elif flow is not None:
            ncx, ncy = flow[0], flow[1]
            cap = self.cfg.unconfident_shift_frac * float(np.hypot(W, H))
            dist = float(np.hypot(ncx - self.cx, ncy - self.cy))
            if dist > cap > 0:
                f = cap / dist
                ncx, ncy = self.cx + (ncx - self.cx) * f, self.cy + (ncy - self.cy) * f
            self.cx, self.cy = ncx, ncy

        # Scale: local flow, damped by the global ego scale (both measure the zoom).
        s = flow[2] if flow is not None else 1.0
        if self.cfg.scale_cross_check and self._ego_conf > 0:
            s = float(np.sqrt(max(s, 1e-6) * max(self._ego_scale, 1e-6)))
        cap_px = float(min(W, H))
        self.w = float(np.clip(self.w * s, self.cfg.min_box, cap_px))
        self.h = float(np.clip(self.h * s, self.cfg.min_box, cap_px))

        box = clamp_bbox(bbox_from_center_wh(self.cx, self.cy, self.w, self.h), W, H)
        self.flow.set_box(bbox_from_center_wh(self.cx, self.cy, self.w, self.h))

        # last_score is the ViT identity (gates template learning); the returned
        # presence score also credits flow (carries the target through the phase
        # where it is too small/low-texture for the ViT to lock).
        self.last_score = vscore
        return True, box, max(vscore, flow[3] if flow is not None else 0.0)

    @property
    def name(self) -> str:
        return f"HYBRID({self.deep.name}+FLOW)"


def create_backend(cfg: TrackerConfig):
    """Build the configured tracker backend."""
    b = cfg.backend
    if b == "hybrid":
        return HybridTracker(cfg, "vit")
    if b in ("vit", "nano"):
        return _DeepTracker(cfg, b)
    if b == "csrt":
        return _CsrtTracker(cfg)
    raise TrackerNotAvailableError(f"Unknown backend {b!r}")


def probe_backends(cfg: Optional[TrackerConfig] = None) -> Dict[str, bool]:
    """Report which backends can actually be constructed in this build."""
    cfg = cfg or TrackerConfig()
    result: Dict[str, bool] = {}
    # CSRT
    try:
        ctor = next((c for c in (_resolve(p) for p in _CSRT_PATHS) if c), None)
        ctor and ctor()
        result["csrt"] = ctor is not None
    except Exception:
        result["csrt"] = False
    # Deep backends: model file present AND the cv2 factory exists.
    result["vit"] = hasattr(cv2, "TrackerVit_create") and os.path.exists(_model_path(cfg.vit_model))
    result["nano"] = (hasattr(cv2, "TrackerNano_create")
                      and os.path.exists(_model_path(cfg.nano_backbone))
                      and os.path.exists(_model_path(cfg.nano_head)))
    result["hybrid"] = result["vit"]
    return result
