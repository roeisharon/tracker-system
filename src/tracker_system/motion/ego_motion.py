"""Global (camera) motion estimation — the ego-motion backbone.

This is the single highest-leverage signal for aerial/UAV footage. The scene is
approximately rigid and dominated by camera motion (pan + descent/zoom); a target
that is fixed in the world (e.g. the hut) moves in the *image* only because the
camera moves. Estimating that per-frame camera transform lets the rest of the
system:

- **predict** where a world-fixed target went even while it is lost/occluded, so
  the re-acquisition search region and motion prior land on the true target
  instead of a stale velocity extrapolation (this is what disambiguates the
  target from identical distractors in repetitive terrain), and
- recover the **scale** change (drone descent = uniform zoom), so appearance
  matching and the tracker box keep pace with a target that grows manyfold.

The transform between the previous and current frame is estimated by sparse
optical flow on background features (``goodFeaturesToTrack`` + pyramidal
Lucas-Kanade) followed by a RANSAC ``estimateAffinePartial2D`` fit — a 4-DOF
*similarity* (translation, rotation, uniform scale), which matches drone motion
(no shear). The target's own box is masked out of feature detection so the
target's motion cannot corrupt the background estimate; RANSAC rejects the small
minority of screen-fixed HUD features. Flow runs on a downscaled grayscale frame
for speed; the resulting transform is reported in **full-resolution** coordinates
so callers can apply it directly to full-res boxes and points.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config.settings import MotionConfig
from ..utils.geometry import BBox
from ..utils.image import resize_frame


class Transform:
    """A 2x3 similarity transform (previous frame -> current frame), full-res.

    ``confidence`` is the RANSAC inlier ratio in ``[0, 1]``; an identity transform
    carries ``0.0`` so callers can cheaply test ``confidence > 0`` to decide
    whether a reliable motion estimate is available this frame.
    """

    __slots__ = ("matrix", "confidence")

    def __init__(self, matrix: np.ndarray, confidence: float) -> None:
        self.matrix = matrix
        self.confidence = confidence

    @classmethod
    def identity(cls) -> "Transform":
        """A no-motion transform flagged low-confidence (the graceful fallback)."""
        return cls(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64), 0.0)

    @property
    def scale(self) -> float:
        """Uniform scale factor of the transform (``> 1`` = the scene grew/zoomed in)."""
        a, b = self.matrix[0, 0], self.matrix[1, 0]
        return float(math.hypot(a, b))

    def apply_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Map a full-resolution point through the transform."""
        x, y = point
        m = self.matrix
        return (
            float(m[0, 0] * x + m[0, 1] * y + m[0, 2]),
            float(m[1, 0] * x + m[1, 1] * y + m[1, 2]),
        )

    def apply_bbox(self, bbox: BBox) -> BBox:
        """Carry a box forward: translate its centre and scale its size."""
        cx, cy = self.apply_point(bbox.center)
        s = self.scale
        w, h = bbox.w * s, bbox.h * s
        return BBox(cx - w / 2.0, cy - h / 2.0, w, h)


class GlobalMotionEstimator:
    """Frame-to-frame camera-motion estimator via sparse optical flow + RANSAC."""

    def __init__(self, config: MotionConfig) -> None:
        self.config = config
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Forget the previous frame (next ``update`` returns identity)."""
        self._prev_gray = None

    def update(
        self, frame: np.ndarray, target_bbox: Optional[BBox] = None
    ) -> Transform:
        """Estimate the transform mapping the previous frame to ``frame``.

        ``target_bbox`` is the target's box in the *previous* frame's coordinates
        (masked out of feature detection). Returns an identity (low-confidence)
        transform on the first frame, when disabled, or when too few features/
        inliers are found — callers then fall back to their prior behaviour.
        """
        if not self.config.enabled:
            return Transform.identity()

        s = self.config.flow_scale
        small = resize_frame(frame, s)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return Transform.identity()

        transform = self._estimate(self._prev_gray, gray, target_bbox, s)
        self._prev_gray = gray
        return transform

    # -- internals ---------------------------------------------------------

    def _estimate(
        self,
        prev_gray: np.ndarray,
        gray: np.ndarray,
        target_bbox: Optional[BBox],
        s: float,
    ) -> Transform:
        cfg = self.config
        mask = self._feature_mask(prev_gray.shape, target_bbox, s)
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=cfg.max_features,
            qualityLevel=cfg.quality_level,
            minDistance=cfg.min_distance,
            mask=mask,
        )
        if prev_pts is None or len(prev_pts) < cfg.min_features:
            return Transform.identity()

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
        if next_pts is None or status is None:
            return Transform.identity()

        status = status.reshape(-1).astype(bool)
        good_prev = prev_pts.reshape(-1, 2)[status]
        good_next = next_pts.reshape(-1, 2)[status]
        if len(good_prev) < cfg.min_features:
            return Transform.identity()

        matrix, inliers = cv2.estimateAffinePartial2D(
            good_prev,
            good_next,
            method=cv2.RANSAC,
            ransacReprojThreshold=cfg.ransac_thresh,
        )
        if matrix is None or inliers is None:
            return Transform.identity()

        n_inliers = int(inliers.sum())
        ratio = n_inliers / len(good_prev)
        if n_inliers < cfg.min_inliers or ratio < cfg.min_inlier_ratio:
            return Transform.identity()

        # The fit is in downscaled coordinates; scale/rotation are invariant to the
        # uniform pre/post scaling but the translation must be divided by the flow
        # downscale factor to express the transform in full-resolution pixels.
        matrix = matrix.astype(np.float64).copy()
        matrix[0, 2] /= s
        matrix[1, 2] /= s
        return Transform(matrix, float(ratio))

    def _feature_mask(
        self, shape: Tuple[int, int], target_bbox: Optional[BBox], s: float
    ) -> Optional[np.ndarray]:
        """255 where features may be sampled, 0 over the (dilated) target box."""
        if target_bbox is None:
            return None
        h, w = shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        # Dilate the target box so its motion-contaminated border is excluded too.
        pad_x = self.config.mask_dilate_frac * target_bbox.w
        pad_y = self.config.mask_dilate_frac * target_bbox.h
        x1 = int(round((target_bbox.x - pad_x) * s))
        y1 = int(round((target_bbox.y - pad_y) * s))
        x2 = int(round((target_bbox.x2 + pad_x) * s))
        y2 = int(round((target_bbox.y2 + pad_y) * s))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0
        return mask
