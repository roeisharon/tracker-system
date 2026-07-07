"""Global (camera) ego-motion estimation.

On aerial footage the scene is ~rigid and dominated by camera pan + descent, so a
world-fixed target moves in the image only because the camera moves. Estimating
that per-frame similarity transform (translation + rotation + uniform scale) lets
the system recover the zoom (to cross-check the box scale) and predict where a
lost target went (to seed the re-acquisition search).

Sparse optical flow on background features (``goodFeaturesToTrack`` + pyramidal
Lucas-Kanade) is fit with RANSAC ``estimateAffinePartial2D``. The target box is
masked out so its motion can't corrupt the background estimate; RANSAC rejects
screen-fixed HUD features. Flow runs downscaled; the transform is full-res.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import MotionConfig
from .geometry import BBox, resize_frame


class Transform:
    """A 2x3 similarity transform (prev -> current frame), full-resolution.

    ``confidence`` is the RANSAC inlier ratio; an identity carries 0.0 so callers
    can test ``confidence > 0`` for "a reliable estimate is available".
    """

    __slots__ = ("matrix", "confidence")

    def __init__(self, matrix: np.ndarray, confidence: float) -> None:
        self.matrix = matrix
        self.confidence = confidence

    @classmethod
    def identity(cls) -> "Transform":
        return cls(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64), 0.0)

    @property
    def scale(self) -> float:
        return float(math.hypot(self.matrix[0, 0], self.matrix[1, 0]))

    def apply_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        x, y = point
        m = self.matrix
        return (float(m[0, 0] * x + m[0, 1] * y + m[0, 2]),
                float(m[1, 0] * x + m[1, 1] * y + m[1, 2]))

    def apply_bbox(self, bbox: BBox) -> BBox:
        cx, cy = self.apply_point(bbox.center)
        s = self.scale
        w, h = bbox.w * s, bbox.h * s
        return BBox(cx - w / 2.0, cy - h / 2.0, w, h)


class GlobalMotionEstimator:
    """Frame-to-frame camera-motion estimator (sparse flow + RANSAC)."""

    def __init__(self, config: MotionConfig) -> None:
        self.config = config
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_gray = None

    def update(self, frame: np.ndarray, target_bbox: Optional[BBox] = None) -> Transform:
        """Estimate prev->current transform; identity (low-conf) on failure/first frame."""
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

    def _estimate(self, prev_gray, gray, target_bbox, s) -> Transform:
        cfg = self.config
        mask = self._feature_mask(prev_gray.shape, target_bbox, s)
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=cfg.max_features, qualityLevel=cfg.quality_level,
            minDistance=cfg.min_distance, mask=mask,
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
            good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_thresh,
        )
        if matrix is None or inliers is None:
            return Transform.identity()
        n_inliers = int(inliers.sum())
        ratio = n_inliers / len(good_prev)
        if n_inliers < cfg.min_inliers or ratio < cfg.min_inlier_ratio:
            return Transform.identity()
        # Rotation/scale are scale-invariant; only the translation needs unscaling.
        matrix = matrix.astype(np.float64).copy()
        matrix[0, 2] /= s
        matrix[1, 2] /= s
        return Transform(matrix, float(ratio))

    def _feature_mask(self, shape, target_bbox, s) -> Optional[np.ndarray]:
        if target_bbox is None:
            return None
        h, w = shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        pad_x = self.config.mask_dilate_frac * target_bbox.w
        pad_y = self.config.mask_dilate_frac * target_bbox.h
        x1 = max(0, int(round((target_bbox.x - pad_x) * s)))
        y1 = max(0, int(round((target_bbox.y - pad_y) * s)))
        x2 = min(w, int(round((target_bbox.x2 + pad_x) * s)))
        y2 = min(h, int(round((target_bbox.y2 + pad_y) * s)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0
        return mask
