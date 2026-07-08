"""Bounding-box geometry + frame-resize helpers.

Boxes are ``(x, y, w, h)`` in pixels, ``x`` = column and ``y`` = row of the
top-left corner (OpenCV convention). All functions are pure so they are trivial
to unit-test and reuse across selection, tracking, appearance, and overlay.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in ``(x, y, w, h)`` pixels (top-left corner + size)."""

    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return self.w * self.h

    def as_int_xywh(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x)), int(round(self.y)),
                int(round(self.w)), int(round(self.h)))

    def scaled(self, factor: float) -> "BBox":
        # Move a box between full-res and the downscaled working frame.
        return BBox(self.x * factor, self.y * factor, self.w * factor, self.h * factor)


def bbox_from_center(cx: float, cy: float, size: float) -> BBox:
    """Square box of side ``size`` centred on ``(cx, cy)``."""
    side = float(size)
    return BBox(cx - side / 2.0, cy - side / 2.0, side, side)


def bbox_from_center_wh(cx: float, cy: float, w: float, h: float) -> BBox:
    """Box of size ``(w, h)`` centred on ``(cx, cy)``."""
    return BBox(cx - w / 2.0, cy - h / 2.0, w, h)


def clamp_point(x: float, y: float, frame_w: int, frame_h: int) -> Tuple[int, int]:
    """Round ``(x, y)`` and clamp it to a valid pixel inside the frame."""
    cx = min(max(int(round(x)), 0), max(frame_w - 1, 0))
    cy = min(max(int(round(y)), 0), max(frame_h - 1, 0))
    return (cx, cy)


def clamp_bbox(bbox: BBox, frame_w: int, frame_h: int) -> BBox:
    """Clamp a box fully inside the frame; keep w/h >= 1 so it stays usable."""
    max_x = max(frame_w - 1.0, 0.0)
    max_y = max(frame_h - 1.0, 0.0)
    x = min(max(bbox.x, 0.0), max_x)
    y = min(max(bbox.y, 0.0), max_y)
    w = max(min(bbox.w, frame_w - x), 1.0)
    h = max(min(bbox.h, frame_h - y), 1.0)
    return BBox(x, y, w, h)


def frame_overlap_ratio(bbox: BBox, frame_w: int, frame_h: int) -> float:
    """Fraction of ``bbox`` area inside the frame (1.0 = fully in, 0.0 = out)."""
    ix1, iy1 = max(bbox.x, 0.0), max(bbox.y, 0.0)
    ix2, iy2 = min(bbox.x2, float(frame_w)), min(bbox.y2, float(frame_h))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / bbox.area if bbox.area > 0 else 0.0


def extract_patch(image: np.ndarray, bbox: BBox) -> np.ndarray:
    """Copy of the image region under ``bbox`` (clamped first, so always safe)."""
    frame_h, frame_w = image.shape[:2]
    x, y, w, h = clamp_bbox(bbox, frame_w, frame_h).as_int_xywh()
    return image[y:min(y + h, frame_h), x:min(x + w, frame_w)].copy()


def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    """Scale a frame (1.0 returns it unchanged); INTER_AREA when shrinking."""
    if scale == 1.0:
        return frame
    frame_h, frame_w = frame.shape[:2]
    new_w = max(1, int(round(frame_w * scale)))
    new_h = max(1, int(round(frame_h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interp)
