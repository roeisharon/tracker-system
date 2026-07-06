"""Bounding-box geometry helpers.

A single small :class:`BBox` value type plus pure functions for the operations
the pipeline repeats constantly: build a box around a point, keep a box inside
the frame, scale a box between the full-resolution and the downscaled working
frame, and cut a patch out of an image. Keeping these here (pure, no OpenCV
state) makes them trivial to unit-test and reuse across selection, tracking,
target profiling, and visualization.

Coordinate convention: boxes are ``(x, y, w, h)`` in pixels where ``x`` is the
column and ``y`` is the row of the top-left corner, matching OpenCV.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class BBox:
    """An axis-aligned bounding box in ``(x, y, w, h)`` pixel coordinates."""
    # (x,y) = top-left corner, w = width, h = height in pixels
    x: float # column
    y: float # row
    w: float
    h: float

    @property
    def center(self) -> Tuple[float, float]:
        """Return the ``(cx, cy)`` centre in pixels."""
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def x2(self) -> float:
        """Right edge ``x + w``."""
        return self.x + self.w

    @property
    def y2(self) -> float:
        """Bottom edge ``y + h``."""
        return self.y + self.h

    @property
    def area(self) -> float:
        return self.w * self.h

    def as_int_xywh(self) -> Tuple[int, int, int, int]:
        """Return integer ``(x, y, w, h)`` suitable for OpenCV calls."""
        return (
            int(round(self.x)),
            int(round(self.y)),
            int(round(self.w)),
            int(round(self.h)),
        )

    def scaled(self, factor: float) -> "BBox":
        """Return a copy scaled about the origin by ``factor``.

        Used to move a box between the full-resolution frame and the downscaled
        working frame (``scale`` one way, ``1/scale`` back).
        """
        return BBox(self.x * factor, self.y * factor, self.w * factor, self.h * factor)


def bbox_from_center(cx: float, cy: float, size: float) -> BBox:
    """Build a square box of side ``size`` centred on ``(cx, cy)``."""
    side = float(size)
    return BBox(cx - side / 2.0, cy - side / 2.0, side, side)


def scale_bbox(bbox: BBox, factor: float) -> BBox:
    """Functional alias for :meth:`BBox.scaled`."""
    return bbox.scaled(factor)


def clamp_point(x: float, y: float, frame_w: int, frame_h: int) -> Tuple[int, int]:
    """Clamp a pixel to the valid ``[0, w-1] x [0, h-1]`` range."""
    cx = min(max(int(round(x)), 0), max(frame_w - 1, 0))
    cy = min(max(int(round(y)), 0), max(frame_h - 1, 0))
    return (cx, cy)


def clamp_bbox(bbox: BBox, frame_w: int, frame_h: int) -> BBox:
    """Clamp a box so it lies fully inside a ``frame_w x frame_h`` frame.

    The top-left corner is pushed inside the frame and the width/height are
    trimmed so the box never spills over the edges. Width and height are kept at
    least ``1`` so the result is always a usable region.
    """
    max_x = max(frame_w - 1.0, 0.0)
    max_y = max(frame_h - 1.0, 0.0)
    x = min(max(bbox.x, 0.0), max_x)
    y = min(max(bbox.y, 0.0), max_y)
    w = min(bbox.w, frame_w - x)
    h = min(bbox.h, frame_h - y)
    w = max(w, 1.0)
    h = max(h, 1.0)
    return BBox(x, y, w, h)


def frame_overlap_ratio(bbox: BBox, frame_w: int, frame_h: int) -> float:
    """Return the fraction of ``bbox``'s area that lies inside the frame.

    ``1.0`` means fully inside, ``0.0`` fully outside. Used by loss detection to
    decide when the target has left the frame.
    """
    ix1 = max(bbox.x, 0.0)
    iy1 = max(bbox.y, 0.0)
    ix2 = min(bbox.x2, float(frame_w))
    iy2 = min(bbox.y2, float(frame_h))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / bbox.area if bbox.area > 0 else 0.0


def extract_patch(image: np.ndarray, bbox: BBox) -> np.ndarray:
    """Return a copy of the image region covered by ``bbox``.

    The box is clamped to the image first, so this is always safe even if the
    box partially leaves the frame. The returned patch is a copy, decoupled from
    the source frame's lifetime (important under streaming, where frames are
    discarded each iteration).
    """
    frame_h, frame_w = image.shape[:2]
    clamped = clamp_bbox(bbox, frame_w, frame_h)
    x, y, w, h = clamped.as_int_xywh()
    x2 = min(x + w, frame_w)
    y2 = min(y + h, frame_h)
    return image[y:y2, x:x2].copy()
