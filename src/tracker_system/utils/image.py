"""Image helpers for the processing/display split.

The pipeline tracks on a *working frame* that may be downscaled for speed, then
renders results onto the full-resolution frame. ``resize_frame`` centralises
that downscale so the interpolation choice is consistent everywhere.
"""

from __future__ import annotations
import cv2
import numpy as np


def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    """Return ``frame`` scaled by ``scale`` (``1.0`` returns the frame as-is).

    ``INTER_AREA`` is used for downscaling (best quality shrinking) and
    ``INTER_LINEAR`` otherwise. At ``scale == 1.0`` the original array is
    returned unchanged to avoid a needless copy.
    """
    if scale == 1.0:
        return frame
    frame_h, frame_w = frame.shape[:2]
    new_w = max(1, int(round(frame_w * scale)))
    new_h = max(1, int(round(frame_h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interp)
