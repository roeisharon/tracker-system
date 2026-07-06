"""Mouse-click target selector using OpenCV HighGUI.

Shows the first frame in a window and waits for a left click, mapping the click
to the same :class:`SelectionResult` a manual ``[i, j]`` would produce. The
click-to-result mapping is factored into :meth:`_build_result` — pure and
window-free — so it can be unit-tested without a display, while :meth:`select`
handles the interactive window loop.

Requires a HighGUI-capable OpenCV build (``opencv-contrib-python``); a headless
build cannot open the window.
"""

from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np
from ..utils.geometry import BBox, bbox_from_center, clamp_bbox
from .target_selector import SelectionError, SelectionResult, TargetSelector

_INSTRUCTIONS = "Click the target  |  ESC to cancel"


class CvClickSelector(TargetSelector):
    """Select a target by clicking it on the first frame."""

    def __init__(self, bbox_size: int, window_name: str = "Select target") -> None:
        self.bbox_size = int(bbox_size)
        self.window_name = window_name
        self._frame_shape: Optional[Tuple[int, int]] = None
        self._result: Optional[SelectionResult] = None

    def select(self, frame: np.ndarray) -> SelectionResult:
        self._frame_shape = frame.shape[:2]
        self._result = None

        display = frame.copy()
        cv2.putText(
            display,
            _INSTRUCTIONS,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        try:
            cv2.imshow(self.window_name, display)
            while self._result is None:
                key = cv2.waitKey(20) & 0xFF
                if key == 27:  # ESC
                    break
                # Check if the window was closed by the user 
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)  # let the window actually close on some backends

        if self._result is None:
            raise SelectionError("Target selection cancelled")
        return self._result

    # Handle mouse events and build the selection result from a click.
    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._result = self._build_result(x, y)

    # Build a SelectionResult from the given click coordinates, clamping to the frame.
    def _build_result(self, x: int, y: int) -> SelectionResult:
        if self._frame_shape is None:
            raise SelectionError("Selector has no frame; call select() first")
        frame_h, frame_w = self._frame_shape
        cx = min(max(int(x), 0), frame_w - 1)
        cy = min(max(int(y), 0), frame_h - 1)
        bbox = clamp_bbox(bbox_from_center(cx, cy, self.bbox_size), frame_w, frame_h)
        return SelectionResult(bbox=bbox, seed_point=(cx, cy), source="mouse")
