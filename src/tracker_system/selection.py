"""Target selection + burned-in overlay (HUD) handling.

Two selectors (manual ``[i, j]`` and mouse click) return the same
:class:`SelectionResult`, so the pipeline is selection-agnostic. The overlay
helpers detect a screen-fixed HUD (crosshair/telemetry that stays constant while
the world moves), inpaint it out of the init patch, and expose the mask so the
tracking path can exclude those pixels from appearance/flow matching.

Manual input is ``[i, j]`` = (row, col); everything else is ``(x, y)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import SelectionConfig
from .geometry import BBox, bbox_from_center, clamp_bbox

_INPAINT_RADIUS = 3
_INSTRUCTIONS = "Click the target  |  ESC to cancel"


class SelectionError(RuntimeError):
    """Raised when a target cannot be selected (out of frame, cancelled, ...)."""


@dataclass(frozen=True)
class SelectionResult:
    bbox: BBox
    seed_point: Tuple[int, int]
    source: str  # "manual" or "mouse"


class TargetSelector(ABC):
    @abstractmethod
    def select(self, frame: np.ndarray) -> SelectionResult:
        """Return the selected target for ``frame`` or raise :class:`SelectionError`."""


class ManualPixelSelector(TargetSelector):
    """Select from manual ``[i, j]`` (row, column) coordinates."""

    def __init__(self, row: int, col: int, bbox_size: int) -> None:
        self.row, self.col, self.bbox_size = int(row), int(col), int(bbox_size)

    def select(self, frame: np.ndarray) -> SelectionResult:
        frame_h, frame_w = frame.shape[:2]
        x, y = self.col, self.row
        if not (0 <= x < frame_w and 0 <= y < frame_h):
            raise SelectionError(
                f"Pixel [i={self.row}, j={self.col}] is outside the {frame_w}x{frame_h} frame"
            )
        bbox = clamp_bbox(bbox_from_center(x, y, self.bbox_size), frame_w, frame_h)
        return SelectionResult(bbox, (x, y), "manual")


class CvClickSelector(TargetSelector):
    """Select by clicking the target on the first frame (HighGUI window)."""

    def __init__(self, bbox_size: int, window_name: str = "Select target") -> None:
        self.bbox_size = int(bbox_size)
        self.window_name = window_name
        self._frame_shape: Optional[Tuple[int, int]] = None
        self._result: Optional[SelectionResult] = None

    def select(self, frame: np.ndarray) -> SelectionResult:
        self._frame_shape = frame.shape[:2]
        self._result = None
        display = frame.copy()
        cv2.putText(display, _INSTRUCTIONS, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        try:
            cv2.imshow(self.window_name, display)
            while self._result is None:
                if (cv2.waitKey(20) & 0xFF) == 27:  # ESC
                    break
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        if self._result is None:
            raise SelectionError("Target selection cancelled")
        return self._result

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._result = self._build_result(x, y)

    def _build_result(self, x: int, y: int) -> SelectionResult:
        if self._frame_shape is None:
            raise SelectionError("Selector has no frame; call select() first")
        frame_h, frame_w = self._frame_shape
        cx = min(max(int(x), 0), frame_w - 1)
        cy = min(max(int(y), 0), frame_h - 1)
        bbox = clamp_bbox(bbox_from_center(cx, cy, self.bbox_size), frame_w, frame_h)
        return SelectionResult(bbox, (cx, cy), "mouse")


# -- burned-in overlay (HUD) detection --------------------------------------

def sample_frames(path: str, count: int) -> List[np.ndarray]:
    """Read up to ``count`` frames spread across the clip (best-effort)."""
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > count * 2:
            for idx in np.linspace(0, total - 2, count).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
        else:
            while len(frames) < count:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
    except cv2.error:
        return frames
    finally:
        cap.release()
    return frames


def detect_static_overlay(frames: List[np.ndarray], config: SelectionConfig) -> Optional[np.ndarray]:
    """0/255 overlay mask, or ``None`` when detection isn't trustworthy.

    A pixel is overlay if it is temporally static (low std) AND structural (an
    edge in the temporal mean). Skipped unless enough of the frame is dynamic —
    without global motion an overlay can't be told from static real content.
    """
    if len(frames) < 5:
        return None
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    std = gray.std(axis=0)
    if float((std >= config.overlay_static_std).mean()) < config.overlay_min_motion_frac:
        return None
    static = std < config.overlay_static_std
    mean = gray.mean(axis=0).astype(np.uint8)
    edges = cv2.Canny(mean, 50, 150) > 0
    overlay = (static & edges).astype(np.uint8)
    overlay = cv2.morphologyEx(overlay, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if int(overlay.sum()) == 0:
        return None
    return overlay * 255


def box_overlaps_overlay(mask: np.ndarray, bbox: BBox) -> bool:
    frame_h, frame_w = mask.shape[:2]
    x, y, w, h = bbox.as_int_xywh()
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_w, x + w), min(frame_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return False
    return bool(mask[y0:y1, x0:x1].any())


def clean_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    dilated = cv2.dilate(mask, np.ones((_INPAINT_RADIUS, _INPAINT_RADIUS), np.uint8))
    return cv2.inpaint(frame, dilated, _INPAINT_RADIUS, cv2.INPAINT_TELEA)


def prepare_init(source_path: str, first_frame: np.ndarray, bbox: BBox,
                 config: SelectionConfig) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return ``(init_frame, hud_mask)`` for a HUD-safe start.

    Detects the overlay once. ``init_frame`` is the first frame with the overlay
    inpainted out *iff* the selection overlaps it (else unchanged). ``hud_mask``
    (or ``None``) is returned for reuse during tracking so appearance/flow matching
    can exclude the HUD pixels. Strict no-op on clean/static footage.
    """
    if not config.handle_overlay:
        return first_frame, None
    mask = detect_static_overlay(sample_frames(source_path, config.overlay_sample_frames), config)
    if mask is None:
        return first_frame, None
    init_frame = clean_overlay(first_frame, mask) if box_overlaps_overlay(mask, bbox) else first_frame
    return init_frame, mask
