"""Target selection + burned-in overlay (HUD) handling.

Two selectors — manual ``[i, j]`` (for scripted/reproducible runs) and the
interactive click/resize UI (the default) — return the same
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

from config import SelectionConfig
from geometry import BBox, bbox_from_center, clamp_bbox
from overlay import draw_hud

_INPAINT_RADIUS = 3
_SELECT_COLOR = (0, 255, 0)   # lime — box outline + centre marker


class SelectionError(RuntimeError):
    """Raised when a target cannot be selected (out of frame, cancelled, ...)."""


@dataclass(frozen=True)
class SelectionResult:
    """The chosen target: its box, the exact click/seed point, and how it was picked."""

    bbox: BBox
    seed_point: Tuple[int, int]
    source: str  # "manual" (--pixel) or "interactive" (click UI)


class TargetSelector(ABC):
    """Interface for the two ways to pick a target — kept uniform so the pipeline
    doesn't care which one is used."""

    @abstractmethod
    def select(self, frame: np.ndarray) -> SelectionResult:
        """Return the selected target for ``frame`` or raise :class:`SelectionError`."""


class ManualPixelSelector(TargetSelector):
    """Select from manual ``[i, j]`` (row, column) coordinates."""

    def __init__(self, row: int, col: int, bbox_size: int) -> None:
        self.row, self.col, self.bbox_size = int(row), int(col), int(bbox_size)

    def select(self, frame: np.ndarray) -> SelectionResult:
        """Build a box centred on the given pixel; error if it's off-frame."""
        frame_h, frame_w = frame.shape[:2]
        x, y = self.col, self.row  # note: input is (row, col); OpenCV is (x, y)
        if not (0 <= x < frame_w and 0 <= y < frame_h):
            raise SelectionError(
                f"Pixel [i={self.row}, j={self.col}] is outside the {frame_w}x{frame_h} frame"
            )
        bbox = clamp_bbox(bbox_from_center(x, y, self.bbox_size), frame_w, frame_h)
        return SelectionResult(bbox, (x, y), "manual")


class InteractiveClickSelector(TargetSelector):
    """Place, drag, and resize the initial box on the first frame, then confirm.

    Controls: click (or click-drag) sets/moves the box centre; ``+``/``-`` resize
    the box within the configured ``[min_size, max_size]`` bounds; ``Enter``/
    ``Space`` confirm; ``Esc`` cancels. This class owns only the selection UI.
    """

    _ENTER_KEYS = (13, 10)
    _SPACE_KEY = 32
    _ESC_KEY = 27
    _GROW_KEYS = (ord("+"), ord("="))      # '=' so grow works without Shift
    _SHRINK_KEYS = (ord("-"), ord("_"))

    def __init__(self, bbox_size: int, min_size: int, max_size: int,
                 window_name: str = "Select target", resize_step: int = 10) -> None:
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.bbox_size = self._clamp_size(int(bbox_size))
        self.window_name = window_name
        self.resize_step = int(resize_step)
        self._frame_shape: Optional[Tuple[int, int]] = None
        self._center: Optional[Tuple[int, int]] = None
        self._size = self.bbox_size

    def select(self, frame: np.ndarray) -> SelectionResult:
        """Open the picker window and block until the user confirms or cancels."""
        self._frame_shape = frame.shape[:2]
        self._center = None
        self._size = self.bbox_size
        confirmed = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        try:
            # Redraw + poll keys until the user confirms, cancels, or closes the window.
            while True:
                cv2.imshow(self.window_name, self._render(frame))
                key = cv2.waitKey(20) & 0xFF
                if key == self._ESC_KEY:
                    break
                # Confirm only once a centre has been placed.
                if (key in self._ENTER_KEYS or key == self._SPACE_KEY) and self._center is not None:
                    confirmed = True
                    break
                elif key in self._GROW_KEYS:
                    self._resize(self.resize_step)
                elif key in self._SHRINK_KEYS:
                    self._resize(-self.resize_step)
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed via the title-bar button
        finally:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        if not confirmed or self._center is None:
            raise SelectionError("Target selection cancelled")
        return self._build_result()

    # State updates, kept free of GUI calls so they can be unit-tested.

    def _on_mouse(self, event, x, y, flags, param) -> None:
        # Click or click-drag both set the centre (drag = continuous move).
        if event == cv2.EVENT_LBUTTONDOWN or (
                event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
            self._center = self._clamp_point(x, y)

    def _resize(self, delta: int) -> None:
        """Grow/shrink the box side by ``delta``, staying within the size bounds."""
        self._size = self._clamp_size(self._size + delta)

    def _clamp_size(self, size: int) -> int:
        """Keep a box side within the configured [min, max] range."""
        return int(min(self.max_size, max(self.min_size, size)))

    def _clamp_point(self, x: int, y: int) -> Tuple[int, int]:
        """Keep a clicked point inside the frame."""
        if self._frame_shape is None:
            raise SelectionError("Selector has no frame; call select() first")
        frame_h, frame_w = self._frame_shape
        return (min(max(int(x), 0), frame_w - 1), min(max(int(y), 0), frame_h - 1))

    def _current_bbox(self) -> BBox:
        """The box implied by the current centre + size (clamped to the frame)."""
        assert self._center is not None and self._frame_shape is not None
        frame_h, frame_w = self._frame_shape
        cx, cy = self._center
        return clamp_bbox(bbox_from_center(cx, cy, self._size), frame_w, frame_h)

    def _build_result(self) -> SelectionResult:
        """Package the confirmed centre + size into a SelectionResult."""
        assert self._center is not None
        return SelectionResult(self._current_bbox(), self._center, "interactive")

    # Rendering the box, centre marker, and controls overlay each frame.

    def _render(self, frame: np.ndarray) -> np.ndarray:
        """Draw the current box, centre marker, and help panel onto a frame copy."""
        display = frame.copy()
        if self._center is not None:
            x, y, w, h = self._current_bbox().as_int_xywh()
            cv2.rectangle(display, (x, y), (x + w, y + h), _SELECT_COLOR, 2, cv2.LINE_AA)
            cx, cy = self._center
            cv2.drawMarker(display, (cx, cy), _SELECT_COLOR, cv2.MARKER_CROSS, 16, 2)
            cv2.circle(display, (cx, cy), 3, _SELECT_COLOR, -1, cv2.LINE_AA)
        self._draw_hud(display)
        return display

    def _draw_hud(self, display: np.ndarray) -> None:
        """Show the control hints, plus the live centre-pixel and box-size readout."""
        # Build the text lines, then hand them to the shared HUD-panel renderer.
        lines = ["Click / drag: set centre", "+ / - : resize box"]
        if self._center is not None:
            cx, cy = self._center
            lines.append(f"pixel [i,j]: {cy}, {cx}")
        lines.append(f"box size: {self._size}px  (min {self.min_size} / max {self.max_size})")
        lines.append("Enter / Space: confirm    Esc: cancel")
        if self._center is None:
            lines.insert(0, "Click the target to place the box")
        draw_hud(display, lines)


# Burned-in overlay (HUD) detection: find and inpaint the screen-fixed crosshair.

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
    """True if any HUD-overlay pixel falls inside ``bbox`` (needs inpainting)."""
    frame_h, frame_w = mask.shape[:2]
    x, y, w, h = bbox.as_int_xywh()
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_w, x + w), min(frame_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return False
    return bool(mask[y0:y1, x0:x1].any())


def clean_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint the (dilated) HUD-overlay pixels out of ``frame``."""
    dilated = cv2.dilate(mask, np.ones((_INPAINT_RADIUS, _INPAINT_RADIUS), np.uint8))
    return cv2.inpaint(frame, dilated, _INPAINT_RADIUS, cv2.INPAINT_TELEA)


def prepare_init(source_path: str, first_frame: np.ndarray, bbox: BBox,
                 config: SelectionConfig) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return ``(init_frame, hud_mask)`` for a HUD-safe start.

    Detects the overlay once. ``init_frame`` is the first frame with the overlay
    inpainted out *iff* the selection overlaps it (else unchanged). ``hud_mask``
    (or ``None``) is returned for reuse during tracking so appearance/flow
    matching can exclude the HUD pixels. Strict no-op on clean/static footage.
    """
    if not config.handle_overlay:
        return first_frame, None
    mask = detect_static_overlay(sample_frames(source_path, config.overlay_sample_frames), config)
    if mask is None:
        return first_frame, None
    init_frame = clean_overlay(first_frame, mask) if box_overlaps_overlay(mask, bbox) else first_frame
    return init_frame, mask
