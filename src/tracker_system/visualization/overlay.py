"""Drawing the tracking overlay onto full-resolution frames.

Pure OpenCV drawing helpers plus a single :func:`render_overlay` that composes
them: the trajectory trail, the bounding box (green while tracking, red when
lost), an optional seed marker, and a small translucent HUD panel showing state,
tracker, and FPS. Drawing happens in place on the frame that will be written to
the output video or shown on screen.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..target.profile import TargetProfile
from ..utils.geometry import BBox

# BGR colours.
COLOR_TRACKING = (0, 200, 0)
COLOR_LOST = (0, 0, 255)
COLOR_SEARCHING = (0, 165, 255)
COLOR_SEED = (0, 215, 255)
COLOR_TRAIL = (255, 180, 0)
_HUD_TEXT = (255, 255, 255)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_MAX_TRAIL_POINTS = 64

# Box colour per state (green = healthy, orange = searching, red = lost/other).
_STATE_COLORS = {
    "TRACKING": COLOR_TRACKING,
    "REACQUIRED": COLOR_TRACKING,
    "READY": COLOR_TRACKING,
    "SEARCHING": COLOR_SEARCHING,
    "LOST": COLOR_LOST,
}


def state_color(state: str) -> Tuple[int, int, int]:
    """Return the box colour for a state name."""
    return _STATE_COLORS.get(state, COLOR_LOST)


def draw_bbox(
    frame: np.ndarray, bbox: BBox, color: Tuple[int, int, int], thickness: int = 2
) -> None:
    x, y, w, h = bbox.as_int_xywh()
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)


def draw_trajectory(
    frame: np.ndarray,
    points: Sequence[Tuple[int, int]],
    color: Tuple[int, int, int] = COLOR_TRAIL,
    max_points: int = _MAX_TRAIL_POINTS,
) -> None:
    """Draw the recent centre trail as a polyline."""
    pts = list(points)[-max_points:]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)


def draw_seed(
    frame: np.ndarray, point: Tuple[int, int], color: Tuple[int, int, int] = COLOR_SEED
) -> None:
    cv2.drawMarker(frame, point, color, cv2.MARKER_CROSS, 18, 2)


def draw_search_region(
    frame: np.ndarray, bbox: BBox, color: Tuple[int, int, int] = COLOR_SEARCHING
) -> None:
    """Draw the (thin) SEARCHING region rectangle for explainability."""
    x, y, w, h = bbox.as_int_xywh()
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)


def draw_debug_search(
    frame: np.ndarray,
    search_region: Optional[BBox],
    candidates,
    accepted,
    predicted_center: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Debug view: search region, every candidate + score, and the selected one.

    Candidates are drawn in yellow with their weighted score; the accepted
    candidate (if any) is drawn thick green. Used only by ``--debug-vis``.
    """
    if search_region is not None:
        draw_search_region(frame, search_region)
    if predicted_center is not None:
        cv2.drawMarker(frame, predicted_center, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
    for c in candidates:
        x, y, w, h = c.bbox.as_int_xywh()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.putText(
            frame, f"{c.score:.2f}", (x, max(10, y - 3)),
            _FONT, 0.35, (0, 255, 255), 1, cv2.LINE_AA,
        )
    if accepted is not None:
        x, y, w, h = accepted.bbox.as_int_xywh()
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_TRACKING, 2)
    return frame


def draw_hud(
    frame: np.ndarray, lines: List[str], origin: Tuple[int, int] = (10, 10)
) -> None:
    """Draw a translucent panel with one text line per entry."""
    if not lines:
        return
    pad, line_h, scale, thick = 6, 22, 0.6, 1
    text_w = max(
        cv2.getTextSize(line, _FONT, scale, thick)[0][0] for line in lines
    )
    box_w = text_w + 2 * pad
    box_h = line_h * len(lines) + 2 * pad
    x0, y0 = origin

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, line in enumerate(lines):
        y = y0 + pad + line_h * (i + 1) - 6
        cv2.putText(
            frame, line, (x0 + pad, y), _FONT, scale, _HUD_TEXT, thick, cv2.LINE_AA
        )


def render_overlay(
    frame: np.ndarray,
    profile: TargetProfile,
    state: str,
    fps: float,
    tracker_name: str,
    seed_point: Optional[Tuple[int, int]] = None,
    reason: Optional[str] = None,
    search_region: Optional[BBox] = None,
) -> np.ndarray:
    """Compose the full overlay in place and return the frame.

    The box colour reflects ``state`` (green healthy, orange searching, red
    lost). The seed marker is drawn only when provided (first frame), the
    ``search_region`` is drawn while SEARCHING, and the loss/status ``reason`` is
    shown in the HUD when present.
    """
    color = state_color(state)
    if search_region is not None:
        draw_search_region(frame, search_region)
    draw_trajectory(frame, profile.trajectory)
    draw_bbox(frame, profile.current_bbox, color)
    if seed_point is not None:
        draw_seed(frame, seed_point)
    lines = [
        f"State:   {state}",
        f"Tracker: {tracker_name}",
        f"FPS:     {fps:5.1f}",
    ]
    if reason:
        lines.append(f"Reason:  {reason}")
    draw_hud(frame, lines)
    return frame
