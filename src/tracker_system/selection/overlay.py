"""Detect and remove burned-in screen-fixed overlays at selection time.

Some footage composites a static graphic overlay on top of every frame — a drone
HUD crosshair and guide-lines, a timestamp, a channel logo, a watermark. When the
user selects a target near such an overlay, the initial box is dominated by the
high-contrast overlay pixels, so the Target Profile (template + histogram) and the
tracker's correlation filter model the *overlay* rather than the physical object.
Because the overlay is screen-fixed while the world moves, the tracker then follows
the overlay (or, once pulled off it, cannot re-acquire the real target).

The defining, video-agnostic property of a burned-in overlay is that it is
**screen-fixed (temporally constant) while the world moves**. This module samples
frames spread across the video, flags pixels that stay constant *despite* global
motion and carry structure (edges/lines/text), and returns an overlay mask. The
mask is then used only at **profile initialisation**: if the selected box overlaps
the overlay, the overlay is inpainted out of the first-frame patch before the
template and tracker are built. The live tracking / re-acquisition path is never
touched.

It is deliberately conservative:

- it runs only when the sampled frames show **enough global motion** to tell an
  overlay apart from genuinely static scene content (otherwise it is a no-op — and
  on a static scene a static overlay does not diverge from the world anyway);
- it cleans the first frame **only when the selection actually overlaps** detected
  overlay pixels, so clean footage and selections away from any overlay are
  completely unaffected;
- any sampling/decoding failure (e.g. a non-seekable stream) degrades to a no-op.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ..config.settings import SelectionConfig
from ..utils.geometry import BBox

# Dilation + inpaint radius (px) applied around detected overlay pixels so thin
# anti-aliased line/text edges are fully removed before inpainting.
_INPAINT_RADIUS = 3


def sample_frames(path: str, count: int) -> List[np.ndarray]:
    """Read up to ``count`` frames spread across the video (best-effort).

    Frames are sampled across the whole clip so the world has moved between them
    (which is what exposes a screen-fixed overlay). Falls back to consecutive
    frames for short clips or sources with an unknown length; returns whatever it
    managed to read (possibly empty) rather than raising.
    """
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


def detect_static_overlay(
    frames: List[np.ndarray], config: SelectionConfig
) -> Optional[np.ndarray]:
    """Return a 0/255 overlay mask, or ``None`` when detection is not trustworthy.

    A pixel is overlay if it is **temporally static** (low std across the sampled
    frames) **and structural** (an edge/line/text pixel in the temporal mean).
    Detection is skipped (returns ``None``) unless a sufficient fraction of the
    frame is dynamic — without global motion an overlay cannot be told apart from
    static real content.
    """
    if len(frames) < 5:
        return None
    gray = np.stack(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    )
    std = gray.std(axis=0)
    dynamic_fraction = float((std >= config.overlay_static_std).mean())
    if dynamic_fraction < config.overlay_min_motion_frac:
        return None  # not enough world motion to trust static-pixel detection

    static = std < config.overlay_static_std
    mean = gray.mean(axis=0).astype(np.uint8)
    edges = cv2.Canny(mean, 50, 150) > 0
    overlay = (static & edges).astype(np.uint8)
    overlay = cv2.morphologyEx(overlay, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if int(overlay.sum()) == 0:
        return None
    return overlay * 255


def box_overlaps_overlay(mask: np.ndarray, bbox: BBox) -> bool:
    """Whether the selection box contains any detected overlay pixels."""
    frame_h, frame_w = mask.shape[:2]
    x, y, w, h = bbox.as_int_xywh()
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_w, x + w), min(frame_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return False
    return bool(mask[y0:y1, x0:x1].any())


def clean_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint the overlay out of ``frame`` (used only for the init patch)."""
    dilated = cv2.dilate(mask, np.ones((_INPAINT_RADIUS, _INPAINT_RADIUS), np.uint8))
    return cv2.inpaint(frame, dilated, _INPAINT_RADIUS, cv2.INPAINT_TELEA)


def overlay_free_first_frame(
    source_path: str, first_frame: np.ndarray, bbox: BBox, config: SelectionConfig
) -> np.ndarray:
    """Return an overlay-free copy of ``first_frame`` for profile initialisation.

    Returns the frame unchanged unless overlay handling is enabled, a static
    overlay is detected, and the selection box overlaps it — so it is a strict
    no-op on clean footage or selections away from any overlay.
    """
    if not config.handle_overlay:
        return first_frame
    frames = sample_frames(source_path, config.overlay_sample_frames)
    mask = detect_static_overlay(frames, config)
    if mask is None or not box_overlaps_overlay(mask, bbox):
        return first_frame
    return clean_overlay(first_frame, mask)
