"""Target selection interface and the manual ``[i, j]`` selector.

Both selection methods (manual coordinates and mouse click) implement the same
:class:`TargetSelector` interface and return the same :class:`SelectionResult`,
so the pipeline is completely selection-agnostic — it just calls
``selector.select(first_frame)``. This is what lets "manual" and "mouse" flow
through one identical tracking pipeline.

Coordinate convention: the assignment specifies the manual input as indices
``[i, j]`` meaning **row, column**. So ``i`` is the row (``y``) and ``j`` is the
column (``x``). Internally everything else is ``(x, y)`` to match OpenCV; the
manual selector is the one place that performs the ``[i, j] -> (x, y)`` mapping.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from ..utils.geometry import BBox, bbox_from_center, clamp_bbox


class SelectionError(RuntimeError):
    """Raised when a target cannot be selected (out of frame, cancelled, ...)."""


@dataclass(frozen=True)
class SelectionResult:
    """The outcome of selecting a target on the first frame.

    Attributes:
        bbox: Initial bounding box in full-resolution ``(x, y, w, h)`` pixels.
        seed_point: The originating ``(x, y)`` pixel (click or mapped ``[i, j]``).
        source: ``"manual"`` or ``"mouse"`` — provenance, for logging/overlay.
    """

    bbox: BBox
    seed_point: Tuple[int, int]
    source: str


class TargetSelector(ABC):
    """Interface for choosing the initial target on the first frame."""

    @abstractmethod
    def select(self, frame: np.ndarray) -> SelectionResult:
        """Return the selected target for ``frame`` or raise :class:`SelectionError`."""


class ManualPixelSelector(TargetSelector):
    """Select a target from manual ``[i, j]`` (row, column) coordinates."""

    def __init__(self, row: int, col: int, bbox_size: int) -> None:
        self.row = int(row)
        self.col = int(col)
        self.bbox_size = int(bbox_size)

    def select(self, frame: np.ndarray) -> SelectionResult:
        frame_h, frame_w = frame.shape[:2]
        x, y = self.col, self.row  # [i, j] = (row, col) -> (x, y)
        if not (0 <= x < frame_w and 0 <= y < frame_h):
            raise SelectionError(
                f"Pixel [i={self.row}, j={self.col}] is outside the "
                f"{frame_w}x{frame_h} frame"
            )
        bbox = clamp_bbox(bbox_from_center(x, y, self.bbox_size), frame_w, frame_h)
        return SelectionResult(bbox=bbox, seed_point=(x, y), source="manual")
