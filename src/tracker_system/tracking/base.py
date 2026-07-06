"""The pluggable tracker interface.

Every tracker backend implements the same tiny contract so the pipeline never
depends on a specific algorithm. This is the seam that later phases use to swap
CSRT for KCF/MOSSE (modes) or to reinitialise after re-acquisition.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from ..utils.geometry import BBox


class Tracker(ABC):
    """Common interface for single-object trackers.

    Lifecycle mirrors the design in ``project-overview.md``: initialise on a
    frame + box, update per frame, and reset (reinitialise) when the target is
    re-acquired.
    """

    @abstractmethod
    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        """Start tracking ``bbox`` in ``frame``."""

    @abstractmethod
    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[BBox]]:
        """Advance to the next frame.

        Returns ``(True, bbox)`` when the target is located, or
        ``(False, None)`` when the tracker reports failure.
        """

    def reset(self, frame: np.ndarray, bbox: BBox) -> None:
        """Reinitialise the tracker on a new box (used after re-acquisition)."""
        self.init(frame, bbox)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable tracker name (e.g. ``"CSRT"``)."""
