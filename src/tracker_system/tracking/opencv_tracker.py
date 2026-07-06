"""Adapter wrapping OpenCV's classical trackers behind :class:`Tracker`.

Translates between the project's :class:`BBox` value type and the plain
``(x, y, w, h)`` integer tuples OpenCV expects, and normalises the
``(ok, bbox)`` update result. The concrete backend (CSRT/KCF/MOSSE) is chosen by
name and built via the capability-probing :mod:`factory`, so an unavailable
tracker fails with a clear error rather than an ``AttributeError``.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from ..utils.geometry import BBox
from .base import Tracker
from .factory import create_tracker


class OpenCVTracker(Tracker):
    """A :class:`Tracker` backed by an OpenCV tracker (CSRT by default)."""

    def __init__(self, tracker_type: str = "CSRT") -> None:
        self._type = tracker_type.upper()
        self._impl = None

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        # A fresh OpenCV tracker instance per init: they are not reusable across
        # re-initialisations in a well-defined way, so we always build a new one.
        self._impl = create_tracker(self._type)
        self._impl.init(frame, bbox.as_int_xywh())

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[BBox]]:
        if self._impl is None:
            raise RuntimeError("OpenCVTracker.update called before init()")
        ok, box = self._impl.update(frame)
        if not ok:
            return False, None
        x, y, w, h = box
        return True, BBox(float(x), float(y), float(w), float(h))

    @property
    def name(self) -> str:
        return self._type
