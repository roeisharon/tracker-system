"""Frames-per-second meter.

Fed one per-frame processing duration at a time (``update(dt)``). It exposes a
smoothed *rolling* value for the on-screen HUD and cumulative *average / min /
max* for the end-of-run performance check. It measures whatever block the caller
times — the pipeline times the core tracking work (resize + tracker update +
overlay), excluding disk encoding and the one-time selection step, so the number
reflects real tracking throughput.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional


class FpsMeter:
    """Accumulates per-frame durations and derives FPS statistics."""

    def __init__(self, window: int = 30) -> None:
        self._recent: deque = deque(maxlen=window)
        self._total_time = 0.0
        self._frames = 0
        self._min: Optional[float] = None
        self._max: Optional[float] = None

    def update(self, dt: float) -> float:
        """Record a frame that took ``dt`` seconds; return the rolling FPS."""
        if dt <= 0.0:
            dt = 1e-6
        fps = 1.0 / dt
        self._recent.append(fps)
        self._total_time += dt
        self._frames += 1
        self._min = fps if self._min is None else min(self._min, fps)
        self._max = fps if self._max is None else max(self._max, fps)
        return self.rolling

    @property
    def rolling(self) -> float:
        """Smoothed FPS over the recent window (for display)."""
        if not self._recent:
            return 0.0
        return sum(self._recent) / len(self._recent)

    @property
    def average(self) -> float:
        """Overall average FPS = frames / total processing time."""
        return self._frames / self._total_time if self._total_time > 0 else 0.0

    @property
    def min(self) -> float:
        return self._min if self._min is not None else 0.0

    @property
    def max(self) -> float:
        return self._max if self._max is not None else 0.0

    @property
    def frames(self) -> int:
        return self._frames

    def summary(self) -> Dict[str, float]:
        """Return a plain dict of the aggregate statistics."""
        return {
            "frames": self._frames,
            "avg_fps": self.average,
            "min_fps": self.min,
            "max_fps": self.max,
        }
