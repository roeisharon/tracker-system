"""Streaming video source over ``cv2.VideoCapture``.

Reads a local file or a direct URL (http/https/rtsp/...) one frame at a time, so
memory stays flat regardless of clip length.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

_URL_SCHEMES = ("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://")


class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be opened or read."""


def is_url(source: str) -> bool:
    return isinstance(source, str) and source.lower().startswith(_URL_SCHEMES)


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata for an opened source (``frame_count`` may be 0 for live streams)."""

    source: str
    is_url: bool
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def duration_seconds(self) -> float:
        if self.fps > 0 and self.frame_count > 0:
            return self.frame_count / self.fps
        return 0.0


class VideoSource:
    """Sequential reader for a video file or URL; usable as a context manager."""

    def __init__(self, source: str) -> None:
        self._source = str(source)
        self._is_url = is_url(self._source)
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    def open(self) -> "VideoSource":
        if self._cap is not None:
            return self
        if not self._is_url and not Path(self._source).exists():
            raise VideoSourceError(f"Video file not found: {self._source}")
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            cap.release()
            raise VideoSourceError(f"Could not open video source: {self._source}")
        self._cap = cap
        self._metadata = self._read_metadata(cap)
        return self

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise VideoSourceError("Video source is not open; call open() first")
        return self._metadata

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self._require_open().read()
        return (True, frame) if ok else (False, None)

    def frames(self) -> Iterator[np.ndarray]:
        cap = self._require_open()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame

    def _require_open(self) -> cv2.VideoCapture:
        if self._cap is None:
            raise VideoSourceError("Video source is not open; call open() first")
        return self._cap

    def _read_metadata(self, cap: cv2.VideoCapture) -> VideoMetadata:
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps < 0 or fps != fps:  # NaN / negative guard
            fps = 0.0
        frame_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        return VideoMetadata(self._source, self._is_url, width, height, fps, frame_count)
