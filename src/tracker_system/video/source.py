"""Streaming video source over :class:`cv2.VideoCapture`.

Supports a local file path or a direct video URL (http/https/rtsp/...). Frames
are read sequentially and yielded one at a time so that memory usage depends on
frame resolution and enabled features, not on the total length of the video —
the streaming principle that underpins the whole system.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple
import cv2
import numpy as np

# Schemes we treat as a network/streaming URL rather than a local file path.
_URL_SCHEMES = ("http://","https://","rtsp://","rtmp://","udp://","tcp://")

class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be opened or read."""


def is_url(source: str) -> bool:
    """Return ``True`` if ``source`` looks like a direct video URL."""
    return isinstance(source, str) and source.lower().startswith(_URL_SCHEMES)


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata describing an opened video source.

    ``frame_count`` and therefore ``duration_seconds`` may be ``0`` for live or
    network streams where the backend cannot report a length in advance.
    """

    source: str
    is_url: bool
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return ``(width, height)`` in pixels."""
        return (self.width, self.height)

    @property
    def duration_seconds(self) -> float:
        """Best-effort duration in seconds (``0.0`` if unknown)."""
        if self.fps > 0 and self.frame_count > 0:
            return self.frame_count / self.fps
        return 0.0


class VideoSource:
    """Sequential reader for a local video file or direct video URL.

    Usage::

        with VideoSource(path).open() as vs:
            print(vs.metadata)
            for frame in vs.frames():
                ...  # process one frame at a time

    The context manager form opens on ``__enter__`` and always releases the
    underlying capture on ``__exit__``.
    """

    def __init__(self, source: str) -> None:
        self._source = str(source)
        self._is_url = is_url(self._source)
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    def open(self) -> "VideoSource":
        """Open the source and read its metadata. Idempotent."""
        if self._cap is not None:
            return self

        if not self._is_url:
            path = Path(self._source)
            if not path.exists():
                raise VideoSourceError(f"Video file not found: {self._source}")

        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            cap.release()
            raise VideoSourceError(f"Could not open video source: {self._source}")

        self._cap = cap
        self._metadata = self._read_metadata(cap)
        return self

    def release(self) -> None:
        """Release the underlying capture. Safe to call multiple times."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # Allows opening a video source inside a 'with' block, e.g. ``with VideoSource(path) as vs: ...``.
    def __enter__(self) -> "VideoSource":
        return self.open()
    # Allows releasing the video source by the end of the 'with' block
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


    @property
    def is_open(self) -> bool:
        return self._cap is not None

    @property
    def metadata(self) -> VideoMetadata:
        """Return metadata for the opened source."""
        if self._metadata is None:
            raise VideoSourceError("Video source is not open; call open() first")
        return self._metadata

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame.

        Returns ``(True, frame)`` on success and ``(False, None)`` at end of
        stream or on a read failure.
        """
        cap = self._require_open()
        ok, frame = cap.read()
        if not ok:
            return False, None
        return True, frame

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames sequentially until end of stream.

        Frames are produced one at a time; the caller is expected to process and
        discard each frame before requesting the next.
        """
        cap = self._require_open()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame

    # Requires that the video source is open and return a cv2.VideoCapture object. Raises VideoSourceError if not open.
    def _require_open(self) -> cv2.VideoCapture:
        if self._cap is None:
            raise VideoSourceError("Video source is not open; call open() first")
        return self._cap
    
    # Reads metadata from the given video capture object.
    def _read_metadata(self, cap: cv2.VideoCapture) -> VideoMetadata:
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps < 0 or fps != fps:  # guard against NaN / negative
            fps = 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 0:
            frame_count = 0
        return VideoMetadata(
            source=self._source,
            is_url=self._is_url,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )
