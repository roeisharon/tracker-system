"""Tests for the streaming VideoSource on a tiny generated clip."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tracker_system.video.source import (
    VideoSource,
    VideoSourceError,
    is_url,
)

WIDTH = 64
HEIGHT = 48
NUM_FRAMES = 10
FPS = 20.0


@pytest.fixture
def tiny_clip(tmp_path):
    """Write a small deterministic AVI clip and return its path."""
    path = tmp_path / "clip.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), "failed to open VideoWriter for test clip"
    for i in range(NUM_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_is_url():
    assert is_url("http://example.com/a.mp4")
    assert is_url("https://example.com/a.mp4")
    assert is_url("rtsp://host/stream")
    assert not is_url("sample-video.mp4")
    assert not is_url("/absolute/path/clip.mp4")


def test_open_reads_metadata(tiny_clip):
    with VideoSource(str(tiny_clip)) as video:
        meta = video.metadata
        assert meta.width == WIDTH
        assert meta.height == HEIGHT
        assert meta.resolution == (WIDTH, HEIGHT)
        assert meta.frame_count == NUM_FRAMES
        assert meta.fps > 0
        assert meta.is_url is False
        assert meta.duration_seconds > 0


def test_frames_streams_all_frames_then_stops(tiny_clip):
    with VideoSource(str(tiny_clip)) as video:
        frames = list(video.frames())
    assert len(frames) == NUM_FRAMES
    assert all(f.shape == (HEIGHT, WIDTH, 3) for f in frames)


def test_read_returns_false_at_eof(tiny_clip):
    with VideoSource(str(tiny_clip)) as video:
        count = 0
        while True:
            ok, frame = video.read()
            if not ok:
                assert frame is None
                break
            count += 1
        assert count == NUM_FRAMES


def test_missing_file_raises():
    with pytest.raises(VideoSourceError):
        VideoSource("/no/such/file.mp4").open()


def test_metadata_before_open_raises():
    video = VideoSource("whatever.mp4")
    with pytest.raises(VideoSourceError):
        _ = video.metadata


def test_read_before_open_raises():
    video = VideoSource("whatever.mp4")
    with pytest.raises(VideoSourceError):
        video.read()


def test_release_is_idempotent(tiny_clip):
    video = VideoSource(str(tiny_clip)).open()
    video.release()
    video.release()  # should not raise
    assert video.is_open is False
