"""Tests for the streaming video source."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from video import VideoSource, VideoSourceError, is_url


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    assert w.isOpened()
    for _ in range(10):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    return path


def test_is_url():
    assert is_url("http://x/y.mp4") and not is_url("/local/file.mp4")


def test_metadata_and_iteration(clip):
    with VideoSource(str(clip)) as vs:
        assert vs.metadata.width == 64 and vs.metadata.height == 48
        assert sum(1 for _ in vs.frames()) == 10


def test_missing_file_raises():
    with pytest.raises(VideoSourceError):
        VideoSource("/no/such/file.mp4").open()


def test_metadata_before_open_raises():
    with pytest.raises(VideoSourceError):
        _ = VideoSource("x").metadata
