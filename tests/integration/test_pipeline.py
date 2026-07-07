"""Integration tests: the pipeline runs generated clips end to end.

These use the classical ``csrt`` backend so no model asset is needed and runs
stay fast and deterministic; the hybrid/ViT path is exercised by the drone
benchmark, not the unit suite.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from config import Settings
from geometry import BBox, bbox_from_center
from pipeline import TrackingPipeline
from selection import ManualPixelSelector, SelectionResult, TargetSelector
from video import VideoSource

WIDTH, HEIGHT, FPS, SQUARE = 320, 240, 20.0, 40


def _settings(**loss_over):
    s = Settings()
    if loss_over:
        s = replace(s, loss=replace(s.loss, **loss_over))
    return s


class StubSelector(TargetSelector):
    def __init__(self, bbox, seed):
        self._bbox, self._seed = bbox, seed

    def select(self, frame) -> SelectionResult:
        return SelectionResult(self._bbox, self._seed, "stub")


def _write(path, draw):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    assert w.isOpened()
    n = 0
    while True:
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        if not draw(n, frame):
            break
        w.write(frame)
        n += 1
    w.release()


@pytest.fixture
def moving_square(tmp_path):
    positions = []

    def draw(i, frame):
        if i >= 30:
            return False
        x, y = 40 + i * 6, HEIGHT // 2
        cv2.rectangle(frame, (x, y - SQUARE // 2), (x + SQUARE, y + SQUARE // 2), (0, 0, 255), -1)
        positions.append((x + SQUARE // 2, y))
        return True

    path = tmp_path / "clip.avi"
    _write(path, draw)
    return path, positions


def test_pipeline_runs_to_eof_and_writes(moving_square, tmp_path):
    path, pos = moving_square
    out = tmp_path / "out.mp4"
    sel = ManualPixelSelector(row=pos[0][1], col=pos[0][0], bbox_size=SQUARE + 10)
    result = TrackingPipeline(_settings(), backend="csrt").run(VideoSource(str(path)), sel, out_path=str(out))
    assert result.frames_processed == 30
    assert result.avg_fps > 0 and out.exists() and out.stat().st_size > 0


def test_pipeline_is_selector_agnostic(moving_square):
    path, pos = moving_square
    x, y = pos[0]
    sel = StubSelector(bbox_from_center(x, y, SQUARE + 10), (x, y))
    result = TrackingPipeline(_settings(), backend="csrt").run(VideoSource(str(path)), sel, out_path=None)
    assert result.frames_processed == 30


def test_pipeline_final_state_tracking(moving_square):
    path, pos = moving_square
    sel = ManualPixelSelector(row=pos[0][1], col=pos[0][0], bbox_size=SQUARE + 10)
    result = TrackingPipeline(_settings(), backend="csrt").run(VideoSource(str(path)), sel, out_path=None)
    assert result.final_state == "TRACKING"


@pytest.fixture
def leave_and_return(tmp_path):
    rng = np.random.default_rng(11)
    tex = rng.integers(0, 255, (SQUARE, SQUARE, 3), dtype=np.uint8)
    tx, ty = 60, HEIGHT // 2 - SQUARE // 2
    present, gone, back = 6, 10, 20

    def draw(i, frame):
        if i >= present + gone + back:
            return False
        if i < present or i >= present + gone:
            frame[ty:ty + SQUARE, tx:tx + SQUARE] = tex
        return True

    path = tmp_path / "return.avi"
    _write(path, draw)
    return path, (tx + SQUARE // 2, ty + SQUARE // 2)


def test_pipeline_reacquires_returning_target(leave_and_return):
    path, (sx, sy) = leave_and_return
    sel = ManualPixelSelector(row=sy, col=sx, bbox_size=SQUARE + 8)
    settings = _settings(t_lost=0.4, lost_patience=3)
    result = TrackingPipeline(settings, backend="csrt").run(VideoSource(str(path)), sel, out_path=None)
    assert result.lost_events >= 1
    states = [e.state.value for e in result.timeline]
    assert "SEARCHING" in states
