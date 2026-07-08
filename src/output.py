"""Annotated-video output management: persistent save vs. temporary preview.

The tracking pipeline always writes an annotated video to a path; this module
decides *which* path and what happens afterwards, keeping that policy out of the
CLI and the pipeline:

- with an explicit ``--save PATH``: write there and keep it;
- without: write to a throwaway temp file, replay it in a self-contained preview
  window after the run (pause / seek / restart), block until the user closes it,
  then delete the temp file — leaving the working directory clean.

The preview is played in-process via HighGUI (not the system default player) so
it terminates *deterministically* when the window is closed: no dependence on an
external app's quit lifecycle or automation permissions.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import cv2

_PREVIEW_WINDOW = "Tracking result  -  Space: pause   <-/->: seek   r: restart   Esc: close"
_ESC_KEY = 27
_SPACE_KEY = 32
# Arrow keys aren't ASCII and their codes are platform-specific (and only come
# back intact from waitKeyEx, not waitKey & 0xFF): Linux/GTK, Windows, macOS.
_LEFT_KEYS = (65361, 2424832, 63234)
_RIGHT_KEYS = (65363, 2555904, 63235)


def preview_video(path: str) -> None:
    """Replay ``path`` in a HighGUI window and block until the user closes it.

    Controls: ``Space`` pause/resume, ``<-``/``->`` seek back/forward ~2s, ``r``
    restart, ``Esc`` or the window close button to finish. Loops at EOF so the
    result stays up for inspection. Returns as soon as the window is gone,
    letting the caller delete the temp file.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return  # nothing to show (unreadable/empty file) — skip silently
    # Play back at the clip's own frame rate; ~2s per seek step.
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_delay = max(1, int(round(1000.0 / fps)))   # ms to wait between frames
    seek = max(1, int(round(fps * 2)))               # frames per left/right jump
    cv2.namedWindow(_PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    paused = False
    frame = None
    try:
        while True:
            # Advance to the next frame unless paused; loop back to the start at EOF.
            if not paused:
                ok, read = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = read
            if frame is not None:
                cv2.imshow(_PREVIEW_WINDOW, frame)
            # Poll one key (short wait while paused so the UI stays responsive).
            key = cv2.waitKeyEx(frame_delay if not paused else 50)
            if key == _ESC_KEY:
                break
            elif key == _SPACE_KEY:
                paused = not paused
            elif key in _LEFT_KEYS or key in _RIGHT_KEYS or key in (ord("r"), ord("R")):
                # Jump the playhead: left = back, right = forward, r = restart.
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                target = (max(0.0, pos - seek) if key in _LEFT_KEYS
                          else min(max(total - 1, 0), pos + seek) if key in _RIGHT_KEYS
                          else 0.0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                if paused:  # while paused, show the jumped-to frame right away
                    ok, read = cap.read()
                    if ok:
                        frame = read
            if cv2.getWindowProperty(_PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # user hit the window close button
    finally:
        cap.release()
        cv2.destroyWindow(_PREVIEW_WINDOW)
        for _ in range(4):  # pump the event loop so the window actually closes
            cv2.waitKey(1)


class AnnotatedVideoOutput:
    """Owns the annotated-video path and its save/temp/preview/cleanup lifecycle.

    ``save_path is None`` selects temporary-preview mode; otherwise the video is
    written to ``save_path`` and preserved.
    """

    def __init__(self, save_path: Optional[str] = None,
                 opener=preview_video) -> None:
        self.save_path = save_path
        self._opener = opener            # injectable so tests can stub the preview
        self._temp_path: Optional[str] = None
        # No save path -> allocate a scratch file in the system temp dir to write into.
        if save_path is None:
            fd, self._temp_path = tempfile.mkstemp(prefix="tracker_preview_", suffix=".mp4")
            os.close(fd)  # we only need the path; the pipeline reopens it to write

    @property
    def is_temporary(self) -> bool:
        """True in preview-and-delete mode (no ``--save`` was given)."""
        return self.save_path is None

    @property
    def path(self) -> str:
        """The path the pipeline should write the annotated video to."""
        return self._temp_path if self.save_path is None else self.save_path  # type: ignore[return-value]

    def finish(self, success: bool = True) -> None:
        """Preview then delete a temp result; no-op when a save path was given."""
        if not self.is_temporary:
            return
        try:
            if success and self._temp_path and os.path.exists(self._temp_path) \
                    and os.path.getsize(self._temp_path) > 0:
                self._opener(self._temp_path)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Delete the temp file if one was created (safe to call repeatedly)."""
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.remove(self._temp_path)
            except OSError:
                pass
        self._temp_path = None
