"""Headless tracking pipeline: state machine + orchestration.

Streams frames one at a time; per frame it advances the tracker, fuses a
confidence (tracker score + appearance verifier), detects loss with hysteresis,
searches and re-acquires when lost, drives an explicit state machine, draws the
overlay, and optionally writes an annotated video.

State flow: ``INIT -> READY -> TRACKING -> LOST -> SEARCHING -> REACQUIRED ->
TRACKING``. Tracking runs the cheap per-frame work; the heavier full-frame search
runs only while SEARCHING.
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple
import cv2
from appearance import AppearanceMemory, Verifier
from config import Settings
from geometry import clamp_bbox, resize_frame
from loss import LossDetector
from motion import GlobalMotionEstimator
from overlay import draw_debug_search, render_overlay
from reacquire import Reacquirer
from selection import TargetSelector, prepare_init
from trackers import create_backend
from video import VideoSource

ProgressCallback = Callable[[int, int], None]


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot run (empty video, bad output path, ...)."""


# State machine: the states, the legal transitions between them, and the timeline.

class TrackerState(Enum):
    """The lifecycle stages a tracked target moves through."""

    INIT = "INIT"              # before the first frame
    READY = "READY"            # target selected, about to track
    TRACKING = "TRACKING"      # actively following the target
    LOST = "LOST"              # confidence collapsed; target gone
    SEARCHING = "SEARCHING"    # scanning the frame to find it again
    REACQUIRED = "REACQUIRED"  # just re-found; resumes tracking next frame


# Which states each state is allowed to move to (anything else is a bug).
_ALLOWED = {
    TrackerState.INIT: {TrackerState.READY},
    TrackerState.READY: {TrackerState.TRACKING},
    TrackerState.TRACKING: {TrackerState.LOST},
    TrackerState.LOST: {TrackerState.SEARCHING, TrackerState.TRACKING},
    TrackerState.SEARCHING: {TrackerState.REACQUIRED, TrackerState.LOST},
    TrackerState.REACQUIRED: {TrackerState.TRACKING, TrackerState.LOST},
}


@dataclass(frozen=True)
class TimelineEvent:
    """One recorded state change: when it happened and why."""

    frame_index: int
    state: TrackerState
    reason: str


@dataclass
class StateMachine:
    """Current state plus the running log of every transition (the timeline)."""

    state: TrackerState = TrackerState.INIT
    timeline: List[TimelineEvent] = field(default_factory=list)

    def to(self, new_state: TrackerState, frame_index: int, reason: str = "") -> None:
        """Move to ``new_state`` and log it; no-op on a self-transition, error if illegal."""
        if new_state == self.state:
            return
        if new_state not in _ALLOWED[self.state]:
            raise RuntimeError(f"Illegal transition {self.state.value} -> {new_state.value}")
        self.state = new_state
        self.timeline.append(TimelineEvent(frame_index, new_state, reason))

    def count(self, state: TrackerState) -> int:
        """How many times the machine entered ``state`` (e.g. number of LOST events)."""
        return sum(1 for e in self.timeline if e.state == state)


class FpsMeter:
    """Per-frame duration accumulator -> rolling/average/min/max FPS."""

    def __init__(self, window: int = 30) -> None:
        self._recent: List[float] = []
        self._window = window
        self._total_time = 0.0
        self._frames = 0
        self._min: Optional[float] = None
        self._max: Optional[float] = None

    def update(self, dt: float) -> float:
        """Record one frame's duration (seconds) and return the rolling FPS."""
        dt = max(dt, 1e-6)  # guard against a zero duration -> division by zero
        fps = 1.0 / dt
        self._recent.append(fps)
        if len(self._recent) > self._window:
            self._recent.pop(0)  # keep only the last ``window`` frames
        self._total_time += dt
        self._frames += 1
        self._min = fps if self._min is None else min(self._min, fps)
        self._max = fps if self._max is None else max(self._max, fps)
        return self.rolling

    @property
    def rolling(self) -> float:
        """Average FPS over the recent window (what the live HUD shows)."""
        return sum(self._recent) / len(self._recent) if self._recent else 0.0

    @property
    def average(self) -> float:
        """Average FPS over the whole run (total frames / total time)."""
        return self._frames / self._total_time if self._total_time > 0 else 0.0

    @property
    def min(self) -> float:
        """Slowest single frame's FPS (0 before any frame)."""
        return self._min or 0.0

    @property
    def max(self) -> float:
        """Fastest single frame's FPS (0 before any frame)."""
        return self._max or 0.0


@dataclass(frozen=True)
class TrackingResult:
    frames_processed: int
    avg_fps: float
    min_fps: float
    max_fps: float
    final_state: str
    output_path: Optional[str]
    lost_events: int
    reacquired_events: int
    timeline: List[TimelineEvent]
    state_frame_counts: Dict[str, int] = field(default_factory=dict)
    tracking_fps: float = 0.0

    @property
    def tracking_uptime(self) -> float:
        """Fraction of frames the target was actively held (TRACKING or REACQUIRED)."""
        if self.frames_processed <= 0:
            return 0.0
        on = self.state_frame_counts.get("TRACKING", 0) + self.state_frame_counts.get("REACQUIRED", 0)
        return on / self.frames_processed


class TrackingPipeline:
    """Tracks one target, detecting loss and re-acquiring it when it returns."""

    def __init__(self, settings: Settings, backend: Optional[str] = None) -> None:
        # An explicit backend (e.g. from --backend) overrides the configured one.
        if backend is not None:
            settings = replace(settings, tracker=replace(settings.tracker, backend=backend))
        self.settings = settings
        self.scale = settings.video.processing_scale
        self.backend = settings.tracker.backend

    def run(self, source: VideoSource, selector: TargetSelector, out_path: Optional[str] = None,
            show: bool = False, progress: Optional[ProgressCallback] = None,
            debug_dir: Optional[str] = None) -> TrackingResult:
        """Track the selected target through ``source`` to EOF and return the stats.

        Selects on the first frame, then streams the rest through the state
        machine. ``out_path`` writes an annotated video, ``show`` displays it live
        (Esc stops), ``debug_dir`` dumps per-SEARCHING-frame candidate overlays.
        """
        st = self.settings
        scale = self.scale
        # estimateAffinePartial2D's RANSAC is seeded from the global cv2 RNG; fix it
        # so a run is reproducible (the source of intermittent early-hut losses).
        cv2.setRNGSeed(0)
        if debug_dir is not None:
            import os
            os.makedirs(debug_dir, exist_ok=True)

        with source:
            meta = source.metadata
            ok, first_full = source.read()
            if not ok or first_full is None:
                raise PipelineError("Video source produced no frames")

            selection = selector.select(first_full)  # full-res coords
            init_full, hud_full = prepare_init(meta.source, first_full, selection.bbox, st.selection)

            # One working coordinate space (scaled). Default scale 1.0 = native.
            first = resize_frame(init_full, scale)
            bbox = selection.bbox.scaled(scale)
            seed_point = (int(selection.seed_point[0] * scale), int(selection.seed_point[1] * scale))
            hud_mask = self._scale_mask(hud_full, first.shape) if hud_full is not None else None
            usable_mask = cv2.bitwise_not(hud_mask) if hud_mask is not None else None
            fw, fh = first.shape[1], first.shape[0]

            # Wire the per-target components off the initial patch.
            memory = AppearanceMemory(st.verifier)
            memory.initialise(first, bbox, hud_mask)
            verifier = Verifier(st.verifier, memory)
            reacquirer = Reacquirer(st.reacquire, memory, verifier)
            detector = LossDetector(st.loss, fw, fh)
            motion = GlobalMotionEstimator(st.motion)
            tracker = create_backend(st.tracker)
            tracker.init(first, bbox, usable_mask)
            motion.update(first, bbox)  # prime the estimator with frame 0

            machine = StateMachine()
            machine.to(TrackerState.READY, 0, "target selected")
            machine.to(TrackerState.TRACKING, 0, "tracking started")

            # Re-acquisition anti-thrash state (updated in _track / _search).
            self._last_reacquire_frame: Optional[int] = None
            self._failed_reacquires = 0
            self._suspended_until = 0
            self._predicted = bbox.center

            writer = self._make_writer(out_path, fw, fh, meta.fps) if out_path else None
            meter = FpsMeter()
            state_counts: Counter = Counter()
            tracking_time, tracking_frames = 0.0, 0
            cur_bbox = bbox
            trajectory: List[Tuple[int, int]] = [(int(bbox.center[0]), int(bbox.center[1]))]
            confidence = 1.0
            reason = "tracking stable"
            frame_index = 0
            frames_processed = 1

            # Emit frame 0 (the selection) before streaming the rest.
            render_overlay(first, cur_bbox, trajectory, machine.state.value, 0.0, tracker.name,
                           confidence=confidence, seed_point=seed_point, reason=reason)
            state_counts[machine.state.value] += 1
            self._emit(first, writer, show)
            self._report(progress, frames_processed, meta.frame_count)

            try:
                for frame_full in source.frames():
                    frame = resize_frame(frame_full, scale)
                    frame_index += 1
                    start = perf_counter()
                    state_before = machine.state

                    # Ego-motion first: it feeds the tracker's scale and seeds the search.
                    transform = motion.update(frame, cur_bbox)
                    if machine.state == TrackerState.REACQUIRED:
                        machine.to(TrackerState.TRACKING, frame_index, "resumed")

                    # Dispatch to the handler for the current state (READY/LOST just coast).
                    if machine.state == TrackerState.TRACKING:
                        cur_bbox, confidence, reason = self._track(
                            frame, frame_index, tracker, verifier, memory, detector,
                            machine, transform, cur_bbox, trajectory, hud_mask, usable_mask)
                    elif machine.state == TrackerState.SEARCHING:
                        cur_bbox, confidence, reason = self._search(
                            frame, frame_index, tracker, memory, detector, reacquirer,
                            machine, transform, cur_bbox, trajectory, hud_mask, usable_mask)

                    if debug_dir is not None and state_before == TrackerState.SEARCHING:
                        self._save_debug(debug_dir, frame_index, frame, reacquirer)

                    render_overlay(frame, cur_bbox, trajectory, machine.state.value, meter.rolling,
                                   tracker.name, confidence=confidence, reason=reason)
                    dt = perf_counter() - start
                    meter.update(dt)
                    state_counts[machine.state.value] += 1
                    # Track FPS over non-SEARCHING frames separately (the real-time metric;
                    # the heavy search only runs while lost).
                    if state_before != TrackerState.SEARCHING:
                        tracking_time += dt
                        tracking_frames += 1
                    frames_processed += 1
                    if not self._emit(frame, writer, show):
                        break
                    self._report(progress, frames_processed, meta.frame_count)
            finally:
                if writer is not None:
                    writer.release()
                if show:
                    # Tear the live window down *now* (before the caller opens the
                    # temp preview). On macOS HighGUI only actually closes a window
                    # once the event loop is pumped, so waitKey a few times.
                    cv2.destroyAllWindows()
                    for _ in range(4):
                        cv2.waitKey(1)

        return TrackingResult(
            frames_processed=frames_processed, avg_fps=meter.average, min_fps=meter.min,
            max_fps=meter.max, final_state=machine.state.value, output_path=out_path,
            lost_events=machine.count(TrackerState.LOST),
            reacquired_events=machine.count(TrackerState.REACQUIRED),
            timeline=list(machine.timeline), state_frame_counts=dict(state_counts),
            tracking_fps=(tracking_frames / tracking_time) if tracking_time > 0 else 0.0,
        )

    # Per-frame handlers, one per active state.

    def _track(self, frame, idx, tracker, verifier, memory, detector, machine,
               transform, cur_bbox, trajectory, hud_mask, usable_mask):
        """One TRACKING frame: advance the tracker, fuse confidence, detect loss.

        Returns ``(bbox, confidence, reason)``. Healthy -> learn the template;
        confirmed-lost -> transition to SEARCHING (with anti-thrash bookkeeping);
        merely suspect -> hold the last good box.
        """
        tracker.set_scale_hint(transform.scale, transform.confidence)
        found, box, score = tracker.update(frame)
        if found:
            app_conf, _ = verifier.appearance_confidence(frame, box, hud_mask)
            conf = verifier.fuse_with_tracker(score, app_conf)
        else:
            box, conf = None, 0.0

        assessment = detector.assess(conf, box, cur_bbox)
        if assessment.healthy and box is not None:
            box = clamp_bbox(box, frame.shape[1], frame.shape[0])
            trajectory.append((int(box.center[0]), int(box.center[1])))
            # Gate template learning on the identity score (not the flow presence)
            # so the recent template never drifts onto arbitrary background.
            memory.update(frame, box, hud_mask, conf, tracker_score=tracker.last_score)
            return box, conf, "tracking stable"

        if assessment.confirmed_lost:
            machine.to(TrackerState.LOST, idx, assessment.reason or "lost")
            self._predicted = cur_bbox.center  # seed the search at the last known centre
            cfg = self.settings.reacquire
            # Losing the target again just after a re-lock means that re-lock was
            # probably wrong — count it as a failed attempt.
            if (self._last_reacquire_frame is not None
                    and idx - self._last_reacquire_frame <= cfg.reacquire_probation_frames):
                self._failed_reacquires += 1
            else:
                self._failed_reacquires = 0
            # After too many bad re-locks in a row, stop searching for a while so the
            # box settles into LOST instead of flickering on/off.
            if self._failed_reacquires >= cfg.max_failed_reacquires:
                self._suspended_until = idx + cfg.reacquire_cooldown_frames
                self._failed_reacquires = 0
            machine.to(TrackerState.SEARCHING, idx, "searching")
            return cur_bbox, conf, f"searching ({assessment.reason})"

        # Suspect but unconfirmed: hold the last good box (no chatter).
        return cur_bbox, conf, f"{assessment.reason}? ({assessment.consecutive_bad})"

    def _search(self, frame, idx, tracker, memory, detector, reacquirer, machine,
                transform, cur_bbox, trajectory, hud_mask, usable_mask):
        """One SEARCHING frame: run the throttled re-acquire search; re-lock on a hit.

        Returns ``(bbox, confidence, reason)``. On a confirmed find, re-init the
        tracker and transition to REACQUIRED; otherwise hold the last box (with a
        ``[cooldown]`` tag while the anti-thrash suspension is active).
        """
        cfg = self.settings.reacquire
        # Carry the prediction forward by camera motion (for the debug marker).
        if transform.confidence > 0:
            self._predicted = transform.apply_point(self._predicted)

        suspended = idx < self._suspended_until
        if not suspended and idx % cfg.reacq_every == 0:
            found = reacquirer.search(frame, hud_mask, self._predicted)
            if found is not None:
                box, conf = found
                box = clamp_bbox(box, frame.shape[1], frame.shape[0])
                tracker.reinit(frame, box, usable_mask)
                memory.update(frame, box, hud_mask, conf)
                detector.reset()
                self._last_reacquire_frame = idx
                trajectory.append((int(box.center[0]), int(box.center[1])))
                machine.to(TrackerState.REACQUIRED, idx, "reacquired")
                return box, conf, "reacquired"
        tag = " [cooldown]" if suspended else ""
        return cur_bbox, 0.0, f"searching{tag}"

    # Output, display, and debug-dump helpers.

    @staticmethod
    def _scale_mask(mask, shape):
        """Resize a HUD mask to match the working frame size (nearest-neighbour)."""
        h, w = shape[:2]
        if mask.shape[:2] == (h, w):
            return mask
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    def _save_debug(self, debug_dir, idx, frame, reacquirer) -> None:
        """Write a debug image of this frame's re-acquire candidates to disk."""
        import os
        vis = frame.copy()
        draw_debug_search(vis, None, reacquirer.last_candidates, reacquirer.last_accepted,
                          reacquirer.last_predicted)
        cv2.imwrite(os.path.join(debug_dir, f"search_{idx:05d}.png"), vis)

    def _make_writer(self, out_path, w, h, fps) -> cv2.VideoWriter:
        """Open an mp4 writer for the annotated output (falls back to 30 FPS if unknown)."""
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps if fps > 0 else 30.0, (w, h))
        if not writer.isOpened():
            raise PipelineError(f"Could not open output video for writing: {out_path}")
        return writer

    def _emit(self, frame, writer, show) -> bool:
        """Write and/or display a frame; return False if the user pressed Esc (stop)."""
        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow("tracking", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                return False
        return True

    @staticmethod
    def _report(progress, done, total) -> None:
        """Forward progress to the optional callback (no-op if none was given)."""
        if progress is not None:
            progress(done, total)
