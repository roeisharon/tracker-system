"""The headless tracking pipeline.

This is the reusable core the whole product is built around: it takes a video
source and a *selector*, tracks the chosen target frame by frame, detects when
tracking fails, searches for and re-acquires the target when it returns, drives
an explicit state machine, draws the overlay, and (optionally) writes an
annotated output video — all while streaming frames one at a time so memory
stays flat regardless of video length.

Design notes:

- **Selector-agnostic.** ``run`` accepts any :class:`TargetSelector`.
- **Processing/display split.** Tracking runs on a (possibly downscaled)
  *working frame*; the raw box is scaled back to full resolution for loss
  detection, matching, and drawing. Controlled by ``processing_scale``.
- **Cheap tracking, heavy recovery only when needed.** Per TRACKING frame we run
  the tracker plus the cheap :class:`LossDetector` checks. The expensive
  multi-scale search runs only while SEARCHING.
- **Explicit states.** ``INIT -> READY -> TRACKING -> LOST -> SEARCHING ->
  REACQUIRED -> TRACKING``, all recorded on a timeline.
- **Honest FPS.** The per-frame timer wraps the core work, excluding the one-time
  selection and disk encoding.
"""

from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Dict, List, Optional
import cv2
import numpy as np
from ..config.settings import LossConfig, MotionConfig, ReacquireConfig, SelectionConfig
from ..loss.detector import LossDetector
from ..metrics.fps import FpsMeter
from ..motion.ego_motion import GlobalMotionEstimator
from ..reacquisition.engine import ReacquisitionEngine
from ..reacquisition.matcher import Matcher
from ..selection.overlay import overlay_free_first_frame
from ..selection.target_selector import TargetSelector
from ..target.profile import TargetProfile
from ..tracking.opencv_tracker import OpenCVTracker
from ..utils.geometry import clamp_bbox
from ..utils.image import resize_frame
from ..video.source import VideoSource
from ..visualization.overlay import render_overlay
from .state import StateMachine, TimelineEvent, TrackerState

ProgressCallback = Callable[[int, int], None]

_REASON_STABLE = "tracking stable"
_REASON_ACQUIRED = "target selected"
_REASON_STARTED = "tracking started"
_REASON_SEARCHING = "searching"
_REASON_REACQUIRED = "reacquired"
_REASON_RESUMED = "resumed"

class PipelineError(RuntimeError):
    """Raised when the pipeline cannot run (empty video, bad output path, ...)."""


def _fmt(label: str, value) -> str:
    """Format an optional float metric for the debug line."""
    return f"{label}={value:.2f}" if value is not None else f"{label}=--"


@dataclass(frozen=True)
class TrackingResult:
    """Summary returned after a full run."""

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
    # Compute FPS of the TRACKING path only (resize + tracker update + overlay),
    # excluding the heavy SEARCHING frames. This is the real-time-relevant number:
    # "how fast does the engine track when it is on the target?"
    tracking_fps: float = 0.0

    @property
    def tracking_uptime(self) -> float:
        """Fraction of processed frames spent actively on the target.

        Counts TRACKING and REACQUIRED (the one-frame recovery flash) as "on
        target"; SEARCHING/LOST count against uptime.
        """
        if self.frames_processed <= 0:
            return 0.0
        on_target = self.state_frame_counts.get("TRACKING", 0) + self.state_frame_counts.get(
            "REACQUIRED", 0
        )
        return on_target / self.frames_processed


class TrackingPipeline:
    """Tracks one target, detecting loss and re-acquiring it when it returns."""

    def __init__(
        self,
        tracker_type: str = "CSRT",
        processing_scale: float = 1.0,
        loss_config: Optional[LossConfig] = None,
        reacquire_config: Optional[ReacquireConfig] = None,
        selection_config: Optional[SelectionConfig] = None,
        motion_config: Optional[MotionConfig] = None,
    ) -> None:
        if not 0.0 < processing_scale <= 1.0:
            raise PipelineError("processing_scale must be in the range (0, 1]")
        self.tracker_type = tracker_type
        self.processing_scale = processing_scale
        self.loss_config = loss_config or LossConfig()
        self.reacquire_config = reacquire_config or ReacquireConfig()
        self.selection_config = selection_config or SelectionConfig()
        self.motion_config = motion_config or MotionConfig()

    def run(
        self,
        source: VideoSource,
        selector: TargetSelector,
        out_path: Optional[str] = None,
        show: bool = False,
        progress: Optional[ProgressCallback] = None,
        debug: bool = False,
        debug_dir: Optional[str] = None,
    ) -> TrackingResult:
        """Run to end of stream and return a :class:`TrackingResult`."""
        scale = self.processing_scale
        self._dbg: dict = {}
        if debug_dir is not None:
            import os

            os.makedirs(debug_dir, exist_ok=True)

        with source:
            meta = source.metadata
            self._diagonal = math.hypot(meta.width, meta.height)

            ok, first = source.read()
            if not ok or first is None:
                raise PipelineError("Video source produced no frames")

            selection = selector.select(first)  # one-time, full-res coords

            # Overlay-safe initialisation: if the selection sits on a burned-in
            # screen-fixed overlay (e.g. a drone HUD crosshair), inpaint it out of
            # the first-frame patch so the template/tracker model the physical
            # object, not the overlay. No-op otherwise; live tracking is unchanged.
            init_frame = overlay_free_first_frame(
                meta.source, first, selection.bbox, self.selection_config
            )

            tracker = OpenCVTracker(self.tracker_type)
            tracker.init(resize_frame(init_frame, scale), selection.bbox.scaled(scale))
            profile = TargetProfile.create(init_frame, selection.bbox)
            detector = LossDetector(self.loss_config, meta.width, meta.height)
            engine = ReacquisitionEngine(
                self.reacquire_config, Matcher(self.reacquire_config),
                meta.width, meta.height,
            )
            motion = GlobalMotionEstimator(self.motion_config)
            # Prime the estimator with the first frame so frame 1 has a reference.
            motion.update(init_frame, selection.bbox)

            machine = StateMachine()
            machine.to(TrackerState.READY, 0, _REASON_ACQUIRED)
            machine.to(TrackerState.TRACKING, 0, _REASON_STARTED)

            # Anti-thrash bookkeeping for re-acquisition (see _track_step/_search_step).
            self._last_reacquire_frame: Optional[int] = None
            self._failed_reacquires = 0
            self._reacquire_suspended_until = 0

            writer = self._make_writer(out_path, meta) if out_path else None
            meter = FpsMeter()
            state_counts: Counter = Counter()
            tracking_time = 0.0
            tracking_frames = 0
            reason = _REASON_STABLE
            frame_index = 0
            frames_processed = 1

            render_overlay(
                first, profile, machine.state.value, 0.0, tracker.name,
                seed_point=selection.seed_point, reason=reason,
            )
            state_counts[machine.state.value] += 1
            self._emit(first, writer, show)
            self._report(progress, frames_processed, meta.frame_count)

            try:
                for frame in source.frames():
                    frame_index += 1
                    start = perf_counter()
                    self._dbg = {}
                    state_before = machine.state

                    # Estimate camera motion (previous -> this frame) up front, on
                    # the previous target box, so the transform is available to both
                    # the tracking and searching paths and accumulation never breaks.
                    transform = motion.update(frame, profile.current_bbox)

                    # A REACQUIRED flash lasts one frame, then normal tracking.
                    if machine.state == TrackerState.REACQUIRED:
                        machine.to(TrackerState.TRACKING, frame_index, _REASON_RESUMED)

                    if machine.state == TrackerState.TRACKING:
                        reason = self._track_step(
                            frame, frame_index, scale, tracker, profile,
                            detector, engine, machine, transform,
                        )
                    elif machine.state == TrackerState.SEARCHING:
                        reason = self._search_step(
                            frame, frame_index, scale, tracker, profile,
                            detector, engine, machine, transform,
                        )

                    if debug:
                        self._log_debug(frame_index, state_before, machine, reason)
                    if debug_dir is not None and state_before == TrackerState.SEARCHING:
                        self._save_debug_vis(debug_dir, frame_index, frame, engine)

                    search_region = (
                        engine.last_search_region
                        if machine.state == TrackerState.SEARCHING
                        else None
                    )
                    render_overlay(
                        frame, profile, machine.state.value, meter.rolling,
                        tracker.name, reason=reason, search_region=search_region,
                    )
                    dt = perf_counter() - start
                    meter.update(dt)
                    state_counts[machine.state.value] += 1
                    # Attribute cost to the work that ran this frame: SEARCHING
                    # frames run the heavy matcher; everything else is the (cheap)
                    # tracking path used for the real-time FPS figure.
                    if state_before != TrackerState.SEARCHING:
                        tracking_time += dt
                        tracking_frames += 1

                    frames_processed += 1
                    if not self._emit(frame, writer, show):
                        break  # user pressed ESC
                    self._report(progress, frames_processed, meta.frame_count)
            finally:
                if writer is not None:
                    writer.release()
                if show:
                    cv2.destroyAllWindows()

        return TrackingResult(
            frames_processed=frames_processed,
            avg_fps=meter.average,
            min_fps=meter.min,
            max_fps=meter.max,
            final_state=machine.state.value,
            output_path=out_path,
            lost_events=machine.count(TrackerState.LOST),
            reacquired_events=machine.count(TrackerState.REACQUIRED),
            timeline=list(machine.timeline),
            state_frame_counts=dict(state_counts),
            tracking_fps=(tracking_frames / tracking_time) if tracking_time > 0 else 0.0,
        )

    # Functional steps for each state; return a reason string for the overlay.
    def _track_step(
        self, frame, frame_index, scale, tracker, profile, detector, engine, machine,
        transform,
    ) -> str:
        """Advance one TRACKING frame; on confirmed loss, enter SEARCHING."""
        work = resize_frame(frame, scale)
        ok, box_work = tracker.update(work)
        csrt_box = box_work.scaled(1.0 / scale) if (ok and box_work is not None) else None

        # Where the camera-motion estimate says the (world-fixed) target should be.
        ego_box = (
            transform.apply_bbox(profile.current_bbox)
            if transform.confidence > 0
            else None
        )
        # Reconcile the tracker with the rigid-scene prediction: trust CSRT when it
        # agrees with the ego prediction (fine local refinement); when it fails or
        # diverges (a distractor grab / momentary drop under a hard pan), fall back
        # to the ego-predicted box and re-init the tracker on it so it recovers
        # instead of chasing the distractor. Genuine loss (occlusion, frame exit)
        # is still caught below by the appearance/overlap checks on the chosen box.
        box, tracker_ok, used_ego = self._reconcile_track_box(
            csrt_box, ego_box, work, scale, tracker
        )

        assessment = detector.assess(
            tracker_ok=tracker_ok,
            bbox=box,
            prev_bbox=profile.current_bbox,
            frame=frame,
            reference_template=profile.template,
            frame_index=frame_index,
            ego_bbox=ego_box,
        )
        self._dbg = {
            "ok": ok,
            "raw_bbox": box,
            "used_ego": used_ego,
            "metrics": dict(detector.last_metrics),
            "assessment": assessment,
        }

        if assessment.healthy and box is not None:
            box = clamp_bbox(box, frame.shape[1], frame.shape[0])
            # Position/velocity/history update. The anchor appearance template is
            # never overwritten (a slow drift would poison the identity reference);
            # only the structural MATCH template is refreshed, under guard, so it
            # keeps pace with the descending drone's zoom (see _maybe_refresh).
            profile.update(box)
            self._maybe_refresh_match_template(profile, frame, box, transform, detector)
            return _REASON_STABLE

        if assessment.confirmed_lost:
            machine.to(TrackerState.LOST, frame_index, assessment.reason or "lost")
            # Freeze the learned identity distribution: it should reflect only the
            # target's initial genuine tracking, never later drift onto background.
            detector.freeze_identity()
            # Hand the learned per-target identity threshold to re-acquisition so it
            # demands the SAME identity confidence needed to keep tracking. This
            # stops the re-lock/lose thrash onto background (e.g. the drone's sand,
            # which passes the fixed colour gate but not the target's learned level).
            floor = (
                detector.identity_threshold
                if self.reacquire_config.use_identity_appearance_floor
                else None
            )
            engine.begin(profile, appearance_floor=floor)
            # If this track collapsed shortly after a re-acquisition it did not
            # really find the target; count consecutive such failures and, past a
            # limit, suspend re-acquisition so we settle into SEARCHING rather than
            # flashing green onto a background we cannot confirm.
            cfg = self.reacquire_config
            if (
                self._last_reacquire_frame is not None
                and frame_index - self._last_reacquire_frame <= cfg.reacquire_probation_frames
            ):
                self._failed_reacquires += 1
            else:
                self._failed_reacquires = 0
            if self._failed_reacquires >= cfg.max_failed_reacquires:
                self._reacquire_suspended_until = frame_index + cfg.reacquire_cooldown_frames
                self._failed_reacquires = 0
            machine.to(TrackerState.SEARCHING, frame_index, _REASON_SEARCHING)
            return f"{_REASON_SEARCHING} ({assessment.reason})"

        # Suspect but not yet confirmed: freeze the last good box (no chatter).
        return f"{assessment.reason}? ({assessment.consecutive_bad})"

    def _search_step(
        self, frame, frame_index, scale, tracker, profile, detector, engine, machine,
        transform,
    ) -> str:
        """Advance one SEARCHING frame; on a good match, re-acquire the target."""
        found = engine.step(frame, transform)
        m = engine.matcher
        self._dbg = {
            "region": engine.last_search_region,
            "predicted": engine.last_predicted_center,
            "radius": engine.last_radius,
            "ncand": len(m.last_candidates),
            "candidates": list(m.last_candidates),
            "accepted": m.last_accepted,
            "best": m.last_best,
            "min_score": self.reacquire_config.min_score,
            "min_tmpl": self.reacquire_config.min_template_score,
            "min_hist": m.last_hist_gate,  # effective gate (raised to learned level)
            "min_motion": self.reacquire_config.min_motion_score,
        }
        # During a cooldown (after repeated failed re-acquisitions) we keep
        # SEARCHING and refuse to re-lock, so the state settles instead of
        # flashing green onto a background the identity check keeps rejecting.
        suspended = frame_index < self._reacquire_suspended_until
        if found is not None and not suspended:
            box = clamp_bbox(found, frame.shape[1], frame.shape[0])
            tracker.init(resize_frame(frame, scale), box.scaled(scale))
            # Reposition only; keep the anchor template (the match already had to
            # clear the appearance gate, but the trusted reference must persist).
            profile.update(box)
            detector.reset()
            self._last_reacquire_frame = frame_index
            machine.to(TrackerState.REACQUIRED, frame_index, _REASON_REACQUIRED)
            return _REASON_REACQUIRED
        tag = " [cooldown]" if suspended else ""
        return f"{_REASON_SEARCHING} (t={engine.frames_since_lost}){tag}"

    # -- ego-motion helpers -----------------------------------------------

    def _reconcile_track_box(self, csrt_box, ego_box, work, scale, tracker):
        """Pick the tracking box from the CSRT result and the ego prediction.

        Returns ``(box, tracker_ok, used_ego)``. CSRT is trusted when it agrees
        with the rigid-scene prediction; otherwise the ego-predicted box is used
        and the tracker re-initialised on it so it re-locks the true target.
        """
        if csrt_box is None:
            if ego_box is not None:
                self._reinit_tracker(tracker, work, ego_box, scale)
                return ego_box, True, True
            return None, False, False
        if ego_box is None:
            return csrt_box, True, False

        (ex, ey), (cx, cy) = ego_box.center, csrt_box.center
        residual = math.hypot(cx - ex, cy - ey)
        if residual <= self.loss_config.max_center_jump_frac * self._diagonal:
            return csrt_box, True, False  # agree -> trust the refined tracker
        self._reinit_tracker(tracker, work, ego_box, scale)
        return ego_box, True, True

    @staticmethod
    def _reinit_tracker(tracker, work, box, scale) -> None:
        """Re-initialise the tracker on a full-res box, in working coordinates."""
        work_h, work_w = work.shape[:2]
        wb = clamp_bbox(box.scaled(scale), work_w, work_h)
        tracker.init(work, wb)

    def _maybe_refresh_match_template(
        self, profile, frame, box, transform, detector
    ) -> None:
        """Refresh the structural match template when the target has zoomed.

        Guarded so drift cannot poison it: only from a confident ego frame, only
        when appearance is comfortably above the target's learned identity level
        (or the config floor before one is learned), and only once the accumulated
        scale drift since the last capture exceeds the configured factor.
        """
        if transform.confidence <= 0:
            return
        drift = profile.match_scale
        thr = self.motion_config.template_refresh_scale
        if not (drift >= thr or drift <= 1.0 / thr):
            return
        similarity = detector.last_metrics.get("similarity_fast")
        floor = detector.identity_threshold
        if floor is None:
            floor = self.loss_config.min_similarity
        # Only recapture when the current patch is a strong identity match, so we
        # never bake background or an occluder into the structural reference.
        if similarity is not None and similarity >= max(floor, self.loss_config.min_similarity):
            profile.refresh_match_template(frame, box)

    # -- debug logging -----------------------------------------------------

    def _log_debug(self, frame_index, state_before, machine, reason) -> None:
        """Print a one-line per-frame diagnostic (enabled by ``debug=True``)."""
        dbg = self._dbg
        parts = [f"f{frame_index:>4}", f"{state_before.value:<10}"]

        if "metrics" in dbg:  # a TRACKING frame
            m = dbg["metrics"]
            box = dbg.get("raw_bbox")
            box_s = (
                f"({box.x:.0f},{box.y:.0f},{box.w:.0f},{box.h:.0f})" if box else "None"
            )
            parts.append(f"ok={int(bool(dbg['ok']))}")
            parts.append(f"bbox={box_s}")
            parts.append(f"ov={m['overlap']:.2f}")
            parts.append(_fmt("jump", m["center_jump"]))
            parts.append(_fmt("scale", m["scale_ratio"]))
            parts.append(_fmt("sim", m.get("similarity_fast")))
            parts.append(_fmt("idctr", m.get("identity_center")))
            parts.append(_fmt("idthr", m.get("identity_threshold")))
            a = dbg["assessment"]
            parts.append(f"reason={a.reason or 'stable'}")
            parts.append(f"bad={a.consecutive_bad}")

        if "region" in dbg:  # a SEARCHING frame
            reg = dbg["region"]
            reg_s = f"{reg.w:.0f}x{reg.h:.0f}" if reg else "None"
            pc = dbg.get("predicted")
            pc_s = f"({pc[0]:.0f},{pc[1]:.0f})" if pc else "None"
            parts.append(f"search region={reg_s}@{pc_s}")
            parts.append(f"ncand={dbg['ncand']}")
            parts.append(f"thr(score={dbg['min_score']:.2f},hist={dbg['min_hist']:.2f})")
            accepted = dbg.get("accepted")
            if accepted is not None:
                cx, cy = accepted.bbox.center
                parts.append(f"ACCEPT@({cx:.0f},{cy:.0f}) score={accepted.score:.3f}")
            else:
                parts.append(f"REJECT ({self._rejection_reason(dbg)})")
            if machine.state != state_before:
                parts.append(f"-> {machine.state.value}")

            print("  ".join(parts))
            # Top-5 candidates with every sub-score and gate status.
            top = sorted(dbg["candidates"], key=lambda c: c.score, reverse=True)[:5]
            for i, c in enumerate(top):
                cx, cy = c.bbox.center
                fails = []
                if c.template_score < dbg["min_tmpl"]:
                    fails.append("TMPL")
                if c.hist_score < dbg["min_hist"]:
                    fails.append("HIST")
                if c.motion_score < dbg["min_motion"]:
                    fails.append("MOTION")
                gate = "ok" if not fails else "<gate:" + ",".join(fails)
                print(
                    f"        cand{i} @({cx:.0f},{cy:.0f}) "
                    f"tmpl={c.template_score:.3f} hist={c.hist_score:.3f} "
                    f"motion={c.motion_score:.3f} weighted={c.score:.3f} [{gate}]"
                )
            return

        if machine.state != state_before:
            parts.append(f"-> {machine.state.value}")
        print("  ".join(parts))

    def _save_debug_vis(self, debug_dir, frame_index, frame, engine) -> None:
        """Save a search-region + candidates visualization for this frame."""
        import os

        from ..visualization.overlay import draw_debug_search

        vis = frame.copy()
        pc = engine.last_predicted_center
        pc_int = (int(pc[0]), int(pc[1])) if pc else None
        draw_debug_search(
            vis, engine.last_search_region, engine.matcher.last_candidates,
            engine.matcher.last_accepted, pc_int,
        )
        cv2.imwrite(os.path.join(debug_dir, f"search_{frame_index:05d}.png"), vis)

    @staticmethod
    def _rejection_reason(dbg: dict) -> str:
        cands = dbg["candidates"]
        if not cands:
            return "no candidates generated"
        eligible = [
            c
            for c in cands
            if c.template_score >= dbg["min_tmpl"]
            and c.hist_score >= dbg["min_hist"]
            and c.motion_score >= dbg["min_motion"]
        ]
        if not eligible:
            return "no candidate passed identity gates (template + colour + spatial)"
        best = max(eligible, key=lambda c: c.score)
        return f"best weighted {best.score:.3f} < min_score {dbg['min_score']:.2f}"

    # -- I/O helpers -------------------------------------------------------

    def _make_writer(self, out_path: str, meta) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = meta.fps if meta.fps > 0 else 30.0
        writer = cv2.VideoWriter(out_path, fourcc, fps, (meta.width, meta.height))
        if not writer.isOpened():
            raise PipelineError(f"Could not open output video for writing: {out_path}")
        return writer

    def _emit(self, frame: np.ndarray, writer, show: bool) -> bool:
        """Write and/or display a frame. Returns False if the user asked to stop."""
        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow("tracking", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                return False
        return True

    @staticmethod
    def _report(progress: Optional[ProgressCallback], done: int, total: int) -> None:
        if progress is not None:
            progress(done, total)
