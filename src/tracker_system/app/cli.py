"""Command-line entry point.

Two commands:

- ``--info <path|url>`` — Prints video metadata and which trackers are available, then exit.
- ``track <path|url>`` — Select a target (manual ``[i, j]`` or mouse click), track it through the video, and write an annotated output video.

Both selection methods build a :class:`TargetSelector` and hand it to the same
:class:`TrackingPipeline`, so they share one identical tracking path.
"""

from __future__ import annotations
import argparse
from typing import Optional, Sequence, Tuple
import cv2
from ..config.settings import ConfigError, Settings, load_settings
from ..selection.cv_click_selector import CvClickSelector
from ..selection.target_selector import ManualPixelSelector, SelectionError, TargetSelector
from ..tracking.factory import TrackerNotAvailableError, probe_trackers
from ..video.source import VideoSource, VideoSourceError
from .pipeline import PipelineError, TrackingPipeline

PROG = "tracker-system"

def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Real-Time Arbitrary Object Tracking & Re-acquisition System."
    )
    parser.add_argument("--info",
        metavar="SOURCE",
        help="Print metadata for a video file path or direct URL, plus the available trackers, then exit."
    )
    parser.add_argument("--config",
        metavar="PATH",
        default=None,
        help="Optional YAML config file (defaults to configs/default.yaml)."
    )

    sub = parser.add_subparsers(dest="command")

    track = sub.add_parser("track",
        help="Track a selected target and write an annotated output video.",
        description="Track a target selected by pixel [i, j] or by mouse click."
    )
    track.add_argument("source", help="Local video file path or direct video URL.")
    group = track.add_mutually_exclusive_group(required=True) # Must choose either --pixel or --select
    group.add_argument("--pixel",
        metavar="I,J",
        help="Target pixel as row,col ([i, j] per the assignment)."
    )
    group.add_argument("--select",
        action="store_true",
        help="Pick the target by clicking it on the first frame."
    )

    track.add_argument("--out", 
        metavar="PATH", 
        default=None, 
        help="Annotated output video path.")
    track.add_argument("--bbox-size",
        type=int,
        default=None,
        help="Initial bounding-box side length in px (overrides config)."
    )
    track.add_argument("--scale",
        type=float,
        default=None,
        help="Processing scale in (0, 1] for the working frame (overrides config)."
    )
    track.add_argument("--show", 
        action="store_true", 
        help="Display the tracking window live."
    )
    track.add_argument("--debug",
        action="store_true",
        help="Print per-frame diagnostics (state, loss metrics, search/candidates).",
    )
    track.add_argument("--debug-vis",
        metavar="DIR",
        default=None,
        help="Save per-SEARCHING-frame candidate visualizations to DIR.",
    )
    # Allow --config after the subcommand too, without clobbering a top-level one.
    track.add_argument("--config", metavar="PATH", default=argparse.SUPPRESS)

    return parser

# --info command
# Format a duration in seconds as a string, or "unknown" if <= 0.
def _format_duration(seconds: float) -> str:
    return f"{seconds:.2f} s" if seconds > 0 else "unknown"

# Print video metadata and tracker availability, returning an exit code.
def _print_info(source: str, settings: Settings) -> int:
    try:
        with VideoSource(source) as video:
            meta = video.metadata
    except VideoSourceError as exc:
        print(f"error: {exc}")
        return 1

    frame_count = f"{meta.frame_count}" if meta.frame_count > 0 else "unknown"
    fps = f"{meta.fps:.2f}" if meta.fps > 0 else "unknown"

    print(f"Video source: {meta.source}")
    print(f"  Type:        {'direct URL' if meta.is_url else 'local file'}")
    print(f"  Resolution:  {meta.width} x {meta.height}")
    print(f"  FPS:         {fps}")
    print(f"  Frames:      {frame_count}")
    print(f"  Duration:    {_format_duration(meta.duration_seconds)}")
    print()

    trackers = probe_trackers()
    print(f"Tracker availability (OpenCV {cv2.__version__}):")
    width = max(len(name) for name in trackers)
    for name, ok in trackers.items():
        status = "available" if ok else "MISSING"
        print(f"  {name:<{width}}  {status}")
    print()
    print(f"Default tracker (config): {settings.tracker.type}")

    if not trackers.get("CSRT", False):
        print()
        print(
            "warning: CSRT is not available. Install 'opencv-contrib-python' "
            "(not 'opencv-python-headless') so scale-adaptive tracking works."
        )
        return 2
    return 0


# track command
# Parse a pixel string "row,col" into a tuple of integers, or raise ValueError.
def _parse_pixel(text: str) -> Tuple[int, int]:
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("expected two comma-separated integers, e.g. 540,960")
    try:
        row, col = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"invalid --pixel value {text!r}: {exc}") from exc
    return row, col

# Make a progress callback function that prints progress every 30 frames or at the end.
def _make_progress():
    def progress(done: int, total: int) -> None:
        if total > 0:
            if done % 30 == 0 or done == total:
                pct = 100.0 * done / total
                print(f"\r  processed {done}/{total} frames ({pct:4.1f}%)", end="")
        elif done % 30 == 0:
            print(f"\r  processed {done} frames", end="")

    return progress

# Run the tracking pipeline with the given arguments and settings, returning an exit code.
def _run_track(parser: argparse.ArgumentParser, args, settings: Settings) -> int:
    bbox_size = args.bbox_size if args.bbox_size is not None else settings.selection.default_bbox_size
    if bbox_size <= 0:
        parser.error("--bbox-size must be positive")

    scale = args.scale if args.scale is not None else settings.video.processing_scale
    if not 0.0 < scale <= 1.0:
        parser.error("--scale must be in the range (0, 1]")

    selector: TargetSelector
    if args.select:
        selector = CvClickSelector(bbox_size) #mouse click selection
    else:
        try:
            row, col = _parse_pixel(args.pixel) #manual pixel selection
        except ValueError as exc:
            parser.error(str(exc))
        selector = ManualPixelSelector(row, col, bbox_size)
        print(f"Target pixel [i={row}, j={col}] -> (x={col}, y={row}), bbox {bbox_size}px")

    # Build the tracking pipeline and run it on the video source with the selector.
    pipeline = TrackingPipeline(
        tracker_type=settings.tracker.type,
        processing_scale=scale,
        loss_config=settings.loss,
        reacquire_config=settings.reacquire,
        selection_config=settings.selection,
        motion_config=settings.motion,
    )
    source = VideoSource(args.source)

    try: # Run the tracking pipeline and handle any errors that occur.
        result = pipeline.run(
            source,
            selector,
            out_path=args.out,
            show=args.show,
            progress=None if args.debug else _make_progress(),
            debug=args.debug,
            debug_dir=args.debug_vis,
        )
    except (VideoSourceError, PipelineError, SelectionError, TrackerNotAvailableError) as exc:
        print(f"\nerror: {exc}")
        return 1

    print()  # end the progress line
    print("Done.")
    print(f"Frames processed: {result.frames_processed}")
    print(
        f"  Processing FPS:   avg {result.avg_fps:.1f} | "
        f"min {result.min_fps:.1f} | max {result.max_fps:.1f}"
    )
    print(f"  Final state:      {result.final_state}")
    print(f"  Lost events:      {result.lost_events}")
    print(f"  Reacquired:       {result.reacquired_events}")
    if result.timeline:
        print("  Timeline:")
        for event in result.timeline:
            print(f"    frame {event.frame_index:>5}  {event.state.value:<11} {event.reason}")
    if result.output_path:
        print(f"  Output video:     {result.output_path}")
    else:
        print("  (no --out given; nothing was written to disk)")

    if result.avg_fps < 30.0:
        print(
            "  note: average processing FPS < 30. Lower --scale (e.g. 0.5) to "
            "trade a little accuracy for speed at 1080p."
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = getattr(args, "config", None)
    try:
        settings = load_settings(config_path)
    except ConfigError as exc:
        parser.error(str(exc))  # exits with code 2

    if args.info:
        return _print_info(args.info, settings)

    if args.command == "track":
        return _run_track(parser, args, settings)

    parser.print_help()
    return 0
