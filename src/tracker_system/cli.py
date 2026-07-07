"""Command-line entry point.

- ``--info <path|url>``  print video metadata + available tracker backends.
- ``track <path|url>``   select a target ([i, j] or click) and track it.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence, Tuple

import cv2

# TrackerVit on the OpenCV 5.0 DNN engine prints a harmless per-init
# "setPreferableTarget ... new graph engine" warning (it runs on CPU regardless).
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:  # pragma: no cover - older/newer cv2 without this helper
    pass

from .config import ConfigError, Settings
from .pipeline import PipelineError, TrackingPipeline
from .selection import CvClickSelector, ManualPixelSelector, SelectionError, TargetSelector
from .trackers import TrackerNotAvailableError, probe_backends
from .video import VideoSource, VideoSourceError

PROG = "tracker-system"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG, description="Real-Time Arbitrary Object Tracking & Re-acquisition System.")
    parser.add_argument("--info", metavar="SOURCE",
                        help="Print metadata + available backends for a video, then exit.")

    sub = parser.add_subparsers(dest="command")
    track = sub.add_parser("track", help="Track a selected target and write an annotated video.")
    track.add_argument("source", help="Local video file path or direct video URL.")
    group = track.add_mutually_exclusive_group(required=True)
    group.add_argument("--pixel", metavar="I,J", help="Target pixel as row,col ([i, j]).")
    group.add_argument("--select", action="store_true", help="Click the target on the first frame.")
    track.add_argument("--save", metavar="PATH", default=None, help="Annotated output video path.")
    track.add_argument("--backend", default=None,
                       help="Tracker backend: hybrid|vit|nano|csrt (overrides config).")
    track.add_argument("--bbox-size", type=int, default=None,
                       help="Initial bounding-box side length in px (overrides config).")
    track.add_argument("--show", action="store_true", help="Display the tracking window live.")
    track.add_argument("--debug-vis", metavar="DIR", default=None,
                       help="Save per-SEARCHING-frame candidate visualizations to DIR.")
    return parser


def _print_info(source: str, settings: Settings) -> int:
    try:
        with VideoSource(source) as video:
            meta = video.metadata
    except VideoSourceError as exc:
        print(f"error: {exc}")
        return 1
    print(f"Video source: {meta.source}")
    print(f"  Type:        {'direct URL' if meta.is_url else 'local file'}")
    print(f"  Resolution:  {meta.width} x {meta.height}")
    print(f"  FPS:         {meta.fps:.2f}" if meta.fps > 0 else "  FPS:         unknown")
    print(f"  Frames:      {meta.frame_count if meta.frame_count > 0 else 'unknown'}")
    print()
    backends = probe_backends(settings.tracker)
    print(f"Tracker backends (OpenCV {cv2.__version__}):")
    for name, ok in backends.items():
        print(f"  {name:<7} {'available' if ok else 'MISSING'}")
    print(f"\nDefault backend (config): {settings.tracker.backend}")
    if not backends.get(settings.tracker.backend, False):
        print("warning: the configured backend is unavailable; run download_models.py "
              "or install opencv-contrib-python.")
        return 2
    return 0


def _parse_pixel(text: str) -> Tuple[int, int]:
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("expected two comma-separated integers, e.g. 540,960")
    return int(parts[0]), int(parts[1])


def _make_progress():
    def progress(done: int, total: int) -> None:
        if total > 0 and (done % 30 == 0 or done == total):
            print(f"\r  processed {done}/{total} frames ({100.0 * done / total:4.1f}%)", end="")
        elif total <= 0 and done % 30 == 0:
            print(f"\r  processed {done} frames", end="")
    return progress


def _run_track(parser, args, settings: Settings) -> int:
    bbox_size = args.bbox_size if args.bbox_size is not None else settings.selection.default_bbox_size
    if bbox_size <= 0:
        parser.error("--bbox-size must be positive")

    selector: TargetSelector
    if args.select:
        selector = CvClickSelector(bbox_size)
    else:
        try:
            row, col = _parse_pixel(args.pixel)
        except ValueError as exc:
            parser.error(str(exc))
        selector = ManualPixelSelector(row, col, bbox_size)
        print(f"Target pixel [i={row}, j={col}] -> (x={col}, y={row}), bbox {bbox_size}px")

    pipeline = TrackingPipeline(settings, backend=args.backend)
    source = VideoSource(args.source)
    try:
        result = pipeline.run(source, selector, out_path=args.save, show=args.show,
                              progress=_make_progress(), debug_dir=args.debug_vis)
    except (VideoSourceError, PipelineError, SelectionError, TrackerNotAvailableError) as exc:
        print(f"\nerror: {exc}")
        return 1

    print("\nDone.")
    print(f"Frames processed: {result.frames_processed}")
    print(f"  Processing FPS:   avg {result.avg_fps:.1f} | min {result.min_fps:.1f} | max {result.max_fps:.1f}")
    print(f"  Tracking uptime:  {100.0 * result.tracking_uptime:.1f}%")
    print(f"  Final state:      {result.final_state}")
    print(f"  Lost / Reacquired: {result.lost_events} / {result.reacquired_events}")
    if result.timeline:
        print("  Timeline:")
        for e in result.timeline:
            print(f"    frame {e.frame_index:>5}  {e.state.value:<11} {e.reason}")
    print(f"  Output video:     {result.output_path}" if result.output_path
          else "  (no --save given; nothing written)")
    if result.avg_fps < 30.0:
        print("  note: avg FPS < 30. Lower video.processing_scale in config.py for speed.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = Settings()
    except ConfigError as exc:
        parser.error(str(exc))
    if args.info:
        return _print_info(args.info, settings)
    if args.command == "track":
        return _run_track(parser, args, settings)
    parser.print_help()
    return 0
