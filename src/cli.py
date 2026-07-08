"""Command-line entry point.

- ``--info <path|url>``  print video metadata + available tracker backends.
- ``track <path|url>``   pick a target (interactive click or ``--pixel i,j``),
  track it, and preview or save the annotated video.
"""

from __future__ import annotations
import argparse
from typing import Optional, Sequence, Tuple
import cv2
from config import ConfigError, Settings
from output import AnnotatedVideoOutput
from pipeline import PipelineError, TrackingPipeline
from selection import InteractiveClickSelector, ManualPixelSelector, SelectionError, TargetSelector
from trackers import TrackerNotAvailableError, probe_backends
from video import VideoSource, VideoSourceError

# Silence OpenCV's noisy per-init DNN logging.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:  # pragma: no cover - older/newer cv2 without this helper
    pass

PROG = "tracker-system"


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI: the top-level ``--info`` flag and the ``track`` subcommand."""
    parser = argparse.ArgumentParser(
        prog=PROG, description="Real-Time Arbitrary Object Tracking & Re-acquisition System.")
    parser.add_argument("--info", metavar="SOURCE",
                        help="Print metadata + available backends for a video, then exit.")

    sub = parser.add_subparsers(dest="command")
    track = sub.add_parser("track", help="Track a target and preview / save an annotated video.")
    track.add_argument("source", help="Local video file path or direct video URL.")
    # Interactive selection is the default; --pixel bypasses it for scripted runs.
    track.add_argument("--pixel", metavar="I,J",
                       help="Target pixel as row,col ([i, j]); bypasses the interactive selector.")
    track.add_argument("--save", metavar="PATH", default=None,
                       help="Save the annotated video to PATH (else preview a temp file, then delete).")
    track.add_argument("--backend", default=None,
                       help="Tracker backend: hybrid|vit|nano|csrt (overrides config).")
    track.add_argument("--bbox", type=int, default=None,
                       help="Initial bounding-box side length in px (overrides config).")
    track.add_argument("--headless", action="store_true",
                       help="Run without displaying the live tracking window.")
    track.add_argument("--debug", metavar="DIR", default=None,
                       help="Save per-SEARCHING-frame candidate visualizations to DIR.")
    return parser


def _print_info(source: str, settings: Settings) -> int:
    """Print a source's metadata and which tracker backends this build can run."""
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
    """Parse a ``"row,col"`` string into an ``(i, j)`` integer pair."""
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("expected two comma-separated integers, e.g. 540,960")
    return int(parts[0]), int(parts[1])


def _make_progress():
    """Build a progress callback that reprints a percentage line every 30 frames."""
    def progress(done: int, total: int) -> None:
        # ``\r`` rewrites the same terminal line; total <= 0 means length is unknown.
        if total > 0 and (done % 30 == 0 or done == total):
            print(f"\r  processed {done}/{total} frames ({100.0 * done / total:4.1f}%)", end="")
        elif total <= 0 and done % 30 == 0:
            print(f"\r  processed {done} frames", end="")
    return progress


def _run_track(parser, args, settings: Settings) -> int:
    """Run the ``track`` subcommand end to end and print the run summary."""
    # Initial box size: --bbox if given, else the configured default.
    sel_cfg = settings.selection
    bbox_size = args.bbox if args.bbox is not None else sel_cfg.default_bbox_size
    if bbox_size <= 0:
        parser.error("--bbox must be positive")

    # Choose how the target is picked: scripted --pixel, else the interactive UI.
    selector: TargetSelector
    if args.pixel is not None:
        try:
            row, col = _parse_pixel(args.pixel)
        except ValueError as exc:
            parser.error(str(exc))
        selector = ManualPixelSelector(row, col, bbox_size)
        print(f"Target pixel [i={row}, j={col}] -> (x={col}, y={row}), bbox {bbox_size}px")
    else:
        selector = InteractiveClickSelector(bbox_size, sel_cfg.min_bbox_size, sel_cfg.max_bbox_size)

    show = not args.headless          # live window on unless --headless
    debug_dir = args.debug
    output = AnnotatedVideoOutput(save_path=args.save)  # decides save vs temp-preview

    pipeline = TrackingPipeline(settings, backend=args.backend)
    source = VideoSource(args.source)
    try:
        result = pipeline.run(source, selector, out_path=output.path, show=show,
                              progress=_make_progress(), debug_dir=debug_dir)
    except (VideoSourceError, PipelineError, SelectionError, TrackerNotAvailableError) as exc:
        output.cleanup()  # drop the temp file if the run failed before producing video
        print(f"\nerror: {exc}")
        return 1

    # Report the run's statistics and state timeline.
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
    if output.is_temporary:
        print("  Output video:     (temporary preview — opening in your video player)")
    else:
        print(f"  Output video:     {result.output_path}")

    output.finish(success=True)  # preview the temp result (if any), then clean up
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: parse args, load config, dispatch to --info or track."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = Settings()  # validated on construction; bad config aborts here
    except ConfigError as exc:
        parser.error(str(exc))
    if args.info:
        return _print_info(args.info, settings)
    if args.command == "track":
        return _run_track(parser, args, settings)
    parser.print_help()
    return 0
