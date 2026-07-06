#!/usr/bin/env python
"""Targeted comparison harness for Phase 4 (Tracking Engine Optimization).

This is intentionally a **simple script, not a benchmarking framework**: it runs a
handful of realistic configurations over the representative videos and prints the
metrics that actually drive engineering decisions (compute FPS, tracking uptime,
lost / re-acquired events, final state). When a new decision comes up, edit the
``SCENARIOS`` / ``CONFIGS`` lists, run, read the table, decide, move on — do not
build generic infrastructure around it.

Usage:
    .venv/bin/python benchmarks/compare.py                 # full matrix
    .venv/bin/python benchmarks/compare.py --scenario drone-bush
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))

from tracker_system.app.pipeline import TrackingPipeline  # noqa: E402
from tracker_system.config.settings import load_settings  # noqa: E402
from tracker_system.selection.target_selector import ManualPixelSelector  # noqa: E402
from tracker_system.video.source import VideoSource  # noqa: E402

# Representative scenarios: (name, video, row, col, bbox_size).
SCENARIOS = [
    ("drone-bush", "samples/sample-video.mp4", 690, 1350, 80),
    ("bottle-left", "samples/bottle-detection.mp4", 195, 95, 80),
]

# The few realistic configurations to compare: (label, tracker_type, scale).
CONFIGS = [
    ("CSRT  @1.00", "CSRT", 1.0),
    ("KCF   @1.00", "KCF", 1.0),
    ("MOSSE @1.00", "MOSSE", 1.0),
    ("CSRT  @0.75", "CSRT", 0.75),
    ("CSRT  @0.50", "CSRT", 0.5),
]


def run_one(video, row, col, bbox, tracker, scale, settings):
    source = VideoSource(os.path.join(_ROOT, video))
    selector = ManualPixelSelector(row=row, col=col, bbox_size=bbox)
    pipeline = TrackingPipeline(
        tracker_type=tracker,
        processing_scale=scale,
        loss_config=settings.loss,
        reacquire_config=settings.reacquire,
    )
    t0 = time.perf_counter()
    result = pipeline.run(source, selector, out_path=None)
    return result, time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=None, help="Run only this scenario.")
    args = parser.parse_args()

    settings = load_settings()
    scenarios = [s for s in SCENARIOS if args.scenario in (None, s[0])]

    # NOTE: `uptime` is the fraction of frames in TRACKING/REACQUIRED. It is NOT a
    # correctness metric — a tracker that fails to notice it has drifted keeps a
    # high uptime while tracking the wrong thing. Use `trk_fps` for the real-time
    # question and pair robustness judgements with visual validation of outputs.
    header = (
        f"{'config':<12} {'trk_fps':>7} {'all_fps':>7} {'uptime':>7} "
        f"{'lost':>4} {'reacq':>5} {'final':>10} {'wall(s)':>7}"
    )
    for name, video, row, col, bbox in scenarios:
        meta = VideoSource(os.path.join(_ROOT, video)).open().metadata
        print(f"\n### {name}  —  {video}  ({meta.width}x{meta.height}, "
              f"{meta.frame_count} frames)  seed=[{row},{col}]")
        print("trk_fps = compute FPS on TRACKING frames only (real-time metric); "
              "uptime is NOT correctness.")
        print(header)
        print("-" * len(header))
        for label, tracker, scale in CONFIGS:
            try:
                result, wall = run_one(video, row, col, bbox, tracker, scale, settings)
            except Exception as exc:  # a tracker may be unavailable / fail
                print(f"{label:<12} ERROR: {exc}")
                continue
            print(
                f"{label:<12} {result.tracking_fps:>7.1f} {result.avg_fps:>7.1f} "
                f"{result.tracking_uptime*100:>6.1f}% {result.lost_events:>4} "
                f"{result.reacquired_events:>5} {result.final_state:>10} {wall:>7.1f}"
            )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
