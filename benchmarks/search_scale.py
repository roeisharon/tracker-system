#!/usr/bin/env python
"""Phase 4 · Increment 2 measurement — SEARCHING downscale (reacquire.search_scale).

Targeted comparison of the recovery-matcher downscale factor on the two
representative clips: does matching a downscaled ROI/template speed up the
lost-target (SEARCHING) periods without regressing re-acquisition? Simple script,
not a framework.

    .venv/bin/python benchmarks/search_scale.py
"""

from __future__ import annotations

import dataclasses
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

SCENARIOS = [
    ("drone-bush", "samples/sample-video.mp4", 690, 1350, 80),
    ("bottle-left", "samples/bottle-detection.mp4", 195, 95, 80),
]
SEARCH_SCALES = [1.0, 0.5]


def main() -> int:
    base = load_settings()
    for name, video, row, col, bbox in SCENARIOS:
        meta = VideoSource(os.path.join(_ROOT, video)).open().metadata
        print(f"\n### {name}  ({meta.width}x{meta.height}, {meta.frame_count} frames)")
        print(f"{'search_scale':>12} {'all_fps':>7} {'trk_fps':>7} {'uptime':>7} "
              f"{'lost':>4} {'reacq':>5} {'final':>10} {'wall(s)':>7}")
        for ss in SEARCH_SCALES:
            rq = dataclasses.replace(base.reacquire, search_scale=ss)
            pipe = TrackingPipeline(
                tracker_type=base.tracker.type,
                processing_scale=base.video.processing_scale,
                loss_config=base.loss,
                reacquire_config=rq,
            )
            t0 = time.perf_counter()
            result = pipe.run(
                VideoSource(os.path.join(_ROOT, video)),
                ManualPixelSelector(row=row, col=col, bbox_size=bbox),
                out_path=None,
            )
            wall = time.perf_counter() - t0
            print(f"{ss:>12} {result.avg_fps:>7.1f} {result.tracking_fps:>7.1f} "
                  f"{result.tracking_uptime*100:>6.1f}% {result.lost_events:>4} "
                  f"{result.reacquired_events:>5} {result.final_state:>10} {wall:>7.1f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
