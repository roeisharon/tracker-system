# Phase 4 — Increment 1: Tracker & processing-scale decision

**Date:** 2026-07-04 · **Method:** `benchmarks/compare.py` on the two representative
clips. Targeted comparison of the realistic alternatives — not an exhaustive sweep.

## Measured (drone clip, 1920×1080, seed [690,1350])

`trk_fps` = compute FPS on TRACKING frames only (the real-time metric).
`all_fps` = average over all frames incl. the heavy SEARCHING frames.
`uptime` is **not** a correctness metric (a tracker that fails to notice it drifted
keeps a high uptime while tracking the wrong thing).

| config      | trk_fps | all_fps | uptime | lost | reacq | final     |
|-------------|--------:|--------:|-------:|-----:|------:|-----------|
| CSRT  @1.00 |  **58.8** |    13.0 |  45.9% |   15 |    14 | SEARCHING |
| KCF   @1.00 |   191.3 |     9.4 |  24.2% |   24 |    23 | SEARCHING |
| MOSSE @1.00 |   439.3 |    10.2 |  32.1% |   24 |    24 | TRACKING  |
| CSRT  @0.75 |    54.9 |    10.3 |  29.3% |   17 |    16 | SEARCHING |
| CSRT  @0.50 |    87.3 |    11.4 |  33.3% |   23 |    22 | SEARCHING |

## Bottle clip (640×360, seed [195,95]) — for reference

| config      | trk_fps* | uptime | lost | reacq | final     |
|-------------|---------:|-------:|-----:|------:|-----------|
| CSRT  @1.00 |   ~120 |  44.7% |    3 |     2 | SEARCHING |
| KCF   @1.00 |   ~149 |  75.2% |   10 |    10 | TRACKING  |
| MOSSE @1.00 |   ~539 |  80.3% |    6 |     5 | SEARCHING |

\* first-pass `all_fps` figures; the small clip is not FPS-constrained.

## Decision

**Tracker = CSRT, processing_scale = 1.0** (confirms the existing default, now
backed by measurement):

- **Real-time is met with margin:** CSRT tracks at **58.8 FPS at 1080p**, ~2× the
  30 FPS requirement. There is therefore **no real-time pressure** to adopt the
  faster KCF/MOSSE.
- **CSRT is the most stable** on the scale-changing drone footage (fewest losses,
  15 vs 24). CSRT is scale-adaptive; KCF/MOSSE use a fixed box and cannot follow
  the descent's dramatic scale change. MOSSE's "final: TRACKING" and high bottle
  uptime are the fixed-scale tracker holding *a* box without noticing failure —
  i.e. not correctness.
- **Downscaling hurts:** CSRT @0.5/0.75 lowers uptime (45.9% → 29–33%) on the
  small 1080p target without a needed FPS gain, so scale stays at 1.0.

## Finding → next increment

`all_fps` collapses to ~13 FPS because **SEARCHING is expensive**: multi-scale
template matching runs over a large region of the *full-resolution* 1080p frame
every SEARCHING frame. When the target is lost often, overall throughput drops
below 30 FPS. **Next optimization target: make SEARCHING cheaper** (e.g. run the
matcher on a downscaled working frame / tighten the search region), measured the
same way.
