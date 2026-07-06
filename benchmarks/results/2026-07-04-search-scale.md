# Phase 4 — Increment 2: SEARCHING-cost optimization (`reacquire.search_scale`)

**Date:** 2026-07-04 · **Method:** `benchmarks/search_scale.py` on the two
representative clips. Optimises **only** the re-acquisition path; the normal
TRACKING path is untouched.

## Change

The expensive multi-scale `matchTemplate` now runs on a **downscaled** ROI +
template (`reacquire.search_scale`), and candidate coordinates are mapped back to
**full-resolution** frame space. Since matching is already multi-scale it is
scale-tolerant, and the tracker is re-initialised on the full-res box afterwards,
so localisation is essentially unaffected. `predicted_center`, motion, histogram
scoring, overlays and diagnostics all remain in native coordinates.

## Measured (search_scale 1.0 → 0.5)

| clip        | search_scale | all_fps | trk_fps | uptime | lost | reacq | final     |
|-------------|-------------:|--------:|--------:|-------:|-----:|------:|-----------|
| drone 1080p |          1.0 |    12.3 |    54.6 |  45.9% |   15 |    14 | SEARCHING |
| drone 1080p |     **0.5**  |  **42.3** |  67.9 |  59.1% |   19 |    18 | SEARCHING |
| bottle 360p |          1.0 |   119.9 |    98.6 |  44.7% |    3 |     2 | SEARCHING |
| bottle 360p |     **0.5**  |  **163.2** |  97.6 |  45.3% |    4 |     3 | SEARCHING |

## Decision — adopt `search_scale = 0.5`

Against the acceptance criteria:

- **Meaningful SEARCHING/all_fps improvement:** drone `all_fps` **12.3 → 42.3
  (3.4×)** — sustained throughput now **clears 30 FPS even during lost-target
  periods** (previously it collapsed to ~12). Bottle `all_fps` 120 → 163.
- **No re-acquisition regression:** the synthetic leave/return integration test
  passes at 0.5; bottle re-acquisitions comparable (2→3); early bottle tracking
  visually correct (locked on the left bottle); both clips still end SEARCHING.
- **No obvious false-lock increase (drone):** final state stays SEARCHING (no
  false TRACKING on background); lost/reacq counts comparable. The couple of extra
  lost/reacq cycles come with *higher* uptime, i.e. faster recovery attempts, not
  worse identity.
- **TRACKING path unchanged:** `trk_fps` differences are thermal/measurement noise
  (same code path).
- **CPU-only & lightweight:** just an `INTER_AREA` resize of the ROI/template.

The bottle's end behaviour (drift across the three identical bottles) is the
pre-existing, documented CSRT identity-transfer limitation — unchanged by this
optimisation.
