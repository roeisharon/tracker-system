# Phase 4 (experimental) — CSRT vs TrackerVit vs TrackerNano bake-off

**Date:** 2026-07-05 · **Method:** `benchmarks/compare_trackers.py` — the *same*
pipeline (loss detection + re-acquisition unchanged) run with three model-free
single-object trackers, identical seeds, `processing_scale = 1.0`. Deep trackers
injected behind the existing `Tracker` interface via a monkeypatched constructor;
**production code untouched.** Models fetched to a scratch dir (ViT
`vittrack.onnx` 0.72 MB; Nano `nano_backbone.onnx` 1.0 MB + `nano_head.onnx`
0.73 MB) — **not vendored**, this is an experiment only.

Both deep trackers run through **`cv2.dnn` on CPU** (`target = DNN_TARGET_CPU`) —
no PyTorch / onnxruntime. Confirmed no GPU path is used.

> `trk_fps` = compute FPS on TRACKING frames only (real-time metric). `uptime` and
> `final` are **NOT correctness** — a tracker that false-locks a cloud keeps a high
> uptime and a green box while tracking the wrong thing. The visual montages
> (`benchmarks/_bakeoff_frames/MONTAGE_*`) are the ground truth for correctness.

## Measured

| clip    | tracker | trk_fps | all_fps | uptime | 1st loss | lost | reacq | final |
|---------|---------|--------:|--------:|-------:|---------:|-----:|------:|-------|
| skydive | CSRT    |  70.9 |  71.6 | 55.3% | 0.53 s | 4  | 4  | TRACKING  |
| skydive | VIT     | 184.7 |  34.7 | 25.2% | 0.35 s | 19 | 18 | SEARCHING |
| skydive | NANO    | 123.1 |  82.7 | 85.7% | 0.52 s | 7  | 7  | TRACKING  |
| drone   | CSRT    |  67.9 |  41.6 | 59.1% | 1.29 s | 19 | 18 | SEARCHING |
| drone   | VIT     | 143.3 |  38.7 | 23.0% | 1.09 s | 17 | 16 | SEARCHING |
| drone   | NANO    | 119.2 |  40.3 | 43.2% | 1.24 s | 17 | 16 | SEARCHING |
| bottle  | CSRT    |  90.5 | 146.6 | 45.3% | 7.41 s | 4  | 3  | SEARCHING |
| bottle  | VIT     | 165.7 | 160.4 | 59.9% | 7.24 s | 3  | 2  | SEARCHING |
| bottle  | NANO    | 185.6 | 193.3 | 70.9% | 8.31 s | 5  | 5  | TRACKING  |

**FPS:** every tracker clears 30 FPS at 1080p with margin. Counter-intuitively the
deep trackers' per-update `trk_fps` exceeds CSRT's — they run a small network on a
*cropped search window*, not the whole frame. `all_fps` dips (VIT skydive 34.7)
only because frequent losses trigger many expensive SEARCHING frames; still > 30.
**Performance is not a differentiator.**

## Qualitative (from the montages — the part that matters)

**Skydive** (target shrinks to a speck, violent whip-pan, then total scene
change). All three hold the green canopy for the first ~30 frames (frame 20: all
boxes on the canopy). Then:
- The parachute becomes a few-pixel dark speck and the camera pans — **all three
  lose it**. CSRT and VIT drop to SEARCHING (honest); **NANO false-locks a patch of
  sky and reports "tracking stable"** (frame 50) — its 85.7% uptime / TRACKING is
  *deceptive*, not success.
- After ~frame 300 the clip becomes a first-person landed shot; the original green
  canopy no longer exists as an object. **No tracker can or should recover it.**
- Net: dominated by non-tracker factors (tiny target, whip-pan, target ceasing to
  exist). ViT/Nano do **not** solve it.

**Drone** (bush leaves permanently as the drone descends). CSRT holds the bush
longest (first loss 1.29 s vs ViT 1.09 s), but later shows **green "tracking
stable" on empty desert** (frame 470) — false-stable. **VIT honestly stays in
SEARCHING** there. All three correctly end SEARCHING. Marginal, and driven by
loss-detection more than the tracker.

**Bottle** (three identical bottles; the left one is picked up/occluded). Decisive
finding: **VIT confidently TRANSFERS to the wrong (right) bottle and reports
"tracking stable"** (frame 700). A stronger appearance model is *more* prone to a
confident false-identity lock on an identical distractor. **CSRT and NANO stay in
SEARCHING** (honest). Here the deep tracker is **worse**.

## Cross-cutting observations

- **ViT loses first on all three clips** (skydive 0.35 s, drone 1.09 s, bottle
  7.24 s) — the "strongest" model is the *least* sticky here, and it adds the
  false-identity-transfer risk (bottle). Poorest fit for these clips.
- **Nano** is fast and holds slightly longer on 2/3 clips, but **false-locks sky**
  when the target shrinks (skydive) — the same confident-wrong failure we are
  trying to eliminate.
- **CSRT** is predictable, competitive on initial hold (best on drone), and its
  weaknesses (false-stable on background) are a *loss-detection* problem we already
  attack in the custom layer.
- The identified failures — tiny/shrinking target, whip-pan, identical distractors,
  permanent target departure — are **re-acquisition / loss-detection /
  target-existence** problems, **not raw-tracker** problems. Swapping the tracker
  shifts the failure modes rather than removing them.

## Decision (experimental — not a production change)

**Do not replace CSRT as the default on this evidence.** No tracker is a decisive
win on the representative clips; the deep trackers shift failure modes (ViT →
confident wrong-bottle; Nano → confident sky-lock) and add a model asset, while the
FPS argument is neutral (all pass). The higher-value Phase 4 levers remain the
**custom re-acquisition + loss detection + threshold tuning**, consistent with the
approved direction.

**Caveat / what this does NOT prove:** these three clips are deliberately
adversarial. On typical footage (smooth motion, one distinct object, gradual scale
change) a deep tracker — ViT or Nano — may clearly beat CSRT. Because the deep
trackers sit behind the existing `Tracker` interface, revisiting this later (with
more/held-out clips) is a one-adapter change if future evidence justifies it.

**Outcome (2026-07-05):** decided **not** to adopt. ViT/Nano are **not** added to
the production path and the model files are **not** vendored — the classical
CSRT-based pipeline remains the default, and Phase 4 continues on loss detection,
re-acquisition, candidate scoring, confidence estimation, and threshold tuning.
The experimental harness (`benchmarks/compare_trackers.py`) was a throwaway and has
been removed; this document is retained as the decision record (mirroring the
appearance-embeddings evaluation).

Reproduce: `TRACKER_MODEL_DIR=<dir> .venv/bin/python benchmarks/compare_trackers.py`
(models: ViT & Nano ONNX, see the sourcing notes above).
