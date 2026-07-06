# Phase 4 — appearance embeddings for identity matching *(EVALUATED → DEFERRED)*

> **Decision (2026-07-04):** the embedding integration was **implemented, measured,
> and then removed.** On the representative clips the *measured* end-to-end benefit
> was modest (drone re-acquisitions 18→16; bottle unchanged) while it added an
> onnxruntime + model dependency and ~18% SEARCHING cost. We chose to prioritise
> classical algorithmic improvements first. The strong per-patch discrimination it
> showed (below) is a good argument to achieve the same effect *classically* (e.g.
> a gradient/edge structural signal). Kept as a record; may be revisited if the
> classical avenues are exhausted.

---


**Date:** 2026-07-04 · **Method:** `benchmarks/embeddings.py` (+ a clean 2-run
drone measurement) on the representative clips.

## Change

A small **CPU ONNX embedding** (MobileNetV2 features, exported by
`models/export_embedder.py`) provides a learned appearance signal. During
**SEARCHING only**, and only for the few candidates that already pass the cheap
template/colour/spatial gates, each candidate patch is embedded (batched) and its
**cosine to the target's embedding** is (a) folded into the weighted score
(`weight_embedding`) and (b) used as a hard gate (`min_embedding_score`). The
TRACKING path is untouched; the composition root injects the embedder, and the
whole thing degrades to the classical pipeline if onnxruntime / the model is
absent.

## Why it helps — embedding discrimination on real patches

| pair | cosine |
|---|---|
| drone: bush ~ bush (target self) | **0.94** |
| drone: bush ~ sand / terrain | **0.43 – 0.53** |
| bottle: left ~ wall (background) | **0.21** |
| bottle: left ~ middle / right (identical bottles) | 0.91 / 0.95 |

The histogram rated the drone sand ~0.72 (indistinguishable); the embedding rates
it ~0.45 — clean separation. Identical bottles correctly stay high (embeddings
cannot and should not separate identical objects — the spatial gate does that).
Gate `0.6` admits the true target (~0.9+) and rejects background (< 0.55).

## Measured (embeddings OFF → ON, gate 0.6)

| clip        | metric         | OFF   | ON    |
|-------------|----------------|------:|------:|
| drone 1080p | all_fps        | 39.7  | 32.6  |
| drone 1080p | trk_fps        | 66.2  | 68.7  |
| drone 1080p | re-acquisitions| 18    | 16    |
| bottle 360p | re-acquisitions| 3     | 2–3   |

## Decision — adopt (`use_embeddings=true`, gate 0.6, weight 0.4)

Against the acceptance criteria:

- **Recovery correctness improved:** the embedding cleanly separates the target
  from similar-but-different background (bush 0.94 vs sand 0.45), rejects those
  candidates, and steers SEARCHING toward structurally-similar regions; drone
  re-acquisitions drop and it stays SEARCHING on background (no false green box).
- **No re-acquisition regression:** synthetic leave/return test passes; bottle
  re-acquisitions comparable; early bottle tracking still correct.
- **No false-lock increase:** drone final state stays SEARCHING; fewer reacquires.
- **CPU-only & SEARCHING-only:** onnxruntime, batched over the few gated
  candidates; **TRACKING path unchanged** (`trk_fps` steady at ~66–69).
- **Additional signal, not a replacement:** it augments the existing
  template/histogram/motion pipeline (extra weighted term + optional gate).
- **Runtime impact:** ~18% on the drone's *SEARCHING* throughput (39.7 → 32.6
  all_fps) — **still above the 30 FPS real-time bar**; zero cost during TRACKING.

Note: the end-to-end gain on these two clips is modest because the drone target
never returns (any reacquire is false; the count is a weak proxy) and the bottle
distractors are identical (appearance cannot separate them). The embedding's
value is the structural discrimination it adds for the general
"similar-but-different distractor" case, with graceful degradation when the model
is unavailable.
