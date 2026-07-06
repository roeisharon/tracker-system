# Phase 4 — structural-confidence signal *(EVALUATED → REJECTED)*

> **Decision (2026-07-05):** implemented, measured, and **reverted.** A grayscale
> NCC "structure" term blended into the appearance/identity signal **degraded all
> three representative clips**. The colour histogram's scale/rotation invariance —
> the very reason it was chosen — is what a same-size NCC sacrifices. Kept as a
> record; the identity-floor increment (`2026-07-05-identity-floor.md`) remains.

## Idea

To attack the residual drone false-stable (sand has the bush's colour but not its
structure), add a structural signal — `structural_similarity`: resize the patch to
the template size and take `TM_CCOEFF_NORMED` on grayscale — and blend it with the
colour histogram, `identity = (1-w)*colour + w*structure`, `w = 0.4`. The blend
would feed both the loss detector's learned per-target distribution and the
re-acquisition identity floor.

## Measured (structural_weight 0.0 → 0.4)

| clip    | final (0.0→0.4)      | lost (0.0→0.4) | reacq (0.0→0.4) | false-stable frames after departure |
|---------|----------------------|---------------:|----------------:|------------------------------------:|
| drone   | SEARCHING → **TRACKING** | 10 → 11    | 9 → 11          | **153 → 292 (worse)**               |
| bottle  | SEARCHING (same)     | 4 → 6          | 3 → 5           | — (more thrash)                     |
| skydive | TRACKING (same)      | 4 → 11         | 4 → 11          | — (more thrash)                     |

Every clip got worse; the drone — the target this was meant to help — regressed the
most and flipped to a **false** `TRACKING` on background.

## Why it fails (measured)

Grayscale NCC of the tracked patch vs the anchor template, sampled while genuinely
tracking the drone bush:

| frame (genuine bush) | colour | structure |
|---|---:|---:|
| 30  | 0.93 | 0.97 |
| 60  | 0.84 | 0.52 |
| 300 | 0.90 | 0.41 |
| 430 | 0.93 | **0.26** |
| (sand 500–700) | 0.05–0.21 | **0.09–0.24** |

**Structure is noisy on the *genuine* target** (it falls to 0.26–0.52 as the bush
scales/rotates/deforms with the descending camera) and its genuine range
**overlaps sand** (0.09–0.24). Colour, by contrast, stays cleanly high on the
genuine bush (0.79–0.93). Blending a noisy signal into the **adaptive** identity
distribution *widens* it (larger MAD) → the learned gate becomes **more permissive**
→ **more** false-stable, not less. A same-size NCC throws away the scale/rotation
invariance that makes the histogram reliable for arbitrary, deforming targets.

## Decision — reject; keep colour-only + identity floor

- **Net negative on all three clips**, worst on the intended target. No weight
  setting rescued it (lower weights only shrink the harm).
- The residual drone false-stable is a real limit of the colour signal, but
  **structure via same-size NCC is not a usable substitute** for these deforming
  targets. A more robust structural formulation (multi-scale / local-search NCC,
  as re-acquisition's `template_score` already uses) could be revisited, but it is
  out of scope for a focused increment and unlikely to separate sand from a bush
  whose own structure reads 0.26.
- Reverted cleanly (`utils/appearance.py` back to colour-only; no config/API
  residue). `pytest` → 102 passed; the identity-floor increment is unchanged
  (drone SEARCHING / 10 / 9).
