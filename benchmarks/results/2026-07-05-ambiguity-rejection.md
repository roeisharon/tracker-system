# Phase 4 — candidate-scoring ambiguity rejection *(EVALUATED → REVERTED)*

> **Decision (2026-07-05):** implemented, measured, and **reverted.** A Lowe-style
> ratio test (refuse a re-acquisition when a spatially-distinct runner-up scores
> ≥ `0.85×` the best) provided **no measurable benefit** on the representative
> clips and perturbed the bottle behaviour without improving it. Kept as a record.

## Idea

After the identity gates pick the best re-acquisition candidate, refuse the
re-lock if a spatially-distinct runner-up (> 8% of the frame diagonal away) scored
almost as high — an unresolvable look-alike ⇒ keep SEARCHING rather than commit.

## Measured (ratio 1.0 = off → 0.85 = on)

| clip    | final | lost (off→on) | reacq (off→on) |
|---------|-------|--------------:|---------------:|
| drone   | SEARCHING (same) | 10 → 11 | 9 → 10 |
| bottle  | SEARCHING (same) | 4 → 5   | 3 → 4  |
| skydive | TRACKING (same)  | 4 → 4   | 4 → 4  |

## Why it fails to help — visual validation (not just event counts)

Frame-by-frame comparison of the bottle (target = left bottle, seed `[195,95]`),
baseline vs ambiguity:

- The left target bottle is **removed by a hand ~frame 200**; both configs then
  lose it identically (f250: both on the now-empty spot, "tracker failure").
- They diverge only mid-run — at f450 the **baseline drifts onto the middle
  bottle** ("tracking stable"), the ambiguity run drifts onto the **wall**; neither
  is on the target. By the end (f1150) both **converge to the same SEARCHING**
  state.
- The wrong-bottle transfer the reviewer objected to is a **pre-existing CSRT
  limitation already present in the committed baseline** (f450 baseline is on the
  middle bottle), *not* introduced by this change.

The gate barely fires because the pipeline already weights **spatial continuity as
primary**, so a distinct competitor almost always has a lower *weighted* score than
the best. It therefore neither helps the provided clips nor cleanly separates
identical distractors — it only perturbs timing.

## Decision — revert

Consistent with the project's discipline (adopt only measured wins) and the
reviewer's instruction ("if ambiguity rejection is neutral or harmful, revert it"),
it was removed: `reacquisition/matcher.py` back to the identity-floor version, the
`reacquire.ambiguity_ratio` config removed. The bottle returns to the committed
baseline behaviour (SEARCHING / 4 / 3). `pytest` → 108 passed.

A ratio test remains a sound idea for footage with a look-alike genuinely adjacent
to the target, but it needs a clip that actually exercises that condition to be
justified — our representative set does not.
