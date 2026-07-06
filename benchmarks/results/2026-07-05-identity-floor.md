# Phase 4 — Increment: shared identity threshold for re-acquisition

**Date:** 2026-07-05 · **Method:** `--debug` root-cause analysis on the drone, then
before/after on the three clips via `reacquire.use_identity_appearance_floor`
(true/false). Classical, algorithmic-only; CSRT-based baseline unchanged.

## Root cause (drone false-stable / re-lock thrash)

`--debug` showed the drone's adaptive identity gate works: it learns
`idthr = 0.75` from the bush's genuine tracking, and when the tracker drifts onto
desert the appearance similarity reads `0.58–0.74 < 0.75`, so **loss detection
correctly declares loss**. But the system then **thrashes**: it re-acquires the
*same sand* within a frame or two and briefly shows `TRACKING / tracking stable`
before loss fires again — 18 such re-acquire/lose cycles.

The cause is an **inconsistent identity standard between the two stages**:

- to *keep* tracking, the loss detector requires `sim >= 0.75` (the target's own
  learned level), but
- to *re-acquire*, the matcher only requires `hist >= min_hist_score = 0.30`.

So sand (`hist ~ 0.72`) is **good enough to re-lock but not good enough to keep** —
a guaranteed re-lock/lose loop onto background, and a stream of brief false-stable
green boxes.

## Change

On the first confirmed loss, hand the detector's **learned identity threshold** to
re-acquisition as a colour-gate floor: the matcher's effective hue-sat gate becomes
`max(min_hist_score, identity_threshold)`. Re-acquisition now demands the **same
identity confidence required to keep tracking**. It degrades gracefully: when no
threshold has been learned yet (target lost early) or the target is noisy (its
learned gate is already below `min_hist_score`), the gate is unchanged. Gated by
`reacquire.use_identity_appearance_floor` (default true) for tunability.

Files: `reacquisition/matcher.py` (`find`/`identity_matches` take an
`appearance_floor`, effective gate exposed for `--debug`), `reacquisition/engine.py`
(`begin` carries the floor), `loss/detector.py` (`identity_threshold` property),
`app/pipeline.py` (passes `detector.identity_threshold` on loss),
`config/settings.py` + `configs/default.yaml` (the flag).

## Measured (floor OFF → ON)

| clip    | final (OFF→ON)     | lost | reacq (OFF→ON) | false-stable frames on background* |
|---------|--------------------|-----:|---------------:|-----------------------------------:|
| drone   | SEARCHING→SEARCHING | 19→10 | **18 → 9**    | **203 → 153 (−25%)**               |
| bottle  | SEARCHING (same)   | 4→4  | 3 → 3          | n/a (target present throughout)    |
| skydive | TRACKING (same)    | 4→4  | 4 → 4          | n/a                                |

\* TRACKING/REACQUIRED frames after the bush departs (~frame 460).

## Decision — adopt (`use_identity_appearance_floor = true`)

- **Fewer false re-acquisitions on background:** drone re-lock/lose cycles
  **halved (18 → 9)**, and false-stable-on-background frames drop **~25%**. Directly
  serves "reduce false-stable tracking" and "reduce false re-acquisition on similar
  distractors."
- **No regression:** bottle and skydive are **byte-identical** — their targets are
  noisy / lost-early, so the floor correctly falls back to `min_hist_score`. Early
  bush tracking is unchanged (the floor only affects re-acquisition, after the first
  loss). `pytest` → 102 passed.
- **Principled & lightweight:** one identity standard for both *staying* and
  *re-acquiring*, reusing the already-learned per-target distribution. No new
  signal, no new dependency, ~15 lines of gating.

## Honest limitation → next increment

This is a partial fix, not a cure. The drone still shows residual false-stable
(e.g. frame 470 is one of the remaining 9 re-locks) because desert hue-sat sits
right at the learned 0.75 threshold — the **colour histogram alone cannot reliably
separate sand from the bush**. Eliminating the residual needs a **structural signal**
(edge/gradient or template NCC) added to confidence and candidate scoring, so a
region with the right colour but no target *structure* is rejected. That is the
next increment (confidence estimation / candidate scoring), building on this one.
