# Phase 4 — overlay-safe target initialisation (burned-in HUD)

**Date:** 2026-07-05 · **Method:** root-cause on the drone HUD, then before/after on
the representative clips. Isolated to the **selection / profile-initialisation
layer**; the tracking / re-acquisition path is untouched.

## Problem

The drone feed composites a **screen-fixed** overlay on every frame: a bright white
crosshair at pixel (960,545), diagonal guide-lines, and corner telemetry —
identical from frame 0 to frame 820 (verified). The intended target (the metal hut
the drone lands on) is a tiny speck directly under the crosshair in frame 1, so a
selection there builds the Target Profile from the **overlay**, not the object.
Empirically, selecting at the reticle gives frame-1 `sim = 1.00` (the box *is* the
crosshair); the moving desert then pulls CSRT off it, the learned identity floor
becomes `0.98` (the crosshair matches itself), and re-acquisition can never re-lock
→ dead-end SEARCHING. A reasonable user cannot select the physical hut.

## Change (`selection/overlay.py`)

A burned-in overlay is **screen-fixed while the world moves** — a video-agnostic
signal. New optional step, run once at initialisation:

1. Sample frames spread across the clip; per-pixel temporal std → **static** pixels.
2. Overlay = static **and** structural (edge/line/text in the temporal mean).
3. **Trust gate:** only proceed when a *majority* of the frame is dynamic (the
   camera is actually moving), so a static camera's real background can't
   masquerade as an overlay.
4. Only if the **selection box overlaps** the detected overlay, inpaint it out of
   the first-frame patch used to build the template and initialise the tracker.

Strict no-op otherwise. The live tracking/re-acquisition path never sees it.

## Measured

| clip / selection | overlay detected | effect |
|---|---|---|
| drone @ reticle `[545,960]` | 0.42% (crosshair + lines) | template built from ground, not crosshair — **box now follows the world** (trajectory moves off the fixed centre) instead of dead-ending on the overlay; losses 8 → 4 |
| drone @ bush `[690,1350]` | 0.42%, but box does **not** overlap it | **byte-identical** (10 lost / 9 reacq) |
| bottle `[195,95]` | **None** (static camera → 33% dynamic < 0.55 gate) | **byte-identical** (no-op) |
| skydive `[764,1402]` | **None** (100% dynamic → no static pixels) | **byte-identical** (no-op) |

Dynamic-fraction separation that drives the trust gate: drone 66%, skydive 100%
(camera moving) vs bottle 33% (static camera) → gate at 0.55.

## Decision — adopt (`handle_overlay = true`)

- **Meets the requirement:** a reasonable user can now select the physical target
  near the HUD without the profile locking onto the crosshair (manual pixel *and*
  mouse click share this init path). The tiny hut is still hard to *hold* through
  heavy ego-motion (the general small-target limit, for the ego-motion increment),
  but it is no longer tracking the overlay.
- **General, not hardcoded:** temporal static-overlay detection works for any
  burned-in overlay (HUD, timestamp, watermark, logo); nothing keys on this video.
- **Safe:** strict no-op on clean footage, static-camera footage, and selections
  away from any overlay — the three non-drone cases are byte-identical. One-time
  cost (~30 sampled frames); zero per-frame cost; `pytest` → 108 passed.

## Known limitations

- Cannot reconstruct a target *fully* occluded by the overlay in frame 1 — it
  removes the contamination so the surrounding object can be tracked, but can't
  invent hidden pixels.
- Semi-transparent lines are only partially detected (the opaque crosshair — the
  main contaminant — is caught well).
- Residual blind spot: a **static camera with a very large moving subject** could
  approach the motion gate; the robust discriminator is ego-motion estimation
  (deferred to the motion increment).
