# Real-Time Arbitrary Object Tracking & Re-acquisition System

A lightweight, **CPU-first** application for real-time tracking of an arbitrary
user-selected object in a video, with automatic loss detection and
re-acquisition when the target returns to the frame.

The target is any visually distinct region chosen by a pixel `[i, j]` on the
first frame — not a predefined object class — so recovery is appearance-based
rather than reliant on a semantic detector such as YOLO.

> **Status:** Phase 3 (Appearance-Based Re-acquisition). The project is built one
> phase at a time following `approved-implementation-roadmap.md`. The performance
> report and the desktop UI arrive in later phases.

---

## Requirements

- Python 3.10+ (developed and verified on 3.13)
- `opencv-contrib-python` — **not** `opencv-python-headless`. The contrib build
  provides the classical trackers (CSRT/KCF/MOSSE) and HighGUI windows the
  system depends on.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For development (tests included):

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Usage (Phase 0)

Print video metadata and the available trackers for the current environment:

```bash
python main.py --info sample-video.mp4
```

Example output:

```text
Video source: sample-video.mp4
  Type:        local file
  Resolution:  1920 x 1080
  FPS:         59.80
  Frames:      855
  Duration:    14.30 s

Tracker availability (OpenCV 4.13.0):
  CSRT   available
  KCF    available
  MOSSE  available

Default tracker (config): CSRT
```

A direct video URL works the same way:

```bash
python main.py --info https://example.com/clip.mp4
```

Use an alternative configuration file with `--config`:

```bash
python main.py --info sample-video.mp4 --config configs/default.yaml
```

## Usage (Phase 1) — tracking

Track a target selected by pixel and write an annotated output video:

```bash
python main.py track sample-video.mp4 --pixel 690,1350 --out out.mp4
```

`--pixel` takes `I,J` = **row,col** (`[i, j]`) per the assignment; internally
this maps to `(x=col, y=row)`. So `690,1350` selects the pixel at row 690,
column 1350.

Or select the target by clicking it on the first frame:

```bash
python main.py track sample-video.mp4 --select --out out.mp4
```

Both selection methods flow through the **same** tracking pipeline. Useful flags:

| Flag | Meaning |
| --- | --- |
| `--out PATH` | Write an annotated MP4. Omit to run headless (e.g. to benchmark FPS). |
| `--show` | Display the tracking window live (ESC to stop). |
| `--bbox-size N` | Initial box side length in px (overrides config). |
| `--scale S` | Track on a downscaled working frame, `S` in `(0, 1]`, for higher FPS. |

The run prints a summary including average/min/max **processing FPS**. On the
sample video at native 1080p with CSRT this is well above the 30 FPS target.

## Configuration

Defaults live in [`configs/default.yaml`](configs/default.yaml). Values are
validated on load; unknown keys or out-of-range values are rejected with a clear
error. Phase 0 recognises:

| Key                          | Meaning                                             | Default |
| ---------------------------- | --------------------------------------------------- | ------- |
| `video.processing_scale`     | Working-frame downscale factor for tracking, `(0,1]`| `1.0`   |
| `video.url_open_timeout_ms`  | Open timeout hint for URL sources                   | `10000` |
| `selection.default_bbox_size`| Side length (px) of the initial bounding box        | `80`    |
| `tracker.type`               | Default tracker: `CSRT` \| `KCF` \| `MOSSE`         | `CSRT`  |
| `loss.max_lost_frames`       | Consecutive bad frames before LOST is declared      | `5`     |
| `loss.max_center_jump_frac`  | Max centre jump per frame (fraction of diagonal)    | `0.25`  |
| `loss.max_scale_ratio`       | Max per-frame box area growth/shrink ratio          | `1.6`   |
| `loss.min_frame_overlap`     | Min fraction of the box that must stay in-frame     | `0.3`   |
| `loss.similarity_ema_alpha`  | Fast-EMA responsiveness of the appearance signal    | `0.3`   |
| `loss.similarity_window`     | Rolling-median baseline window (frames)             | `30`    |
| `loss.similarity_drop_ratio` | Relative-drop trigger vs the appearance baseline    | `0.6`   |
| `loss.min_similarity`        | Absolute appearance safety floor                    | `0.15`  |
| `reacquire.min_score`        | Min weighted score to accept a re-lock              | `0.45`  |
| `reacquire.min_template_score` | Identity gate: min template (structure) match     | `0.3`   |
| `reacquire.min_hist_score`   | Identity gate: min colour-histogram match           | `0.3`   |
| `reacquire.min_motion_score` | Identity gate: min spatial proximity to prediction  | `0.3`   |
| `reacquire.search_radius_frac` | Initial search radius (fraction of frame diagonal) | `0.15` |
| `reacquire.search_expansion_frac` | Search-radius growth per lost frame        | `0.03` |
| `reacquire.max_search_radius_frac` | Search-radius cap (fraction of diagonal)   | `0.75` |
| `reacquire.max_region_frac`  | Hard cap on region size (fraction of each frame dim)| `0.6`   |
| `reacquire.scales`           | Template scales tried during matching               | `[0.8, 1.0, 1.25]` |
| `reacquire.weight_template` / `weight_histogram` / `weight_motion` | Scoring weights (spatial primary) | `0.3 / 0.2 / 0.5` |
| `reacquire.motion_sigma_frac`| Motion-prior locality (fraction of diagonal)        | `0.15`  |

When tracking fails (target leaves the frame, tracker destabilises, appearance
drifts), the state transitions `TRACKING → LOST → SEARCHING`. Loss uses a
*relative* appearance drop (vs a rolling baseline) so a noisy/translucent target
is not falsely lost. While SEARCHING the system scans a **bounded, motion-anchored**
region and performs **identity matching**: a candidate must pass appearance
(template + colour) *and* spatial (motion) gates — so it re-locks the true target,
not an identical look-alike elsewhere. On a confident match it transitions
`REACQUIRED → TRACKING`. `--debug` prints per-frame diagnostics (including the
top-5 candidates and every sub-score); `--debug-vis DIR` saves search-region
visualisations. All state changes appear in the overlay HUD and the run timeline.

## Tests

```bash
pytest
```

## Project layout (through Phase 1)

```text
tracker-system/
  main.py                    # thin entry point -> package CLI
  requirements.txt           # runtime deps (opencv-contrib-python, numpy, PyYAML)
  requirements-dev.txt       # + pytest
  pyproject.toml
  configs/
    default.yaml             # validated default configuration
  src/tracker_system/
    app/
      cli.py                 # `--info` and `track` commands
      pipeline.py            # streaming tracking loop (selector-agnostic core)
    config/settings.py       # typed, validated settings loader
    video/source.py          # streaming VideoSource + metadata
    tracking/
      base.py                # Tracker interface
      opencv_tracker.py      # CSRT/KCF/MOSSE adapter
      factory.py             # tracker capability probe + factory
    selection/
      target_selector.py     # interface + manual [i, j] selector
      cv_click_selector.py    # mouse-click selector
    target/profile.py        # TargetProfile (bbox, template, velocity, history)
    visualization/overlay.py # bbox / trajectory / HUD drawing
    metrics/fps.py           # FPS meter
    utils/
      geometry.py            # BBox + clamp/scale/patch helpers
      image.py               # resize (processing/display split)
  tests/
    unit/                    # config, video, factory, geometry, selection, profile, fps
    integration/             # end-to-end pipeline run
```

Directories for later phases (loss detection, re-acquisition, UI) are created
only when their phase begins, per the roadmap.

## Design references

- `project-overview.md` — architecture and product vision.
- `approved-implementation-roadmap.md` — the implementation contract.
- `TASK.md` — the assignment requirements.
