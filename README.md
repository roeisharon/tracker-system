# Real-Time Arbitrary Object Tracking & Re-acquisition System

A **CPU-first** application that tracks an arbitrary user-selected object through
a video from a single seed pixel `[i, j]`, with automatic **loss detection** and
**re-acquisition** when the target returns. Class-agnostic (no YOLO / semantic
detector) and real-time on 1920×1080 footage.

## Approach

The evaluation clip is a drone descending onto a target that grows from an ~80 px
speck to filling the frame (a ~15× scale change). Fixed-scale trackers fail this:
CSRT freezes its box while a raw ViT balloons to the whole frame. The design
therefore **decouples the three signals**:

- **Centre** — a deep Siamese tracker (OpenCV `TrackerVit`, ONNX via `cv2.dnn`,
  CPU) when it is confident; otherwise the local flow centre (the target is too
  small/low-texture for the ViT to lock early in the descent).
- **Scale** — a local optical-flow **similarity transform** (LK + RANSAC on
  features inside the box) gives the true per-frame zoom, damped by the global
  ego-motion scale. This is what makes the box grow with the target.
- **Identity** — an independent appearance **verifier** (grayscale NCC + HSV
  histogram + ORB/RANSAC inliers) over an **appearance memory** (an immutable
  anchor + a confidence-gated EMA "recent" template). Its confidence is fused
  with the tracker score to drive loss detection and to confirm re-acquisition.

Loss is declared by hysteresis on the fused confidence; while lost, a full-frame
multi-scale search is **appearance-confirmed** (stricter threshold) rather than
snapping to a predicted point. A burned-in HUD overlay (crosshair/telemetry) is
detected and excluded from all matching.

## Requirements

- Python 3.10+
- `opencv-contrib-python` (**not** `opencv-python-headless`) — provides
  `TrackerVit`/`TrackerNano`/CSRT and HighGUI.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_models.py            # fetch the ViT ONNX (~0.7 MB) if not vendored
```

## Usage

```bash
# environment + backend availability
python main.py --info sample-video.mp4

# track a pixel [i, j] = row,col and write an annotated video
python main.py track sample-video.mp4 --pixel 545,960 --out out.mp4

# or click the target on the first frame
python main.py track sample-video.mp4 --select --out out.mp4
```

Useful flags: `--backend hybrid|vit|nano|csrt` (override the default),
`--show` (live window), `--bbox-size N`, `--debug-vis DIR` (dump search frames
while re-acquiring). The run prints uptime, lost/reacquired counts, the state
timeline, and average/min/max processing FPS.

### Benchmark

```bash
.venv/bin/python benchmarks/drone.py                 # hut + bush, default backend
.venv/bin/python benchmarks/drone.py --backend csrt  # compare a backend
```

## Configuration

All tunables live in [`src/tracker_system/config.py`](src/tracker_system/config.py)
as dataclass defaults — the single source of truth, validated on construction
(out-of-range values raise `ConfigError`). Sections: `video`, `selection` (incl.
HUD-overlay handling), `tracker` (backend + flow-scale), `motion` (ego-motion),
`verifier` (cue weights + memory update gates), `loss` (fused-confidence
hysteresis), `reacquire` (appearance-confirmed search). Edit the file to tune.

## Tests

```bash
pytest
```

## Project layout

```text
tracker-system/
  main.py                    # entry point -> package CLI
  download_models.py         # fetch the ViT / NanoTrack ONNX models
  models/vittrack.onnx       # vendored ViT tracker weights
  benchmarks/drone.py        # hut + bush benchmark
  src/tracker_system/
    cli.py                   # --info and track commands
    pipeline.py              # streaming loop + state machine + FPS meter
    trackers.py              # backends: hybrid (ViT+flow), vit, nano, csrt
    appearance.py            # appearance memory + multi-cue verifier
    motion.py                # global ego-motion estimator
    loss.py                  # fused-confidence loss detection
    reacquire.py             # appearance-confirmed re-acquisition
    selection.py             # selectors + burned-in overlay handling
    overlay.py               # bbox / trajectory / HUD drawing
    geometry.py              # BBox + clamp/patch/resize helpers
    config.py                # typed, validated settings
    video.py                 # streaming VideoSource
  tests/                     # unit + integration
```
