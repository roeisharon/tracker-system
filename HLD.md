# High-Level Design - Real-Time Arbitrary Object Tracking & Re-acquisition

## 1. Purpose

A **CPU-only** application that tracks an arbitrary, user-selected object through a
video and **re-acquires** it after it is lost (occlusion, leaving frame, drastic
appearance/scale change).

---

## 2. The core idea - three decoupled signals

A deep tracker alone drifts or balloons because it tries to answer three questions
with one score. This system answers them **separately** and fuses the results:

```mermaid
flowchart LR
    F[Frame] --> C["Centre<br/>(where)"]
    F --> S["Scale<br/>(how big)"]
    F --> I["Identity<br/>(is it really the target)"]
    C -->|ViT deep tracker,<br/>flow fallback when unsure| BOX[Fused box]
    S -->|local optical-flow<br/>similarity transform| BOX
    I -->|appearance verifier<br/>NCC + HSV + ORB| CONF[Confidence]
    BOX --> CONF
    CONF --> D{Loss / re-acquire<br/>decisions}
```

- **Centre** — OpenCV `TrackerVit` (ONNX, CPU) when confident; otherwise the local
  flow centre (the target is too small/low-texture for the ViT to lock early).
- **Scale** — a local Lucas-Kanade + RANSAC **similarity transform** on features
  inside the box gives the true per-frame zoom, damped by the global camera scale.
- **Identity** — an independent **verifier** (grayscale NCC + HSV histogram +
  ORB/RANSAC inliers) over an **appearance memory**; its confidence, fused with the
  tracker score, drives loss detection and confirms re-acquisition.

---

## 3. Component / module view

Modules are a flat set on the import path (`src/`). Arrows mean "depends on".

```mermaid
graph TD
    subgraph Entry["Entry / CLI"]
        main[main.py]
        cli[cli.py]
    end
    subgraph Orchestration
        pipeline[pipeline.py<br/>state machine + loop]
    end
    subgraph Perception["Perception components"]
        trackers[trackers.py<br/>hybrid / vit / nano / csrt]
        motion[motion.py<br/>ego-motion]
        appearance[appearance.py<br/>memory + verifier]
        loss[loss.py<br/>loss detection]
        reacquire[reacquire.py<br/>re-acquisition]
    end
    subgraph IO["I/O & UI"]
        video[video.py]
        selection[selection.py]
        output[output.py]
        overlay[overlay.py]
    end
    subgraph Foundation
        config[config.py]
        geometry[geometry.py]
    end

    main --> cli
    cli --> pipeline
    cli --> config
    cli --> selection
    cli --> trackers
    cli --> video
    cli --> output

    pipeline --> trackers
    pipeline --> motion
    pipeline --> appearance
    pipeline --> loss
    pipeline --> reacquire
    pipeline --> selection
    pipeline --> overlay
    pipeline --> video
    pipeline --> geometry
    pipeline --> config

    reacquire --> appearance
    selection --> overlay
    trackers --> geometry
    appearance --> geometry
    motion --> geometry
    loss --> geometry
    overlay --> geometry
    trackers --> config
    appearance --> config
    motion --> config
    loss --> config
    reacquire --> config
```

| Module | Responsibility |
|---|---|
| `main.py` | Root entry; puts `src/` on the path and calls the CLI. |
| `cli.py` | Argument parsing, dispatch, run-summary printing. |
| `pipeline.py` | Streaming loop, `TrackerState` machine, FPS metering, result assembly. |
| `trackers.py` | Backend implementations behind one interface; the hybrid ViT+flow tracker. |
| `motion.py` | Global camera (ego) motion as a similarity `Transform`. |
| `appearance.py` | `AppearanceMemory` (anchor + recent + gallery) and the multi-cue `Verifier`. |
| `loss.py` | Fused-confidence loss detection with hysteresis. |
| `reacquire.py` | Coarse multi-scale search + full-identity confirm while LOST. |
| `selection.py` | Target selectors (manual + interactive) and burned-in HUD handling. |
| `video.py` | Streaming `VideoSource` over `cv2.VideoCapture`. |
| `output.py` | Save vs. temp-preview lifecycle + the in-process preview player. |
| `overlay.py` | Drawing the box, trail, seed marker, and HUD panel. |
| `config.py` | Typed, validated `Settings` (single source of truth). |
| `geometry.py` | `BBox` + pure clamp/patch/resize helpers. |

---

## 4. Class view

```mermaid
classDiagram
    class TrackingPipeline {
        +run(source, selector, out_path, show, progress, debug_dir)
        -_track()
        -_search()
    }
    class StateMachine {
        +state
        +timeline
        +to(new_state, frame, reason)
    }
    class TrackerState {
        <<enum>>
        +INIT
        +READY
        +TRACKING
        +LOST
        +SEARCHING
        +REACQUIRED
    }
    class TrackingResult {
        +frames_processed
        +avg_fps
        +tracking_fps
        +tracking_uptime
        +timeline
    }

    class TargetSelector {
        <<abstract>>
        +select(frame)
    }
    class ManualPixelSelector
    class InteractiveClickSelector
    class SelectionResult {
        +bbox
        +seed_point
        +source
    }

    class HybridTracker {
        +update(frame)
        +init()
        +reinit()
        +set_scale_hint()
    }
    class FlowTracker {
        +update()
    }
    class _DeepTracker {
        +update() ViT or Nano
    }
    class _CsrtTracker

    class AppearanceMemory {
        +anchor
        +recent
        +snapshots
        +extract()
        +update()
        +reacq_templates()
    }
    class Verifier {
        +appearance_confidence()
        +fuse_with_tracker()
    }
    class Template {
        +gray
        +hist
        +keypoints
        +descriptors
        +size
    }

    class GlobalMotionEstimator {
        +update(frame, bbox)
    }
    class Transform {
        +matrix
        +confidence
        +scale()
        +apply_point()
    }
    class LossDetector {
        +assess(conf, bbox, prev)
    }
    class Reacquirer {
        +search(frame, hud_mask, predicted)
    }

    class VideoSource {
        +frames()
        +metadata
    }
    class AnnotatedVideoOutput {
        +path
        +finish()
        +cleanup()
    }
    class Settings {
        +video
        +tracker
        +motion
        +verifier
        +loss
        +reacquire
        +selection
    }

    TargetSelector <|-- ManualPixelSelector
    TargetSelector <|-- InteractiveClickSelector
    TargetSelector ..> SelectionResult

    TrackingPipeline *-- StateMachine
    TrackingPipeline ..> TrackingResult
    TrackingPipeline o-- HybridTracker
    TrackingPipeline o-- GlobalMotionEstimator
    TrackingPipeline o-- AppearanceMemory
    TrackingPipeline o-- Verifier
    TrackingPipeline o-- LossDetector
    TrackingPipeline o-- Reacquirer
    TrackingPipeline o-- VideoSource
    StateMachine o-- TrackerState

    HybridTracker *-- _DeepTracker
    HybridTracker *-- FlowTracker
    Verifier o-- AppearanceMemory
    AppearanceMemory *-- Template
    Reacquirer o-- AppearanceMemory
    Reacquirer o-- Verifier
    GlobalMotionEstimator ..> Transform
    LossDetector ..> LossAssessment
    TrackingPipeline ..> Settings
```

---

## 5. Tracker state machine

```mermaid
stateDiagram-v2
    [*] --> INIT
    INIT --> READY: target selected
    READY --> TRACKING: tracking started
    TRACKING --> LOST: loss confirmed (hysteresis)
    LOST --> SEARCHING: begin scan (same frame)
    SEARCHING --> SEARCHING: no match — keep searching
    SEARCHING --> REACQUIRED: appearance-confirmed match
    REACQUIRED --> TRACKING: resume next frame
```

---

## 6. Per-frame data flow

```mermaid
flowchart TD
    A[Read frame] --> B[Downscale to working resolution]
    B --> C[GlobalMotionEstimator.update → Transform]
    C --> D{State?}

    D -->|TRACKING| E[tracker.update → found, box, score]
    E --> F[Verifier.appearance_confidence]
    F --> G[fuse tracker score + appearance]
    G --> H[LossDetector.assess]
    H -->|healthy| I[memory.update + append trajectory]
    H -->|confirmed lost| J[to LOST → SEARCHING]
    H -->|suspect| K[hold last box]

    D -->|SEARCHING| L{reacq_every frame<br/>and not in cooldown?}
    L -->|yes| M[Reacquirer.search:<br/>coarse match + identity confirm]
    M -->|found| N[tracker.reinit + memory.update<br/>→ REACQUIRED]
    M -->|not found| O[hold, keep searching]
    L -->|no| O

    I --> P[render overlay]
    J --> P
    K --> P
    N --> P
    O --> P
    P --> Q[write to video and/or show live]
    Q --> A
```

---

## 7. Sequence - a full `track` run

```mermaid
sequenceDiagram
    actor User
    participant CLI as cli._run_track
    participant Out as AnnotatedVideoOutput
    participant Pipe as TrackingPipeline
    participant Sel as TargetSelector
    participant Src as VideoSource
    participant Comp as Perception components
    participant Ovl as overlay

    User->>CLI: python main.py track video.mp4
    CLI->>Out: create (temp or --save path)
    CLI->>Pipe: run(source, selector, out_path, show)
    Pipe->>Src: read() first frame
    Pipe->>Sel: select(first_frame)
    Sel-->>User: interactive picker (click / +/- / Enter)
    Sel-->>Pipe: SelectionResult(bbox, seed)
    Pipe->>Comp: initialise memory, tracker, motion, detector

    loop each frame until EOF / Esc
        Pipe->>Src: next frame
        Pipe->>Comp: motion + tracker + verifier + loss (TRACKING)
        Note over Pipe,Comp: or reacquire search while SEARCHING
        Pipe->>Ovl: render_overlay(frame, box, state, fps)
        Pipe->>Out: write frame (and show live)
    end

    Pipe-->>CLI: TrackingResult (uptime, fps, timeline)
    CLI->>User: print run summary
    CLI->>Out: finish()
    Out-->>User: preview window (Space/←/→/Esc), then delete temp
```

---

## 8. Sequence - re-acquisition (while LOST)

```mermaid
sequenceDiagram
    participant Pipe as TrackingPipeline._search
    participant Motion
    participant Reac as Reacquirer
    participant Mem as AppearanceMemory
    participant Ver as Verifier

    Pipe->>Motion: carry predicted centre by camera motion
    alt reacq_every frame and not in cooldown
        Pipe->>Reac: search(frame, hud_mask, predicted)
        Reac->>Mem: reacq_templates() (anchor + gallery + recent)
        Reac->>Reac: coarse multi-scale/rotation matchTemplate -> best peak
        Reac->>Ver: appearance_confidence(best box, force_orb)
        alt confidence >= t_reacq (stricter when ambiguous)
            Reac-->>Pipe: (box, confidence)
            Pipe->>Pipe: tracker.reinit, detector.reset -> REACQUIRED
        else below bar
            Reac-->>Pipe: None (stay LOST)
        end
    else throttled / cooldown
        Pipe->>Pipe: hold last box
    end
```
