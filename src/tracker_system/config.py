"""Typed, validated configuration.

All tunables live here as dataclass defaults — the single source of truth (no
external config file). Every :class:`Settings` is validated on construction;
out-of-range values raise :class:`ConfigError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pathlib import Path

# src/tracker_system/config.py -> parents[2] == repo root (for model paths).
REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_BACKENDS = ("hybrid", "vit", "nano", "csrt")


class ConfigError(ValueError):
    """Raised when configuration is malformed or fails validation."""


@dataclass(frozen=True)
class VideoConfig:
    processing_scale: float = 1.0
    url_open_timeout_ms: int = 10000


@dataclass(frozen=True)
class SelectionConfig:
    """Initial selection + burned-in overlay (HUD) handling."""

    default_bbox_size: int = 90
    handle_overlay: bool = True
    overlay_sample_frames: int = 30      # frames sampled across the clip
    overlay_static_std: float = 12.0     # temporal std below which a pixel is "static"
    overlay_min_motion_frac: float = 0.55  # min dynamic frame fraction to trust detection


@dataclass(frozen=True)
class TrackerConfig:
    """Backend selection + the flow-scale estimator used by the hybrid tracker.

    ``hybrid`` = deep tracker (ViT) for the centre + optical-flow similarity
    transform for the size, which is what survives the drone's extreme zoom.
    ``vit``/``nano``/``csrt`` are single-backend fallbacks.
    """

    backend: str = "hybrid"
    vit_model: str = "models/vittrack.onnx"
    nano_backbone: str = "models/nanotrack_backbone.onnx"
    nano_head: str = "models/nanotrack_head.onnx"
    min_box: float = 12.0
    # Above this native ViT score, trust the ViT centre; below it, fall back to the
    # flow-tracked centre (the small/low-texture target the ViT can't lock early).
    vit_trust_score: float = 0.30
    # Flow-scale estimator (drives box size under zoom).
    flow_max_features: int = 200
    flow_min_features: int = 8
    flow_scale_min: float = 0.9        # per-frame scale clamp (lower)
    flow_scale_max: float = 1.1        # per-frame scale clamp (upper)
    flow_replenish_below: int = 40     # re-detect features when fewer remain
    flow_score_norm: float = 60.0      # inlier count that maps to score 1.0
    scale_cross_check: bool = True     # blend flow scale with ego global scale


@dataclass(frozen=True)
class MotionConfig:
    """Global (camera) ego-motion estimator — used for scale + reacquire seeding."""

    enabled: bool = True
    flow_scale: float = 0.5
    max_features: int = 600
    quality_level: float = 0.01
    min_distance: int = 8
    min_features: int = 12
    ransac_thresh: float = 3.0
    min_inliers: int = 10
    min_inlier_ratio: float = 0.5
    mask_dilate_frac: float = 0.35


@dataclass(frozen=True)
class VerifierConfig:
    """Multi-cue appearance verifier + appearance-memory update gates."""

    w_ncc: float = 0.4          # grayscale NCC (structure)
    w_hist: float = 0.3         # HSV histogram (colour)
    w_orb: float = 0.3          # ORB + RANSAC inliers (distinctive geometry)
    w_tracker: float = 0.5      # blend of tracker native score vs appearance
    orb_every: int = 5          # run the (costly) ORB cue every N frames
    orb_nfeatures: int = 500
    max_patch: int = 256        # cap patch side for descriptors (bounds cost as the box grows)
    ema_alpha: float = 0.3      # recent-template gray EMA rate
    ema_update_conf: float = 0.6   # fused-conf gate to refresh the recent template
    tmpl_update_score: float = 0.7  # OR raw tracker-score gate (breaks the deadlock)


@dataclass(frozen=True)
class LossConfig:
    """Fused-confidence loss detection with a hysteresis window."""

    t_lost: float = 0.35        # fused confidence below this = a bad frame
    lost_patience: int = 8      # consecutive bad frames before LOST
    min_frame_overlap: float = 0.3   # min box fraction inside the frame
    max_scale_ratio: float = 4.0     # per-frame box area explosion/collapse sanity


@dataclass(frozen=True)
class ReacquireConfig:
    """Appearance-confirmed re-acquisition while LOST."""

    t_reacq: float = 0.55       # stricter accept threshold than t_lost
    reacq_every: int = 1        # run the (downscaled) search every N lost frames
    reacq_downscale: float = 0.5
    reacq_scales: tuple = (0.5, 0.75, 1.0, 1.5, 2.0)
    # Anti-thrash: a re-acquire that collapses back within probation counts as
    # failed; after too many, cool down so we settle into LOST instead of flashing.
    reacquire_probation_frames: int = 20
    max_failed_reacquires: int = 3
    reacquire_cooldown_frames: int = 90


@dataclass(frozen=True)
class Settings:
    video: VideoConfig = field(default_factory=VideoConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    reacquire: ReacquireConfig = field(default_factory=ReacquireConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not 0.0 < self.video.processing_scale <= 1.0:
            raise ConfigError("video.processing_scale must be in (0, 1]")
        if self.selection.default_bbox_size <= 0:
            raise ConfigError("selection.default_bbox_size must be positive")
        if self.selection.overlay_sample_frames < 5:
            raise ConfigError("selection.overlay_sample_frames must be >= 5")
        if not 0.0 <= self.selection.overlay_min_motion_frac <= 1.0:
            raise ConfigError("selection.overlay_min_motion_frac must be in [0, 1]")

        trk = self.tracker
        if trk.backend not in ALLOWED_BACKENDS:
            raise ConfigError(f"tracker.backend must be one of {list(ALLOWED_BACKENDS)}")
        if trk.min_box <= 0:
            raise ConfigError("tracker.min_box must be positive")
        if not 0.0 < trk.flow_scale_min <= 1.0 <= trk.flow_scale_max:
            raise ConfigError("tracker.flow_scale_min <= 1 <= flow_scale_max required")
        if trk.flow_min_features < 3:
            raise ConfigError("tracker.flow_min_features must be >= 3")

        mot = self.motion
        if not 0.0 < mot.flow_scale <= 1.0:
            raise ConfigError("motion.flow_scale must be in (0, 1]")
        if not 0.0 < mot.min_inlier_ratio <= 1.0:
            raise ConfigError("motion.min_inlier_ratio must be in (0, 1]")

        ver = self.verifier
        if min(ver.w_ncc, ver.w_hist, ver.w_orb) < 0 or (ver.w_ncc + ver.w_hist + ver.w_orb) <= 0:
            raise ConfigError("verifier cue weights must be non-negative and not all zero")
        if not 0.0 <= ver.w_tracker <= 1.0:
            raise ConfigError("verifier.w_tracker must be in [0, 1]")
        if ver.orb_every < 1:
            raise ConfigError("verifier.orb_every must be >= 1")
        if not 0.0 < ver.ema_alpha <= 1.0:
            raise ConfigError("verifier.ema_alpha must be in (0, 1]")

        loss = self.loss
        if not -1.0 <= loss.t_lost <= 1.0:
            raise ConfigError("loss.t_lost must be in [-1, 1]")
        if loss.lost_patience < 1:
            raise ConfigError("loss.lost_patience must be >= 1")
        if not 0.0 <= loss.min_frame_overlap <= 1.0:
            raise ConfigError("loss.min_frame_overlap must be in [0, 1]")
        if loss.max_scale_ratio <= 1.0:
            raise ConfigError("loss.max_scale_ratio must be > 1.0")

        rq = self.reacquire
        if not -1.0 <= rq.t_reacq <= 1.0:
            raise ConfigError("reacquire.t_reacq must be in [-1, 1]")
        if rq.reacq_every < 1:
            raise ConfigError("reacquire.reacq_every must be >= 1")
        if not 0.0 < rq.reacq_downscale <= 1.0:
            raise ConfigError("reacquire.reacq_downscale must be in (0, 1]")
        if not rq.reacq_scales or any(s <= 0 for s in rq.reacq_scales):
            raise ConfigError("reacquire.reacq_scales must be positive and non-empty")
        if rq.max_failed_reacquires < 1:
            raise ConfigError("reacquire.max_failed_reacquires must be >= 1")
