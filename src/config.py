"""Typed, validated configuration.

All tunables live here as dataclass defaults — the single source of truth (no
external config file). Every :class:`Settings` is validated on construction;
out-of-range values raise :class:`ConfigError`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# src/config.py -> parents[1] == repo root (for model paths).
REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_BACKENDS = ("hybrid", "vit", "nano", "csrt")


class ConfigError(ValueError):
    """Raised when configuration is malformed or fails validation."""


@dataclass(frozen=True)
class VideoConfig:
    processing_scale: float = 1.0      # downscale every frame before tracking (1.0 = native res)
    url_open_timeout_ms: int = 10000   # give up opening a URL source after this long


@dataclass(frozen=True)
class SelectionConfig:
    """Initial selection + burned-in overlay (HUD) handling."""

    default_bbox_size: int = 90          # initial box side when --bbox is not given (px)
    min_bbox_size: int = 20              # interactive-selector resize floor
    max_bbox_size: int = 200            # interactive-selector resize ceiling
    handle_overlay: bool = True          # detect + inpaint a burned-in HUD before tracking
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

    backend: str = "hybrid"            # hybrid|vit|nano|csrt (see class docstring)
    vit_model: str = "models/vittrack.onnx"                # ViT tracker ONNX weights
    nano_backbone: str = "models/nanotrack_backbone.onnx"  # NanoTrack ONNX backbone
    nano_head: str = "models/nanotrack_head.onnx"          # NanoTrack ONNX head
    min_box: float = 12.0              # floor on the tracked box side length (px)
    vit_trust_score: float = 0.30      # trust the ViT centre above this score, else use flow
    flow_max_features: int = 200       # max LK features tracked in the box for the scale estimate
    flow_min_features: int = 8         # below this the flow scale is treated as unreliable
    flow_scale_min: float = 0.8        # per-frame scale-shrink clamp (headroom for fast zoom)
    flow_scale_max: float = 1.25       # per-frame scale-grow clamp
    flow_replenish_below: int = 40     # re-detect features when fewer remain
    flow_score_norm: float = 60.0      # inlier count that maps to score 1.0
    scale_cross_check: bool = True     # blend flow scale with ego global scale


@dataclass(frozen=True)
class MotionConfig:
    """Global (camera) ego-motion estimator — used for scale + reacquire seeding."""

    enabled: bool = True               # estimate global camera motion at all
    flow_scale: float = 0.5            # downscale factor for the ego-motion flow
    max_features: int = 600           # max background features for the estimate
    quality_level: float = 0.01       # goodFeaturesToTrack quality threshold
    min_distance: int = 8             # min spacing between features (px)
    min_features: int = 12            # need at least this many to estimate motion
    ransac_thresh: float = 3.0        # RANSAC inlier reprojection threshold (px)
    min_inliers: int = 10             # min inliers for a trustworthy transform
    min_inlier_ratio: float = 0.5     # min inlier fraction for a trustworthy transform
    mask_dilate_frac: float = 0.35    # dilate the target-exclusion mask by this box fraction


@dataclass(frozen=True)
class VerifierConfig:
    """Multi-cue appearance verifier + appearance-memory update gates."""

    w_ncc: float = 0.4          # grayscale NCC (structure)
    w_hist: float = 0.3         # HSV histogram (colour)
    w_orb: float = 0.3          # ORB + RANSAC inliers (distinctive geometry)
    w_tracker: float = 0.5      # blend of tracker native score vs appearance
    orb_every: int = 5          # run the (costly) ORB cue every N frames
    orb_nfeatures: int = 500    # ORB keypoints detected per patch
    max_patch: int = 256        # cap patch side for descriptors (bounds cost as the box grows)
    rot_ncc_step: int = 45      # also score NCC vs the anchor rotated every N° (0 disables)
    ema_alpha: float = 0.3      # recent-template gray EMA rate
    ema_update_conf: float = 0.6   # fused-conf gate to refresh the recent template
    tmpl_update_score: float = 0.7  # OR raw tracker-score gate (breaks the deadlock)
    max_snapshots: int = 4      # gallery size: snapshots spanning scale change (0 disables)
    snapshot_scale_step: float = 1.5  # capture a snapshot per this much box-scale change


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

    t_reacq: float = 0.55       # accept a re-lock above this fused confidence (stricter than t_lost)
    ambiguity_ratio: float = 0.9      # a rival peak >= this fraction of the best = ambiguous scene
    t_reacq_ambiguous: float = 0.75   # stricter accept bar when the match is ambiguous
    reacq_every: int = 3        # run the heavy full-frame search every N lost frames (throttle)
    reacq_downscale: float = 0.5  # coarse-search downscale; full-res confirm still gates identity
    reacq_scales: tuple = (0.5, 0.75, 1.0, 1.5, 2.0)  # template scales tried in the coarse search
    rot_step: int = 45          # coarse-search rotation step in degrees (0 disables the sweep)
    rot_every: int = 3          # run the rotation sweep every N lost frames
    reacquire_probation_frames: int = 20  # a re-lock collapsing within this many frames = failed
    max_failed_reacquires: int = 3        # this many failures in a row triggers a cooldown
    reacquire_cooldown_frames: int = 90   # stay LOST this long after too many failed re-locks


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
        if not (0 < self.selection.min_bbox_size <= self.selection.default_bbox_size
                <= self.selection.max_bbox_size):
            raise ConfigError(
                "selection sizes must satisfy 0 < min_bbox_size <= default_bbox_size <= max_bbox_size")
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
        if ver.rot_ncc_step < 0 or (ver.rot_ncc_step and 360 % ver.rot_ncc_step):
            raise ConfigError("verifier.rot_ncc_step must be 0 or a divisor of 360")
        if ver.max_snapshots < 0:
            raise ConfigError("verifier.max_snapshots must be >= 0")
        if ver.snapshot_scale_step <= 1.0:
            raise ConfigError("verifier.snapshot_scale_step must be > 1.0")

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
        if not -1.0 <= rq.t_reacq_ambiguous <= 1.0:
            raise ConfigError("reacquire.t_reacq_ambiguous must be in [-1, 1]")
        if not 0.0 < rq.ambiguity_ratio <= 1.0:
            raise ConfigError("reacquire.ambiguity_ratio must be in (0, 1]")
        if rq.rot_step < 0 or (rq.rot_step and 360 % rq.rot_step):
            raise ConfigError("reacquire.rot_step must be 0 or a divisor of 360")
        if rq.rot_every < 1:
            raise ConfigError("reacquire.rot_every must be >= 1")
        if rq.reacq_every < 1:
            raise ConfigError("reacquire.reacq_every must be >= 1")
        if not 0.0 < rq.reacq_downscale <= 1.0:
            raise ConfigError("reacquire.reacq_downscale must be in (0, 1]")
        if not rq.reacq_scales or any(s <= 0 for s in rq.reacq_scales):
            raise ConfigError("reacquire.reacq_scales must be positive and non-empty")
        if rq.max_failed_reacquires < 1:
            raise ConfigError("reacquire.max_failed_reacquires must be >= 1")
