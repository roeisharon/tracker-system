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
    # Per-frame scale clamp. Widened from 0.9/1.1 to give headroom for a fast
    # approach / optical zoom on general footage; neutral on the drone's gradual
    # zoom. (Looser bounds trade a little bad-fit robustness for that headroom.)
    flow_scale_min: float = 0.8
    flow_scale_max: float = 1.25
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
    # Rotation-tolerant identity: score the NCC cue against the anchor rotated at
    # every ``rot_ncc_step`` degrees (0 disables). Lets a rotated view of the target
    # (spinning camera / object) still match its own anchor — NCC is rotation-blind.
    rot_ncc_step: int = 45
    ema_alpha: float = 0.3      # recent-template gray EMA rate
    ema_update_conf: float = 0.6   # fused-conf gate to refresh the recent template
    tmpl_update_score: float = 0.7  # OR raw tracker-score gate (breaks the deadlock)
    # Multi-snapshot gallery: capture a template each time the box scale changes by
    # ``snapshot_scale_step`` during confident tracking, keeping up to
    # ``max_snapshots``. Re-acquisition searches the whole gallery, so a target that
    # grew/changed a lot before loss is still *proposed* on return (not just the
    # stale anchor + last recent). 0 snapshots disables the gallery.
    max_snapshots: int = 4
    snapshot_scale_step: float = 1.5


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
    # Ambiguity gate: if the winning match peak has a rival peak >= ambiguity_ratio
    # of it (repetitive scene — identical bushes/roof tiles), the best peak carries
    # little identity, so demand the higher ``t_reacq_ambiguous`` before re-locking.
    ambiguity_ratio: float = 0.9
    t_reacq_ambiguous: float = 0.75
    # Confirm the top-K spatially-distinct coarse candidates with the (costly) full
    # identity verifier and accept the best that clears its bar — the coarse-best
    # peak is not always the identity-best, so ranking by correlation alone can miss
    # the true target when a look-alike edges it out in raw correlation.
    confirm_topk: int = 5
    reacq_every: int = 1        # run the (downscaled) search every N lost frames
    reacq_downscale: float = 0.5
    reacq_scales: tuple = (0.5, 0.75, 1.0, 1.5, 2.0)
    # Rotation sweep in the coarse search: every ``rot_every`` lost frames, also try
    # the anchor rotated at ``rot_step`` degrees so a rotated returning target is
    # localized (the upright search would miss it). 0 disables.
    rot_step: int = 45
    rot_every: int = 3
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
        if rq.confirm_topk < 1:
            raise ConfigError("reacquire.confirm_topk must be >= 1")
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
