"""Typed, validated configuration loaded from YAML.

Configuration is layered: built-in dataclass defaults are always present, and an
optional YAML file (``configs/default.yaml`` by default, or a user-supplied
path) is merged on top and validated. Keeping the loader small and strict —
unknown keys and out-of-range values raise :class:`ConfigError` — means
misconfiguration surfaces immediately rather than causing subtle runtime
behaviour later in the pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping
import yaml

# Repository root, derived from this file's location:
# src/tracker_system/config/settings.py -> parents[3] == repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
ALLOWED_TRACKERS = ("CSRT", "KCF", "MOSSE")


class ConfigError(ValueError):
    """Raised when configuration is malformed or fails validation."""

@dataclass(frozen=True)
class VideoConfig:
    """Video input / working-frame settings."""

    processing_scale: float = 1.0
    url_open_timeout_ms: int = 10000


@dataclass(frozen=True)
class SelectionConfig:
    """Initial target-selection settings.

    ``handle_overlay`` guards against burned-in screen-fixed overlays (a drone HUD
    crosshair, a timestamp, a watermark) contaminating the Target Profile when the
    user selects a target near one. It samples frames across the video, detects
    temporally-static structural pixels, and — only if the selection box overlaps
    them — inpaints the overlay out of the first-frame patch used to build the
    template and initialise the tracker. It is a strict no-op on clean footage,
    on static scenes, and on selections away from any overlay, and never touches
    the tracking / re-acquisition path.
    """

    default_bbox_size: int = 80
    handle_overlay: bool = True
    overlay_sample_frames: int = 30      # frames sampled across the clip for detection
    overlay_static_std: float = 12.0     # temporal std below which a pixel is "static"
    # Only trust overlay detection when a MAJORITY of the frame is dynamic, i.e. the
    # camera itself is moving so real background can't masquerade as a screen-fixed
    # overlay. A static camera with a moving subject (fewer dynamic pixels) is left
    # untouched. (A static camera with a very large moving subject is the residual
    # blind spot; the robust discriminator is ego-motion estimation, deferred.)
    overlay_min_motion_frac: float = 0.55


@dataclass(frozen=True)
class TrackerConfig:
    """Tracker backend settings."""

    type: str = "CSRT"


@dataclass(frozen=True)
class MotionConfig:
    """Global (camera) motion estimation settings.

    Ego-motion compensation is the backbone for this aerial footage: it predicts
    where a world-fixed target went under camera pan/zoom (disambiguating it from
    identical distractors) and recovers the scale change of the descending drone.
    Flow runs on a downscaled grayscale frame for speed; the estimate is a 4-DOF
    similarity transform fit by RANSAC and reported in full-resolution pixels.
    Every value is a strict no-op on clean/static footage (identity transform).
    """

    enabled: bool = True
    flow_scale: float = 0.5            # downscale for optical flow (speed)
    max_features: int = 600           # goodFeaturesToTrack corner budget
    quality_level: float = 0.01       # corner quality threshold
    min_distance: int = 8             # min spacing between corners (downscaled px)
    min_features: int = 12            # min tracked points to attempt a fit
    ransac_thresh: float = 3.0        # RANSAC reprojection threshold (downscaled px)
    min_inliers: int = 10             # min inliers to trust the transform
    min_inlier_ratio: float = 0.5     # min inlier fraction to trust the transform
    mask_dilate_frac: float = 0.35    # dilate the target box by this frac when masking
    # Accumulated scale drift (either direction) since the last template capture
    # that triggers a guarded refresh of the structural match template under zoom.
    template_refresh_scale: float = 1.3


@dataclass(frozen=True)
class LossConfig:
    """Loss-detection thresholds and confirmation window.

    ``max_lost_frames`` is the hysteresis window: a target must look bad for this
    many consecutive frames before LOST is declared, which prevents state
    chatter and false positives from momentary tracker wobble. The appearance
    (histogram) similarity check runs every frame — it is cheap, and gating it to
    every Nth frame would make it unable to accumulate across the hysteresis
    window and so unable to catch a slow drift onto the background.
    """

    max_lost_frames: int = 8
    max_center_jump_frac: float = 0.25
    max_scale_ratio: float = 1.6
    min_frame_overlap: float = 0.3
    # Adaptive, per-target identity validation. Rather than a fixed similarity
    # floor (which over-fits the trivial ~0.94 self-match of the first frames),
    # the detector learns the target's OWN similarity distribution from sustained
    # genuine tracking and gates on a robust outlier bound:
    #
    #     threshold = median - identity_k * robust_std      (robust_std from MAD)
    #
    # A stable target (drone bush: tight distribution) gets a strict gate that
    # catches a drift onto the background; a noisy target (translucent bottle:
    # wide distribution) gets a very low/negative gate, so appearance becomes a
    # weak signal and motion/spatial continuity carries identity instead. Samples
    # are only collected during a SUSTAINED healthy track and PERSIST across
    # re-acquisitions, so the tracker never re-seeds the reference onto background.
    similarity_ema_alpha: float = 0.3     # fast EMA responsiveness
    identity_stable_frames: int = 8       # consecutive healthy frames before sampling
    identity_window: int = 120            # rolling window of the distribution
    identity_min_samples: int = 45        # samples before the adaptive gate activates
    identity_k: float = 8.0               # robust-std multiplier (higher = more tolerant)
    # Absolute safety floor (below this the appearance is bad regardless of the
    # distribution) — a backstop for the degenerate near-zero case.
    min_similarity: float = 0.15


@dataclass(frozen=True)
class ReacquireConfig:
    """Re-acquisition (SEARCHING) settings.

    The search window is anchored on the motion-predicted position and is kept
    bounded (it must never grow to cover the whole frame, or it would stop being
    a spatial constraint). Re-acquisition is an *identity match*, not mere
    detection: a candidate must pass BOTH an appearance gate (``min_hist_score``)
    and a spatial gate (``min_motion_score`` — it must be near where the target
    is predicted to be). Among the survivors, spatial continuity is the primary
    ranking signal (``weight_motion`` is the largest), with appearance as
    validation. Only matches above ``min_score`` are accepted.
    """

    min_score: float = 0.45
    # Appearance gates: a candidate must look like the target in BOTH structure
    # (template match) and colour (histogram) — this rejects near-prediction
    # background that has neither structure nor the right colour.
    min_template_score: float = 0.3
    min_hist_score: float = 0.3
    # Spatial/identity gate: a candidate must be near the predicted position
    # (motion prior). This is what stops re-acquisition from locking onto an
    # identical look-alike elsewhere in the frame (e.g. a different bottle).
    min_motion_score: float = 0.3
    search_radius_frac: float = 0.15
    search_expansion_frac: float = 0.03
    max_search_radius_frac: float = 0.75
    # Hard cap on the search-region size as a fraction of each frame dimension,
    # so the region stays a locality and never becomes the whole frame.
    max_region_frac: float = 0.6
    # Downscale factor for the (expensive) multi-scale template matching only.
    # SEARCHING was the FPS bottleneck at 1080p; matching a downscaled ROI/template
    # is far cheaper and, since matching is already multi-scale, barely affects
    # localisation (the tracker is re-initialised on the full-res box afterwards).
    # All candidate coordinates are reported back in full-resolution frame space.
    search_scale: float = 0.5
    scales: tuple = (0.8, 1.0, 1.25)
    # Spatial continuity is the primary identity signal; appearance validates.
    weight_template: float = 0.3
    weight_histogram: float = 0.2
    weight_motion: float = 0.5
    # Locality of the motion prior (fraction of frame diagonal): the target is
    # expected to reappear within roughly this distance of its prediction.
    motion_sigma_frac: float = 0.15
    max_prediction_frames: int = 30
    max_candidates: int = 3
    # Anti-thrash: a re-acquisition that collapses back to LOST within
    # ``reacquire_probation_frames`` did not really find the target. After
    # ``max_failed_reacquires`` such failures in a row, suspend re-acquisition for
    # ``reacquire_cooldown_frames`` so the system settles into SEARCHING instead
    # of repeatedly flashing a green box onto a background it cannot confirm.
    reacquire_probation_frames: int = 20
    max_failed_reacquires: int = 3
    reacquire_cooldown_frames: int = 90
    # Raise the re-acquisition colour gate to the target's OWN learned identity
    # threshold (from loss detection) instead of the fixed ``min_hist_score``, so
    # re-acquisition uses the same identity standard as staying-tracked. Prevents
    # re-lock/lose thrash onto background that clears the fixed gate but not the
    # target's learned level (drone sand). Falls back to ``min_hist_score`` until a
    # threshold is learned, and for noisy targets whose learned gate is lower.
    use_identity_appearance_floor: bool = True


@dataclass(frozen=True)
class Settings:
    """Top-level application settings."""

    video: VideoConfig = field(default_factory=VideoConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    reacquire: ReacquireConfig = field(default_factory=ReacquireConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Settings":
        """Build and validate settings from a (possibly partial) mapping."""
        if not isinstance(data, Mapping):
            raise ConfigError("Configuration root must be a mapping")

        known_sections = {f.name for f in fields(cls)}
        unknown = set(data) - known_sections
        if unknown:
            raise ConfigError(
                f"Unknown configuration section(s): {sorted(unknown)}. "
                f"Known sections: {sorted(known_sections)}"
            )

        settings = cls(
            video=_build_section(VideoConfig, data, "video"),
            selection=_build_section(SelectionConfig, data, "selection"),
            tracker=_build_section(TrackerConfig, data, "tracker"),
            motion=_build_section(MotionConfig, data, "motion"),
            loss=_build_section(LossConfig, data, "loss"),
            reacquire=_build_section(ReacquireConfig, data, "reacquire"),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Validate value ranges; raise :class:`ConfigError` on any violation."""
        if not 0.0 < self.video.processing_scale <= 1.0:
            raise ConfigError(
                "video.processing_scale must be in the range (0, 1], got "
                f"{self.video.processing_scale}"
            )
        if self.video.url_open_timeout_ms <= 0:
            raise ConfigError(
                "video.url_open_timeout_ms must be positive, got "
                f"{self.video.url_open_timeout_ms}"
            )
        if self.selection.default_bbox_size <= 0:
            raise ConfigError(
                "selection.default_bbox_size must be positive, got "
                f"{self.selection.default_bbox_size}"
            )
        sel = self.selection
        if sel.overlay_sample_frames < 5:
            raise ConfigError("selection.overlay_sample_frames must be >= 5")
        if sel.overlay_static_std <= 0:
            raise ConfigError("selection.overlay_static_std must be positive")
        if not 0.0 <= sel.overlay_min_motion_frac <= 1.0:
            raise ConfigError("selection.overlay_min_motion_frac must be in [0, 1]")
        if self.tracker.type not in ALLOWED_TRACKERS:
            raise ConfigError(
                f"tracker.type must be one of {list(ALLOWED_TRACKERS)}, got "
                f"{self.tracker.type!r}"
            )

        mot = self.motion
        if not 0.0 < mot.flow_scale <= 1.0:
            raise ConfigError("motion.flow_scale must be in (0, 1]")
        if mot.max_features < 4:
            raise ConfigError("motion.max_features must be >= 4")
        if not 0.0 < mot.quality_level < 1.0:
            raise ConfigError("motion.quality_level must be in (0, 1)")
        if mot.min_distance < 1:
            raise ConfigError("motion.min_distance must be >= 1")
        if mot.min_features < 3:
            raise ConfigError("motion.min_features must be >= 3")
        if mot.ransac_thresh <= 0:
            raise ConfigError("motion.ransac_thresh must be positive")
        if mot.min_inliers < 3:
            raise ConfigError("motion.min_inliers must be >= 3")
        if not 0.0 < mot.min_inlier_ratio <= 1.0:
            raise ConfigError("motion.min_inlier_ratio must be in (0, 1]")
        if mot.mask_dilate_frac < 0:
            raise ConfigError("motion.mask_dilate_frac must be >= 0")
        if mot.template_refresh_scale <= 1.0:
            raise ConfigError("motion.template_refresh_scale must be > 1.0")

        loss = self.loss
        if loss.max_lost_frames < 1:
            raise ConfigError("loss.max_lost_frames must be >= 1")
        if loss.max_center_jump_frac <= 0:
            raise ConfigError("loss.max_center_jump_frac must be positive")
        if loss.max_scale_ratio <= 1.0:
            raise ConfigError("loss.max_scale_ratio must be > 1.0")
        if not -1.0 <= loss.min_similarity <= 1.0:
            raise ConfigError("loss.min_similarity must be in [-1, 1]")
        if not 0.0 <= loss.min_frame_overlap <= 1.0:
            raise ConfigError("loss.min_frame_overlap must be in [0, 1]")
        if not 0.0 < loss.similarity_ema_alpha <= 1.0:
            raise ConfigError("loss.similarity_ema_alpha must be in (0, 1]")
        if loss.identity_stable_frames < 1:
            raise ConfigError("loss.identity_stable_frames must be >= 1")
        if loss.identity_window < 1:
            raise ConfigError("loss.identity_window must be >= 1")
        if loss.identity_min_samples < 1:
            raise ConfigError("loss.identity_min_samples must be >= 1")
        if loss.identity_k < 0:
            raise ConfigError("loss.identity_k must be >= 0")

        rq = self.reacquire
        if not -1.0 <= rq.min_score <= 1.0:
            raise ConfigError("reacquire.min_score must be in [-1, 1]")
        if not -1.0 <= rq.min_template_score <= 1.0:
            raise ConfigError("reacquire.min_template_score must be in [-1, 1]")
        if not -1.0 <= rq.min_hist_score <= 1.0:
            raise ConfigError("reacquire.min_hist_score must be in [-1, 1]")
        if not 0.0 <= rq.min_motion_score <= 1.0:
            raise ConfigError("reacquire.min_motion_score must be in [0, 1]")
        if not 0.0 < rq.max_region_frac <= 1.0:
            raise ConfigError("reacquire.max_region_frac must be in (0, 1]")
        if not 0.0 < rq.search_scale <= 1.0:
            raise ConfigError("reacquire.search_scale must be in (0, 1]")
        if rq.search_radius_frac <= 0:
            raise ConfigError("reacquire.search_radius_frac must be positive")
        if rq.search_expansion_frac < 0:
            raise ConfigError("reacquire.search_expansion_frac must be >= 0")
        if not 0.0 < rq.max_search_radius_frac <= 1.5:
            raise ConfigError("reacquire.max_search_radius_frac must be in (0, 1.5]")
        if not rq.scales or any(s <= 0 for s in rq.scales):
            raise ConfigError("reacquire.scales must be a non-empty list of positive numbers")
        if min(rq.weight_template, rq.weight_histogram, rq.weight_motion) < 0:
            raise ConfigError("reacquire weights must be non-negative")
        if rq.weight_template + rq.weight_histogram + rq.weight_motion <= 0:
            raise ConfigError("reacquire weights must not all be zero")
        if rq.motion_sigma_frac <= 0:
            raise ConfigError("reacquire.motion_sigma_frac must be positive")
        if rq.max_prediction_frames < 0:
            raise ConfigError("reacquire.max_prediction_frames must be >= 0")
        if rq.max_candidates < 1:
            raise ConfigError("reacquire.max_candidates must be >= 1")
        if rq.reacquire_probation_frames < 1:
            raise ConfigError("reacquire.reacquire_probation_frames must be >= 1")
        if rq.max_failed_reacquires < 1:
            raise ConfigError("reacquire.max_failed_reacquires must be >= 1")
        if rq.reacquire_cooldown_frames < 0:
            raise ConfigError("reacquire.reacquire_cooldown_frames must be >= 0")


def _build_section(section_cls: type, data: Mapping[str, Any], name: str) -> Any:
    """Construct one settings section, rejecting unknown keys."""
    raw = data.get(name, {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Configuration section {name!r} must be a mapping")

    valid_keys = {f.name for f in fields(section_cls)}
    unknown = set(raw) - valid_keys
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in section {name!r}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid_keys)}"
        )
    return section_cls(**raw)


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from a YAML file, falling back to built-in defaults.

    Args:
        path: Explicit config file path. If ``None``, ``configs/default.yaml`` is
            used when present; otherwise built-in defaults are returned.

    Raises:
        ConfigError: If a requested file is missing, is not valid YAML mapping,
            or fails validation.
    """
    explicit = path is not None
    source_path = Path(path) if explicit else DEFAULT_CONFIG_PATH

    if not source_path.exists():
        if explicit:
            raise ConfigError(f"Config file not found: {source_path}")
        # No default file on disk: rely on built-in defaults.
        return Settings.from_mapping({})

    try:
        with open(source_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config {source_path}: {exc}") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(
            f"Top-level YAML in {source_path} must be a mapping, got {type(loaded).__name__}"
        )
    return Settings.from_mapping(loaded)
