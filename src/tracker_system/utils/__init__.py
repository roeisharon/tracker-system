"""Small, dependency-light helpers (geometry, image ops)."""

from .geometry import (
    BBox,
    bbox_from_center,
    clamp_bbox,
    clamp_point,
    extract_patch,
    frame_overlap_ratio,
    scale_bbox,
)
from .image import resize_frame

__all__ = [
    "BBox",
    "bbox_from_center",
    "clamp_bbox",
    "clamp_point",
    "extract_patch",
    "frame_overlap_ratio",
    "scale_bbox",
    "resize_frame",
]
