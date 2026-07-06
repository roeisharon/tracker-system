"""Target selection: manual pixel coordinates and mouse click."""

from .cv_click_selector import CvClickSelector
from .target_selector import (
    ManualPixelSelector,
    SelectionError,
    SelectionResult,
    TargetSelector,
)

__all__ = [
    "CvClickSelector",
    "ManualPixelSelector",
    "SelectionError",
    "SelectionResult",
    "TargetSelector",
]
