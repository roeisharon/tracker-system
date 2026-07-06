"""Tracker factory and runtime capability probe.

OpenCV moves classical trackers between namespaces across versions and builds:
CSRT/KCF may live at ``cv2.TrackerCSRT_create`` or ``cv2.legacy.TrackerCSRT_create``,
and MOSSE is typically only available as ``cv2.legacy.TrackerMOSSE_create``. A
headless build (``opencv-python-headless``) may omit the contrib trackers
entirely.

Rather than assume a layout, this module resolves each tracker by trying an
ordered list of candidate constructors, and verifies availability by actually
constructing an instance. This is the Phase 0 environment gate: it lets the
application confirm at runtime that at least CSRT is present before any tracking
logic runs.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional
import cv2

# Ordered candidate constructor paths (relative to the ``cv2`` module) per
# tracker. The first one that resolves and constructs is used.
_TRACKER_CONSTRUCTORS: Dict[str, tuple] = {
    "CSRT": ("TrackerCSRT_create", "legacy.TrackerCSRT_create"),
    "KCF": ("TrackerKCF_create", "legacy.TrackerKCF_create"),
    "MOSSE": ("legacy.TrackerMOSSE_create", "TrackerMOSSE_create"),
}

class TrackerNotAvailableError(RuntimeError):
    """Raised when a requested tracker cannot be created in this OpenCV build."""

def known_trackers() -> List[str]:
    """Return the list of tracker names this factory knows how to build."""
    return list(_TRACKER_CONSTRUCTORS)

def _resolve(path: str) -> Optional[Callable]:
    """Resolve a dotted attribute path under ``cv2`` to a callable, or ``None``."""
    obj = cv2
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if callable(obj) else None

def _constructor_for(name: str) -> Optional[Callable]:
    """Return the first resolvable constructor for ``name``, or ``None``.

    Raises:
        TrackerNotAvailableError: If ``name`` is not a known tracker.
    """
    key = name.upper()
    if key not in _TRACKER_CONSTRUCTORS:
        raise TrackerNotAvailableError(
            f"Unknown tracker {name!r}. Known trackers: {known_trackers()}"
        )
    for path in _TRACKER_CONSTRUCTORS[key]:
        ctor = _resolve(path)
        if ctor is not None:
            return ctor
    return None

def create_tracker(name: str):
    """Create a new tracker instance by name (e.g. ``"CSRT"``).

    Raises:
        TrackerNotAvailableError: If the tracker is unknown, absent from this
            OpenCV build, or cannot be instantiated.
    """
    ctor = _constructor_for(name)
    if ctor is None:
        raise TrackerNotAvailableError(
            f"Tracker {name.upper()!r} is not available in this OpenCV build. "
            "Install 'opencv-contrib-python' (not the headless build). "
            f"Currently available: {available_trackers()}"
        )
    try:
        return ctor()
    except Exception as exc:  # pragma: no cover - defensive
        raise TrackerNotAvailableError(
            f"Failed to create tracker {name.upper()!r}: {exc}"
        ) from exc

def probe_trackers() -> Dict[str, bool]:
    """Probe every known tracker, returning ``{name: available}``.

    Availability is verified by actually constructing an instance, so the result
    reflects what this specific OpenCV build can really do.
    """
    result: Dict[str, bool] = {}
    for name in _TRACKER_CONSTRUCTORS:
        ctor = _constructor_for(name)
        available = False
        if ctor is not None:
            try:
                ctor()
                available = True
            except Exception:
                available = False
        result[name] = available
    return result

def available_trackers() -> List[str]:
    """Return the names of trackers that are actually usable in this build."""
    return [name for name, ok in probe_trackers().items() if ok]
