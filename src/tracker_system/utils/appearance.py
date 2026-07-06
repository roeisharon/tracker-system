"""Cheap appearance similarity for loss detection and re-acquisition.

Two complementary, size-independent signals are blended:

- **HSV hue-saturation histogram correlation** — colour identity. Scale-free and
  cheap, but weak on this footage where the dark hut and dark desert bushes share
  a colour distribution (a colour-only gate lets the tracker drift bush-to-bush).
- **Structural correlation** — both patches are resized to a common small grid and
  compared by normalised cross-correlation on the grayscale (gradient) content.
  Resizing to a shared size removes scale, so the hut's *rectangular, straight-
  edged* structure still matches itself across the drone's zoom, while an
  irregular bush blob does not. This is what separates the target from same-colour
  distractors — the discriminator the colour histogram alone lacks.

The blend is the identity signal shared by loss detection (drift onto a
structurally-different distractor is caught) and re-acquisition candidate scoring
(same-colour look-alikes score lower).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# Common grid both patches are resized to for the structural comparison. Small
# enough to be cheap and scale-normalising, large enough to keep coarse shape.
_STRUCT_SIZE = 48
# Blend weight of the structural term against the colour histogram term.
_STRUCT_WEIGHT = 0.5


def hs_histogram(patch: np.ndarray) -> np.ndarray:
    """Normalised hue-saturation histogram of a BGR patch."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def _histogram_correlation(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    score = cv2.compareHist(hs_histogram(patch_a), hs_histogram(patch_b), cv2.HISTCMP_CORREL)
    return 0.0 if math.isnan(score) else float(score)


def _structural_correlation(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    """Scale-normalised grayscale NCC in ``[-1, 1]`` (shape/structure match)."""
    a = cv2.cvtColor(patch_a, cv2.COLOR_BGR2GRAY) if patch_a.ndim == 3 else patch_a
    b = cv2.cvtColor(patch_b, cv2.COLOR_BGR2GRAY) if patch_b.ndim == 3 else patch_b
    a = cv2.resize(a, (_STRUCT_SIZE, _STRUCT_SIZE), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (_STRUCT_SIZE, _STRUCT_SIZE), interpolation=cv2.INTER_AREA)
    result = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
    score = float(result[0, 0])
    return 0.0 if math.isnan(score) else score


def appearance_similarity(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    """Blended colour+structure similarity in roughly ``[-1, 1]``.

    Returns ``1.0`` when a patch is missing/empty (cannot assess -> assume fine,
    never a false positive). Higher means "more likely the same object" in both
    colour and shape; the structural term is what rejects same-colour distractors.
    """
    if patch_a is None or patch_b is None or patch_a.size == 0 or patch_b.size == 0:
        return 1.0
    hist = _histogram_correlation(patch_a, patch_b)
    struct = _structural_correlation(patch_a, patch_b)
    return (1.0 - _STRUCT_WEIGHT) * hist + _STRUCT_WEIGHT * struct
