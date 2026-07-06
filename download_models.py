"""Download the ONNX tracker models into ./models.

    python download_models.py           # ViT tracker (default hybrid backend)
    python download_models.py --nano    # also fetch the NanoTrack fallback models

The ViT model (~0.7 MB) is all the default backend needs and is normally vendored
in ./models already; this script is a convenience for a fresh checkout. Weights
come from the official mirrors (opencv_zoo stores Git-LFS stubs).
"""
from __future__ import annotations

import argparse
import os
import urllib.request

VIT = "models/vittrack.onnx"
NANO_BACKBONE = "models/nanotrack_backbone.onnx"
NANO_HEAD = "models/nanotrack_head.onnx"

VIT_URL = ("https://huggingface.co/opencv/object_tracking_vittrack/resolve/main/"
           "object_tracking_vittrack_2023sep.onnx?download=true")
NANO_BACKBONE_URL = ("https://raw.githubusercontent.com/HonglinChu/SiamTrackers/master/"
                     "NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx")
NANO_HEAD_URL = ("https://raw.githubusercontent.com/HonglinChu/SiamTrackers/master/"
                 "NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx")


def _fetch(url: str, path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(f"[skip] {os.path.basename(path)} already present")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[get ] {url}")
    urllib.request.urlretrieve(url, path)
    print(f"[ok  ] {os.path.basename(path)} ({os.path.getsize(path)} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nano", action="store_true", help="also download NanoTrack models")
    args = ap.parse_args()
    _fetch(VIT_URL, VIT)
    if args.nano:
        _fetch(NANO_BACKBONE_URL, NANO_BACKBONE)
        _fetch(NANO_HEAD_URL, NANO_HEAD)


if __name__ == "__main__":
    main()
