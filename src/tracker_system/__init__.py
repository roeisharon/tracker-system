"""Real-Time Arbitrary Object Tracking & Re-acquisition System.

CPU-only tracking of an arbitrary user-selected object, with loss detection and
automatic re-acquisition. The default ``hybrid`` backend pairs a deep Siamese
tracker (centre) with an optical-flow scale estimator (size) so the box follows
extreme zoom, cross-checked by an independent multi-cue appearance verifier.
"""

__version__ = "1.0.0"
