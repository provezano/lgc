"""Learning with Local and Global Consistency."""

from .core import (
    build_affinity,
    lgc,
    normalize_affinity,
    predict,
    propagate,
)

__all__ = [
    "build_affinity",
    "lgc",
    "normalize_affinity",
    "predict",
    "propagate",
]
