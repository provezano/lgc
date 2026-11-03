"""Learning with Local and Global Consistency (LGC).

The public functions in this module implement the algorithm from:

    Zhou et al., "Learning with local and global consistency", NeurIPS 2004.

Labels are either integer class ids (with ``-1`` for unlabeled samples) or a
two-dimensional label-confidence matrix whose unlabeled rows are all zero.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from scipy.spatial.distance import cdist

Array: Final = np.ndarray


def build_affinity(
    X: Array,
    sigma: float = 0.1,
    *,
    include_self: bool = False,
) -> Array:
    """Build a Gaussian affinity matrix for samples in ``X``.

    The squared Euclidean distance is used in the Gaussian kernel. Computing
    the kernel directly from squared distances avoids the Python-level
    ``vectorize`` loop used by the original demo and is substantially faster.
    """
    samples = np.asarray(X, dtype=float)
    if samples.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if samples.shape[0] == 0:
        raise ValueError("X must contain at least one sample")
    if not np.isfinite(samples).all():
        raise ValueError("X must contain only finite values")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive finite number")

    squared_distances = cdist(samples, samples, metric="sqeuclidean")
    affinity = np.exp(-squared_distances / (2.0 * sigma**2))
    if not include_self:
        np.fill_diagonal(affinity, 0.0)
    return affinity


def normalize_affinity(affinity: Array) -> Array:
    """Return ``D^-1/2 W D^-1/2`` with safe handling of isolated samples."""
    matrix = np.asarray(affinity, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("affinity must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("affinity must contain only finite values")
    if np.any(matrix < 0):
        raise ValueError("affinity must be non-negative")

    degree = matrix.sum(axis=1)
    inverse_sqrt_degree = np.zeros_like(degree)
    nonzero = degree > 0
    inverse_sqrt_degree[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    return matrix * inverse_sqrt_degree[:, None] * inverse_sqrt_degree[None, :]


def _label_matrix(labels: Array, n_samples: int) -> Array:
    values = np.asarray(labels)
    if values.ndim == 1:
        if len(values) != n_samples:
            raise ValueError("labels must have one entry per sample")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("class labels must be integers")
        known = values[values >= 0]
        if len(known) == 0:
            raise ValueError("at least one sample must be labeled")
        n_classes = int(known.max()) + 1
        result = np.zeros((n_samples, n_classes), dtype=float)
        result[np.flatnonzero(values >= 0), values[values >= 0]] = 1.0
        return result

    if values.ndim != 2 or values.shape[0] != n_samples:
        raise ValueError("labels must be a vector or a matrix with one row per sample")
    if values.shape[1] == 0:
        raise ValueError("labels must contain at least one class")
    result = values.astype(float, copy=True)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("label confidences must be finite and non-negative")
    if not np.any(result.sum(axis=1) > 0):
        raise ValueError("at least one sample must be labeled")
    return result


def propagate(
    normalized_affinity: Array,
    label_matrix: Array,
    *,
    alpha: float = 0.99,
    max_iter: int = 400,
    tol: float = 1e-6,
) -> tuple[Array, int]:
    """Propagate labels until convergence and return scores and iterations."""
    S = np.asarray(normalized_affinity, dtype=float)
    Y = np.asarray(label_matrix, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("normalized_affinity must be square")
    if Y.ndim != 2 or Y.shape[0] != S.shape[0]:
        raise ValueError("label_matrix must match normalized_affinity")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a non-negative finite number")

    F = Y.copy()
    restart = (1.0 - alpha) * Y
    for iteration in range(1, int(max_iter) + 1):
        updated = alpha * (S @ F) + restart
        if np.max(np.abs(updated - F)) <= tol:
            return updated, iteration
        F = updated
    return F, int(max_iter)


def lgc(
    X: Array,
    labels: Array,
    *,
    sigma: float = 0.1,
    alpha: float = 0.99,
    max_iter: int = 400,
    tol: float = 1e-6,
) -> Array:
    """Run LGC and return one class-confidence row per sample."""
    samples = np.asarray(X, dtype=float)
    Y = _label_matrix(labels, len(samples))
    affinity = build_affinity(samples, sigma)
    normalized = normalize_affinity(affinity)
    scores, _ = propagate(
        normalized,
        Y,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
    )
    return scores


def predict(scores: Array) -> Array:
    """Convert class-confidence rows into integer class predictions."""
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("scores must be a non-empty two-dimensional array")
    return np.argmax(values, axis=1)
