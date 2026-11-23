"""scikit-learn compatible estimator for LGC."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .core import (
    _label_matrix,
    build_affinity,
    build_knn_affinity,
    normalize_affinity,
    propagate,
)


class LGCClassifier(ClassifierMixin, BaseEstimator):
    """Semi-supervised LGC classifier.

    ``y`` uses ``-1`` for unlabeled rows when it is numeric. Predictions for
    new samples use their Gaussian affinities to the fitted training graph.
    """

    def __init__(
        self,
        *,
        sigma: float = 0.1,
        alpha: float = 0.99,
        max_iter: int = 400,
        tol: float = 1e-6,
        n_neighbors: int | None = None,
    ) -> None:
        self.sigma = sigma
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGCClassifier":
        X_checked, y_checked = check_X_y(X, y, dtype="numeric", y_numeric=False)
        if y_checked.dtype.kind in "biufc":
            known_mask = y_checked != -1
        elif y_checked.dtype.kind == "O":
            known_mask = np.array([value != -1 for value in y_checked], dtype=bool)
        else:
            known_mask = np.ones(len(y_checked), dtype=bool)
        if not np.any(known_mask):
            raise ValueError("y must contain at least one labeled sample")

        self.classes_, encoded = np.unique(y_checked[known_mask], return_inverse=True)
        encoded_labels = np.full(y_checked.shape, -1, dtype=int)
        encoded_labels[known_mask] = encoded
        self.X_ = X_checked
        self.n_features_in_ = X_checked.shape[1]
        self._train_affinity_ = (
            build_affinity(self.X_, self.sigma)
            if self.n_neighbors is None
            else build_knn_affinity(self.X_, self.sigma, self.n_neighbors)
        )
        self._train_normalized_ = normalize_affinity(self._train_affinity_)
        Y = _label_matrix(encoded_labels, len(self.X_))
        self.scores_, self.n_iter_ = propagate(
            self._train_normalized_,
            Y,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self._train_inverse_sqrt_degree_ = np.zeros(len(self.X_))
        degree = np.asarray(self._train_affinity_.sum(axis=1)).ravel()
        nonzero = degree > 0
        self._train_inverse_sqrt_degree_[nonzero] = 1 / np.sqrt(degree[nonzero])
        if self.n_neighbors is not None:
            from sklearn.neighbors import NearestNeighbors

            self._neighbor_model_ = NearestNeighbors(
                n_neighbors=min(self.n_neighbors, len(self.X_)),
                algorithm="auto",
            ).fit(self.X_)
        return self

    def _scores_for(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("X_", "scores_", "classes_"))
        X_checked = check_array(X, dtype="numeric")
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )
        if self.n_neighbors is not None:
            distances, indices = self._neighbor_model_.kneighbors(X_checked)
            weights = np.exp(-(distances**2) / (2 * self.sigma**2))
            row_degree = weights.sum(axis=1)
            row_inverse = np.zeros_like(row_degree)
            nonzero = row_degree > 0
            row_inverse[nonzero] = 1 / np.sqrt(row_degree[nonzero])
            cross_normalized = sparse.csr_matrix(
                (weights.ravel() * row_inverse.repeat(weights.shape[1]),
                 (np.repeat(np.arange(len(X_checked)), weights.shape[1]), indices.ravel())),
                shape=(len(X_checked), len(self.X_)),
            )
            return cross_normalized.multiply(
                self._train_inverse_sqrt_degree_[None, :]
            ) @ self.scores_
        squared_distances = np.maximum(
            0.0,
            np.sum(X_checked**2, axis=1)[:, None]
            + np.sum(self.X_**2, axis=1)[None, :]
            - 2 * X_checked @ self.X_.T,
        )
        weights = np.exp(-squared_distances / (2 * self.sigma**2))
        if self.n_neighbors is not None:
            count = min(self.n_neighbors, self.X_.shape[0])
            nearest = np.argpartition(weights, -count, axis=1)[:, -count:]
            mask = np.zeros_like(weights, dtype=bool)
            rows = np.arange(len(X_checked))[:, None]
            mask[rows, nearest] = True
            weights[~mask] = 0.0
        row_degree = weights.sum(axis=1)
        row_inverse = np.zeros_like(row_degree)
        nonzero = row_degree > 0
        row_inverse[nonzero] = 1 / np.sqrt(row_degree[nonzero])
        cross_normalized = (
            weights
            * row_inverse[:, None]
            * self._train_inverse_sqrt_degree_[None, :]
        )
        return cross_normalized @ self.scores_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for samples in ``X``."""
        check_is_fitted(self, ("X_", "scores_", "classes_"))
        scores = self._scores_for(X)
        totals = scores.sum(axis=1, keepdims=True)
        probabilities = np.divide(
            scores,
            totals,
            out=np.full_like(scores, 1 / len(self.classes_)),
            where=totals > 0,
        )
        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in ``X``."""
        check_is_fitted(self, ("X_", "scores_", "classes_"))
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean accuracy on the supplied labeled samples."""
        check_is_fitted(self, ("X_", "scores_", "classes_"))
        y_checked = check_array(y, ensure_2d=False, dtype=None)
        if np.any(y_checked == -1):
            raise ValueError("score requires labeled target values")
        return float(np.mean(self.predict(X) == y_checked))
