import numpy as np
import pytest

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scipy import sparse

from lgc import (
    LGCClassifier,
    build_affinity,
    build_knn_affinity,
    lgc,
    normalize_affinity,
    predict,
    propagate,
)


def test_gaussian_affinity_uses_squared_distance_and_zero_diagonal():
    affinity = build_affinity(np.array([[0.0], [2.0]]), sigma=1.0)
    assert affinity[0, 0] == 0.0
    np.testing.assert_allclose(affinity[0, 1], np.exp(-2.0))


def test_normalization_handles_isolated_samples():
    normalized = normalize_affinity(
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )
    assert np.isfinite(normalized).all()
    assert normalized[2].sum() == 0.0
    np.testing.assert_allclose(normalized[:2, :2], [[0.0, 1.0], [1.0, 0.0]])


def test_knn_affinity_is_sparse_and_symmetric():
    X = np.arange(30, dtype=float).reshape(-1, 1)
    affinity = build_knn_affinity(X, sigma=1.0, n_neighbors=2)
    assert sparse.isspmatrix_csr(affinity)
    assert affinity.nnz <= len(X) * 4
    np.testing.assert_allclose(affinity.toarray(), affinity.T.toarray())


def test_lgc_knn_mode_preserves_cluster_predictions():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    scores = lgc(X, np.array([0, -1, 1, -1]), sigma=1.0, n_neighbors=1)
    np.testing.assert_array_equal(predict(scores), [0, 0, 1, 1])


def test_lgc_accepts_integer_labels_and_predicts_two_clusters():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    labels = np.array([0, -1, 1, -1])
    scores = lgc(X, labels, sigma=1.0, alpha=0.9, tol=1e-10)
    np.testing.assert_array_equal(predict(scores), [0, 0, 1, 1])


def test_lgc_accepts_one_hot_label_matrix():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    labels = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    scores = lgc(X, labels, sigma=1.0, alpha=0.9)
    np.testing.assert_array_equal(predict(scores), [0, 0, 1, 1])


def test_propagation_stops_before_iteration_limit():
    S = np.eye(2)
    Y = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores, iterations = propagate(S, Y, alpha=0.9, max_iter=100, tol=1e-12)
    assert iterations == 1
    np.testing.assert_allclose(scores, Y)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sigma": 0}, "sigma"),
        ({"alpha": 1}, "alpha"),
        ({"max_iter": 0}, "max_iter"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        lgc(np.array([[0.0], [1.0]]), np.array([0, -1]), **kwargs)


def test_sklearn_classifier_supports_unlabeled_rows_and_probabilities():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    y = np.array(["left", -1, "right", -1], dtype=object)
    estimator = LGCClassifier(sigma=1.0, alpha=0.9, tol=1e-10).fit(X, y)

    np.testing.assert_array_equal(estimator.predict(X), ["left", "left", "right", "right"])
    probabilities = estimator.predict_proba(np.array([[0.05], [10.05]]))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_sklearn_classifier_works_in_pipeline_and_clone():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    y = np.array([0, -1, 1, -1])
    pipeline = make_pipeline(StandardScaler(), LGCClassifier(sigma=1.0))
    fitted = pipeline.fit(X, y)
    assert fitted.score(X[[0, 2]], y[[0, 2]]) == 1.0
    cloned = clone(LGCClassifier(sigma=0.5, max_iter=10))
    assert cloned.get_params()["sigma"] == 0.5


def test_sklearn_classifier_knn_mode_predicts_new_samples():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    y = np.array([0, -1, 1, -1])
    estimator = LGCClassifier(sigma=1.0, n_neighbors=1).fit(X, y)
    np.testing.assert_array_equal(estimator.predict([[0.05], [10.05]]), [0, 1])
