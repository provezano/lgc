import numpy as np
import pytest

from lgc import build_affinity, lgc, normalize_affinity, predict, propagate


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
