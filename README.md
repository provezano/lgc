# Learning with Local and Global Consistency

An efficient, tested implementation of the LGC semi-supervised learning
algorithm from Zhou et al., *Learning with Local and Global Consistency*
(NeurIPS 2004).

## Install

Runtime dependencies:

```bash
python3 -m pip install .
```

For development and tests:

```bash
python3 -m pip install -e '.[dev]'
pytest
```

For the optional visualization demo:

```bash
python3 -m pip install -e '.[demo]'
python3 -m lgc.demo
```

## Quick start

Use integer class labels and mark unlabeled samples with `-1`:

```python
import numpy as np
from lgc import lgc, predict

scores = lgc(
    X,
    np.array([0, -1, 1, -1]),
    sigma=0.2,
    alpha=0.99,
    tol=1e-6,
)
classes = predict(scores)
```

For larger datasets, use a sparse k-nearest-neighbor graph:

```python
scores = lgc(X, labels, sigma=0.2, n_neighbors=15)
```

This uses scikit-learn's exact CPU nearest-neighbor search and SciPy sparse
matrices, so no FAISS runtime or platform-specific index is required. The
same mode avoids allocating an all-pairs matrix during prediction.

`lgc()` returns one confidence score per class and sample. The function also
accepts a one-hot label-confidence matrix, where unlabeled rows contain only
zeros.

## scikit-learn

Use `LGCClassifier` with `Pipeline`, `clone`, cross-validation helpers, and
the usual classifier methods:

```python
from lgc import LGCClassifier

model = LGCClassifier(sigma=0.2, alpha=0.99)
model.fit(X, y)                 # use -1 in y for unlabeled rows
labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

Set `n_neighbors` on `LGCClassifier` to use the sparse graph mode:

```python
model = LGCClassifier(sigma=0.2, n_neighbors=15).fit(X, y)
```

The estimator stores the training graph and applies the fitted class scores
to new samples through their affinities to that graph. `-1` is reserved as
the unlabeled marker, so it cannot be used as a class label.

## API

- `build_affinity(X, sigma)` builds a Gaussian affinity matrix.
- `build_knn_affinity(X, sigma, n_neighbors)` builds a sparse Gaussian graph.
- `normalize_affinity(W)` computes `D^-1/2 W D^-1/2`.
- `propagate(S, Y, alpha, max_iter, tol)` runs label propagation.
- `lgc(X, labels, ...)` runs the complete pipeline.
- `predict(scores)` converts scores to class ids.

## Design notes

- The Gaussian kernel uses squared Euclidean distance.
- Affinity construction is vectorized through SciPy/NumPy; no Python callback
  runs once per pair of samples.
- Isolated samples are handled without division by zero.
- Propagation stops early when scores converge.
- Dense affinity is `O(n²)` in memory. Sparse k-NN affinity uses
  `O(n * n_neighbors)` stored edges and sparse matrix multiplication.
- The k-NN graph is symmetrized by taking the maximum of reciprocal edge
  weights.

The original exploratory notebook is retained as `lgc.ipynb`.
