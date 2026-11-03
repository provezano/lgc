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

`lgc()` returns one confidence score per class and sample. The function also
accepts a one-hot label-confidence matrix, where unlabeled rows contain only
zeros.

## API

- `build_affinity(X, sigma)` builds a Gaussian affinity matrix.
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
- The dense affinity matrix is `O(n²)` in memory. For very large datasets,
  use a sparse k-nearest-neighbor graph rather than increasing this matrix.

The original exploratory notebook is retained as `lgc.ipynb`.
