# Learning with Local and Global Consistency

An efficient, tested implementation of the LGC semi-supervised learning
algorithm from Zhou et al., *Learning with Local and Global Consistency*
(NeurIPS 2004).

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Usage

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

The implementation also accepts a one-hot label-confidence matrix, where
unlabeled rows contain only zeros.

## Improvements over the original demo

- Correct Gaussian affinity kernel using squared Euclidean distance.
- Vectorized kernel computation instead of `np.vectorize` and Python callbacks.
- Safe normalization for isolated samples.
- Early stopping based on the maximum score change.
- Reusable functions with input validation and deterministic tests.
- Plotting and the optional `scikit-learn` demo are loaded only when running
  `python lgc.py`; importing the algorithm does not require Matplotlib.

## Tests

```bash
pytest -q
```

The original exploratory notebook is retained as `lgc.ipynb`.
