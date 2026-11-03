"""Small deterministic LGC visualization demo."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from .core import lgc, predict


def main() -> None:
    X, y = make_moons(n_samples=500, noise=0.1, random_state=0)
    labels = np.full(len(y), -1, dtype=int)
    labels[:10] = y[:10]
    scores = lgc(X, labels, sigma=0.2)
    plt.scatter(X[:, 0], X[:, 1], c=predict(scores), cmap="coolwarm")
    plt.title("LGC semi-supervised classification")
    plt.show()


if __name__ == "__main__":
    main()
