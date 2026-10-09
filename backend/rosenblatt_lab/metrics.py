"""Metrics for small classification experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def accuracy_score(y_true: NDArray[np.integer], y_pred: NDArray[np.integer]) -> float:
    """Return classification accuracy, validating vector lengths."""
    truth = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if len(truth) == 0:
        raise ValueError("accuracy is undefined for an empty target vector")
    if len(truth) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    return float(np.mean(truth == pred))


def confusion_matrix(
    y_true: NDArray[np.integer], y_pred: NDArray[np.integer], n_classes: int
) -> NDArray[np.int64]:
    """Return an integer confusion matrix with rows=true and columns=predicted."""
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if len(truth) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    if len(truth) and (
        truth.min() < 0 or pred.min() < 0 or truth.max() >= n_classes or pred.max() >= n_classes
    ):
        raise ValueError("class labels must be in [0, n_classes)")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (truth, pred), 1)
    return matrix
