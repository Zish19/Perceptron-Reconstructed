"""Reproducible experiment runner for shape-discrimination tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .metrics import accuracy_score, confusion_matrix
from .model import PerceptronConfig, RosenblattPerceptron
from .stimuli import make_shape_dataset


@dataclass(frozen=True)
class ExperimentResult:
    """Model, learning history, test metrics, and generated data for an experiment."""

    model: RosenblattPerceptron
    history: list[dict[str, float]]
    initial_test_accuracy: float
    final_train_accuracy: float
    final_test_accuracy: float
    test_confusion: NDArray[np.int64]
    X_train: NDArray[np.uint8]
    y_train: NDArray[np.int64]
    X_test: NDArray[np.uint8]
    y_test: NDArray[np.int64]
    seed: int


def run_experiment(
    config: PerceptronConfig | None = None,
    epochs: int = 20,
    train_samples_per_class: int = 100,
    test_samples_per_class: int = 50,
    stimulus_noise: float = 0.02,
    stimulus_jitter: int = 0,
    seed: int = 7,
) -> ExperimentResult:
    """Train and evaluate a historical-model instance on noisy geometric shapes."""
    cfg = config or PerceptronConfig(seed=seed)
    if cfg.n_responses != 2:
        raise ValueError("the starter shape experiment has exactly two classes")
    train = make_shape_dataset(
        samples_per_class=train_samples_per_class,
        size=cfg.retina_size,
        jitter=stimulus_jitter,
        noise=stimulus_noise,
        seed=seed,
    )
    test = make_shape_dataset(
        samples_per_class=test_samples_per_class,
        size=cfg.retina_size,
        jitter=stimulus_jitter,
        noise=stimulus_noise,
        seed=seed + 1,
    )
    model = RosenblattPerceptron(cfg)
    initial_pred = model.predict(test.X)
    initial_accuracy = accuracy_score(test.y, initial_pred)
    history = model.fit(train.X, train.y, epochs=epochs)
    train_pred = model.predict(train.X)
    test_pred = model.predict(test.X)
    return ExperimentResult(
        model=model,
        history=history,
        initial_test_accuracy=initial_accuracy,
        final_train_accuracy=accuracy_score(train.y, train_pred),
        final_test_accuracy=accuracy_score(test.y, test_pred),
        test_confusion=confusion_matrix(test.y, test_pred, n_classes=2),
        X_train=train.X,
        y_train=train.y,
        X_test=test.X,
        y_test=test.y,
        seed=seed,
    )
