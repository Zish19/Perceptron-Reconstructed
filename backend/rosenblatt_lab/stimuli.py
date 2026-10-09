"""Synthetic visual stimuli for repeatable perceptron experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ShapeLabel = Literal["circle", "square"]


@dataclass(frozen=True)
class StimulusDataset:
    """Binary retinal patterns and integer class labels."""

    X: NDArray[np.uint8]
    y: NDArray[np.int64]
    image_size: int
    class_names: tuple[str, ...] = ("circle", "square")

    def images(self) -> NDArray[np.uint8]:
        return self.X.reshape(-1, self.image_size, self.image_size)


def make_shape_stimulus(
    kind: ShapeLabel,
    size: int = 16,
    jitter: int = 2,
    noise: float = 0.02,
    rng: np.random.Generator | None = None,
) -> NDArray[np.uint8]:
    """Create a noisy, shifted circle or square on a binary retina."""
    if size < 8:
        raise ValueError("size must be at least 8")
    if jitter < 0 or jitter > size // 4:
        raise ValueError("jitter must be between 0 and size // 4")
    if not 0.0 <= noise < 1.0:
        raise ValueError("noise must be in [0, 1)")
    if kind not in ("circle", "square"):
        raise ValueError("kind must be 'circle' or 'square'")

    generator = rng or np.random.default_rng()
    base = (size - 1) / 2
    cx = base + int(generator.integers(-jitter, jitter + 1))
    cy = base + int(generator.integers(-jitter, jitter + 1))
    radius = float(generator.choice([size * 0.20, size * 0.23, size * 0.25]))
    y_grid, x_grid = np.indices((size, size))

    if kind == "circle":
        image = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius**2).astype(np.uint8)
    else:
        half_width = int(round(radius * 0.85))
        image = (
            (np.abs(x_grid - cx) <= half_width)
            & (np.abs(y_grid - cy) <= half_width)
        ).astype(np.uint8)

    if noise:
        flip_mask = generator.random((size, size)) < noise
        image[flip_mask] = 1 - image[flip_mask]
    return image


def make_shape_dataset(
    samples_per_class: int = 100,
    size: int = 16,
    jitter: int = 2,
    noise: float = 0.02,
    seed: int = 0,
) -> StimulusDataset:
    """Generate a balanced circle-versus-square dataset.

    Labels are 0 for circles and 1 for squares. Each sample is flattened to a
    binary vector, matching the S-point input expected by the model.
    """
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1")
    rng = np.random.default_rng(seed)
    total = samples_per_class * 2
    X = np.empty((total, size * size), dtype=np.uint8)
    y = np.empty(total, dtype=np.int64)
    index = 0
    for label, kind in enumerate(("circle", "square")):
        for _ in range(samples_per_class):
            X[index] = make_shape_stimulus(kind, size, jitter, noise, rng).reshape(-1)
            y[index] = label
            index += 1
    order = rng.permutation(total)
    return StimulusDataset(X=X[order], y=y[order], image_size=size)
