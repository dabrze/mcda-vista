"""Dataset augmentation utilities for VISTA analyses.

Provides functions to expand an alternative dataset with random or
regularly-spaced points, replacing the legacy string-parsing approach in
``dataset_handling.py``.
"""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np

__all__ = ["augment_random", "augment_regular_grid"]


def augment_random(
    dataset: np.ndarray,
    n_points: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Append *n_points* random alternatives sampled uniformly in [0, 1].

    Parameters
    ----------
    dataset : np.ndarray
        Existing alternatives, shape ``(m, n)`` where *n* is the number of
        criteria.
    n_points : int
        Number of random rows to add.
    seed : int | None
        Seed for the random number generator (reproducibility).

    Returns
    -------
    np.ndarray
        Augmented dataset with shape ``(m + n_points, n)``.
    """
    if n_points <= 0:
        return dataset.copy()  # type: ignore[no-any-return]

    rng = np.random.default_rng(seed)
    n_criteria = dataset.shape[1]
    random_rows = rng.random((n_points, n_criteria))
    return np.vstack([dataset, random_rows])


def augment_regular_grid(
    dataset: np.ndarray,
    grid_shape: tuple[int, ...] | Sequence[int],
) -> np.ndarray:
    """Append alternatives placed on a regular grid in [0, 1]^n.

    For each axis *k* the grid positions are ``(i + 1) / (grid_shape[k] + 1)``
    for ``i`` in ``range(grid_shape[k])``, matching the original VISTA
    behaviour.

    Parameters
    ----------
    dataset : np.ndarray
        Existing alternatives, shape ``(m, n)``.
    grid_shape : tuple[int, ...]
        Number of divisions along each criterion axis.  Length must equal the
        number of criteria *n*.  For example ``(3, 3)`` on a 2-criterion
        dataset adds 9 evenly-spaced points.

    Returns
    -------
    np.ndarray
        Augmented dataset with shape ``(m + prod(grid_shape), n)``.

    Raises
    ------
    ValueError
        If ``len(grid_shape)`` does not match the number of criteria.
    """
    grid_shape = tuple(grid_shape)
    n_criteria = dataset.shape[1]

    if len(grid_shape) != n_criteria:
        raise ValueError(
            f"grid_shape has {len(grid_shape)} elements but dataset has "
            f"{n_criteria} criteria"
        )

    if any(s <= 0 for s in grid_shape):
        return dataset.copy()  # type: ignore[no-any-return]

    axes = [
        np.arange(1, s + 1) / (s + 1) for s in grid_shape
    ]
    grid_points = np.array(list(product(*axes)))
    return np.vstack([dataset, grid_points])
