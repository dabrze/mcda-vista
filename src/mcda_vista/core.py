"""Central engine for the VISTA library.

Implements the grid-sweep logic using extensible design built around
:class:`VistaGenerator` and the :class:`VistaResult` dataclass.

Typical usage
-------------
>>> from mcda_vista.core import generate_vista
>>> result = generate_vista("topsis", resolution=51, n_criteria=2)
>>> result.grid.shape
(2601, 2)

Or with an explicit method callable::

    from mcda_vista.core import VistaGenerator

    def my_method(dataset, weights, **params):
        ...  # return a Relation

    gen = VistaGenerator(method=my_method, n_criteria=2, resolution=101)
    result = gen.generate()
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from mcda_vista.relation import Relation

__all__ = [
    "VistaResult",
    "VistaGenerator",
    "generate_vista",
]


# ── Result container ────────────────────────────────────────────────────


@dataclass
class VistaResult:
    """Immutable container for a single VISTA grid-sweep run.

    Attributes
    ----------
    grid : np.ndarray
        Coordinates of every test point, shape ``(resolution**n, n)``.
    relations : np.ndarray
        Integer-valued :class:`Relation` outcome for each grid point,
        shape ``(resolution**n,)``.
    reference : np.ndarray
        The reference alternative used during the sweep, shape ``(n,)``.
    weights : np.ndarray
        Criteria weights passed to the MCDA method, shape ``(n,)``.
    method_name : str
        Human-readable identifier of the method (e.g. ``"topsis"``).
    params : dict
        Method-specific keyword arguments forwarded to the callable.
    resolution : int
        Number of sample points per criterion axis.
    n_criteria : int
        Dimensionality of the criterion space.
    third_alternative : np.ndarray | None
        Optional fixed third alternative, shape ``(n,)``, or *None*.
    extra_alternatives : np.ndarray | None
        Additional rows appended to the dataset, shape ``(m, n)``, or
        *None*.
    metadata : dict
        Auxiliary information (e.g. ``elapsed_seconds``).
    """

    grid: np.ndarray
    relations: np.ndarray
    reference: np.ndarray
    weights: np.ndarray
    method_name: str
    params: dict
    resolution: int
    n_criteria: int
    third_alternative: np.ndarray | None = None
    extra_alternatives: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


# ── Generator ───────────────────────────────────────────────────────────


class VistaGenerator:
    """Build and evaluate a VISTA grid-sweep for an arbitrary MCDA method.

    Parameters
    ----------
    method : Callable | str
        Either a callable with signature
        ``(dataset: np.ndarray, weights: np.ndarray, **params) -> Relation``
        or a string key that will be looked up via
        :func:`mcda_vista.methods.get_method`.
    n_criteria : int
        Number of criteria (2 or 3).
    resolution : int
        Number of evenly-spaced sample points per axis.  The total number
        of grid points is ``resolution ** n_criteria``.
    reference : Sequence[float] | None
        Reference alternative.  Defaults to ``[0.5] * n_criteria``.
    weights : Sequence[float] | None
        Criteria weight vector.  Defaults to ``[1.0] * n_criteria``.

    Raises
    ------
    ValueError
        If *n_criteria* is not 2 or 3, or vector lengths are inconsistent.
    """

    def __init__(
        self,
        method: Callable[..., Relation] | str,
        n_criteria: int = 2,
        resolution: int = 101,
        reference: Sequence[float] | None = None,
        weights: Sequence[float] | None = None,
    ) -> None:
        if n_criteria not in (2, 3):
            raise ValueError(f"n_criteria must be 2 or 3, got {n_criteria}")
        if resolution < 2:
            raise ValueError(f"resolution must be >= 2, got {resolution}")

        self.n_criteria = n_criteria
        self.resolution = resolution

        if isinstance(method, str):
            from mcda_vista.methods import get_method

            self._method_name = method
            adapter = get_method(method)
            # MethodAdapter instances expose .evaluate(); wrap so that the
            # internal loop can always use a plain callable.
            if callable(adapter):
                self._method: Callable[..., Relation] = adapter
            else:
                self._method = adapter.evaluate
        else:
            self._method_name = getattr(method, "__name__", repr(method))
            self._method = method

        self.reference = np.asarray(
            reference if reference is not None else [0.5] * n_criteria,
            dtype=np.float64,
        )
        self.weights = np.asarray(
            weights if weights is not None else [1.0] * n_criteria,
            dtype=np.float64,
        )

        if self.reference.shape != (n_criteria,):
            raise ValueError(
                f"reference has shape {self.reference.shape}, "
                f"expected ({n_criteria},)"
            )
        if self.weights.shape != (n_criteria,):
            raise ValueError(
                f"weights has shape {self.weights.shape}, " f"expected ({n_criteria},)"
            )

    def generate(
        self,
        third_alternative: Sequence[float] | None = None,
        extra_alternatives: np.ndarray | None = None,
        progress: bool = True,
        **method_params: Any,
    ) -> VistaResult:
        """Execute the grid sweep and return a :class:`VistaResult`.

        Parameters
        ----------
        third_alternative : Sequence[float] | None
            Optional fixed third alternative appended as row 2 of every
            dataset passed to the method.
        extra_alternatives : np.ndarray | None
            Additional alternatives (shape ``(m, n)``) appended after the
            third alternative.
        progress : bool
            If *True* **and** ``tqdm`` is installed, display a progress bar.
        **method_params
            Forwarded as keyword arguments to the MCDA method callable.

        Returns
        -------
        VistaResult
        """
        n = self.n_criteria
        res = self.resolution

        # ── build the template dataset ───────────────────────────────────
        third_arr: np.ndarray | None = None
        extra_arr: np.ndarray | None = None

        rows: list[np.ndarray] = [self.reference.copy()]  # row 0
        rows.append(np.zeros(n, dtype=np.float64))  # row 1 (placeholder)

        if third_alternative is not None:
            third_arr = np.asarray(third_alternative, dtype=np.float64)
            if third_arr.shape != (n,):
                raise ValueError(
                    f"third_alternative has shape {third_arr.shape}, "
                    f"expected ({n},)"
                )
            rows.append(third_arr)

        if extra_alternatives is not None:
            extra_arr = np.asarray(extra_alternatives, dtype=np.float64)
            if extra_arr.ndim == 1:
                extra_arr = extra_arr.reshape(1, -1)
            if extra_arr.shape[1] != n:
                raise ValueError(
                    f"extra_alternatives has {extra_arr.shape[1]} columns, "
                    f"expected {n}"
                )
            for row in extra_arr:
                rows.append(row)

        dataset = np.vstack(rows)  # shape (num_rows, n)

        # ── generate grid ────────────────────────────────────────────────
        ticks = np.linspace(0.0, 1.0, res)
        total_points = res**n
        grid = np.empty((total_points, n), dtype=np.float64)
        relations = np.empty(total_points, dtype=np.uint8)

        # Build the full grid up-front using itertools.product.
        grid_iter = itertools.product(*(ticks for _ in range(n)))

        # Optional progress bar
        iterator: Any
        try:
            if progress:
                from tqdm import tqdm

                iterator = tqdm(
                    enumerate(grid_iter),
                    total=total_points,
                    desc=f"VISTA ({self._method_name})",
                    unit="pt",
                )
            else:
                iterator = enumerate(grid_iter)
        except ImportError:
            iterator = enumerate(grid_iter)

        # ── sweep ────────────────────────────────────────────────────────
        t_start = time.perf_counter()

        for i, point in iterator:
            dataset[1, :] = point
            grid[i, :] = point
            result = self._method(dataset, self.weights, **method_params)
            relations[i] = int(result)

        elapsed = time.perf_counter() - t_start

        # ── pack result ──────────────────────────────────────────────────
        return VistaResult(
            grid=grid,
            relations=relations,
            reference=self.reference.copy(),
            weights=self.weights.copy(),
            method_name=self._method_name,
            params=method_params,
            resolution=res,
            n_criteria=n,
            third_alternative=third_arr.copy() if third_arr is not None else None,
            extra_alternatives=extra_arr.copy() if extra_arr is not None else None,
            metadata={"elapsed_seconds": elapsed},
        )


# ── Convenience wrapper ─────────────────────────────────────────────────


def generate_vista(
    method: Callable[..., Relation] | str,
    resolution: int = 101,
    reference: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
    n_criteria: int = 2,
    third_alternative: Sequence[float] | None = None,
    extra_alternatives: np.ndarray | None = None,
    progress: bool = True,
    **method_params: Any,
) -> VistaResult:
    """One-call convenience wrapper around :class:`VistaGenerator`.

    Creates a :class:`VistaGenerator`, invokes :meth:`~VistaGenerator.generate`,
    and returns the result — useful for quick experiments and scripts.

    Parameters
    ----------
    method : Callable | str
        MCDA method callable or registered name string.
    resolution : int
        Grid resolution per axis (default 101).
    reference : Sequence[float] | None
        Reference alternative; defaults to ``[0.5]*n_criteria``.
    weights : Sequence[float] | None
        Criteria weight vector; defaults to ``[1.0]*n_criteria``.
    n_criteria : int
        Number of criteria (2 or 3).
    third_alternative : Sequence[float] | None
        Optional fixed third alternative.
    extra_alternatives : np.ndarray | None
        Additional alternatives to include in the dataset.
    progress : bool
        Show a ``tqdm`` progress bar (default *True*).
    **method_params
        Forwarded to the MCDA method callable.

    Returns
    -------
    VistaResult
    """
    gen = VistaGenerator(
        method=method,
        n_criteria=n_criteria,
        resolution=resolution,
        reference=reference,
        weights=weights,
    )
    return gen.generate(
        third_alternative=third_alternative,
        extra_alternatives=extra_alternatives,
        progress=progress,
        **method_params,
    )
