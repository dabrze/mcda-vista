"""ASSESS (Keeney–Raiffa multiplicative utility) method adapter for VISTA.

Uses the Keeney–Raiffa multiplicative utility model with piecewise-linear
value functions.  Ported from ``assess_in_pyFile.py`` — custom
implementation, **not** pyDecision.

.. note::
   For form ``'R'`` (random breakpoints), the breakpoints differ on each
   call unless a fixed seed is used or pre-generated breakpoints are
   passed via the *breakpoints* parameter.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mcda_vista.converters import relation_from_aggregates
from mcda_vista.methods import register_method
from mcda_vista.relation import Relation
from mcda_vista.value_functions import (
    Breakpoints,
    generate_breakpoints,
    utility_of_value,
)

__all__ = ["ASSESSAdapter"]


def _keeney_raiffa_utility(
    alternative: np.ndarray,
    n: int,
    kk: np.ndarray | list[float],
    breakpoints: Breakpoints,
) -> float:
    """Compute the Keeney–Raiffa multiplicative utility of one alternative.

    Parameters
    ----------
    alternative : np.ndarray, shape (n,)
        Criterion values for a single alternative.
    n : int
        Number of criteria.
    kk : array-like, shape (n,)
        Scaling constants (used directly, **not** normalised).
    breakpoints : Breakpoints
        Piecewise-linear value-function breakpoints.

    Returns
    -------
    float
        The multiplicative utility value.
    """
    K = (1 - sum(kk)) / math.prod(kk)
    uu = [utility_of_value(float(alternative[j]), breakpoints) for j in range(n)]
    if K != 0.0:
        return (1 / K) * (math.prod(K * kk[i] * uu[i] + 1 for i in range(n)) - 1)
    return sum(kk[i] * uu[i] for i in range(n))


@register_method("assess")
class ASSESSAdapter:
    """ASSESS adapter — Keeney–Raiffa multiplicative utility aggregation."""

    name: str = "assess"
    display_name: str = "ASSESS"

    def evaluate(
        self,
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        k: int = 5,
        form: str | list = "L",
        delta: float = 0.10,
        breakpoints: Breakpoints | None = None,
        **_kw: Any,
    ) -> Relation:
        """Compare alternative at *dataset[1]* against reference at *dataset[0]*.

        Parameters
        ----------
        dataset : np.ndarray, shape (m, n)
            Decision matrix.  Row 0 is the reference, row 1 is the test
            alternative.
        weights : np.ndarray, shape (n,)
            Scaling constants for the Keeney–Raiffa model (used directly,
            **not** normalised).
        k : int, optional
            Number of breakpoints (default 5).
        form : str or list, optional
            Value-function shape descriptor (default ``'L'``).  A single
            character like ``'L'`` is automatically wrapped as ``['L']``;
            a list such as ``['S', 4]`` is used directly.
        delta : float, optional
            Indifference threshold (default 0.10).
        breakpoints : Breakpoints or None, optional
            Pre-generated breakpoints.  When *None* (default), they are
            generated from *k* and *form*.  Passing pre-generated
            breakpoints avoids regeneration — especially important for
            form ``'R'`` where each generation yields different values.
        """
        if breakpoints is None:
            form_list = [form] if isinstance(form, str) else form
            breakpoints = generate_breakpoints("gain", k, ["L"], form_list)

        n = dataset.shape[1]
        kk = [float(w) for w in weights]

        utilities = [
            _keeney_raiffa_utility(dataset[i, :], n, kk, breakpoints)
            for i in range(dataset.shape[0])
        ]

        return relation_from_aggregates(utilities[0], utilities[1], delta)

    def default_params(self) -> dict[str, Any]:
        """Return default method-specific parameters."""
        return {"k": 5, "form": "L", "delta": 0.10}

    def param_space(self) -> dict[str, dict]:
        """Return parameter ranges for sweeps and dashboard sliders."""
        return {
            "k": {
                "min": 3,
                "max": 9,
                "default": 5,
                "step": 2,
                "label": "k (breakpoints)",
            },
            "form": {
                "choices": ["L", "X", "V", "S"],
                "default": "L",
                "label": "Value function form",
            },
            "delta": {
                "min": 0.0,
                "max": 0.5,
                "default": 0.1,
                "step": 0.05,
                "label": "δ",
            },
        }
