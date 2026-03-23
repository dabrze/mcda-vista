"""Fuzzy MCDA method wrappers for VISTA.

Provides fuzzification utilities and VISTA-compatible wrapper functions
for pyDecision's fuzzy ranking methods. Each wrapper follows the standard
VISTA callable interface: ``(dataset, weights, **params) -> Relation``.

VISTA itself is unchanged — these are simply custom callables that
internally convert crisp inputs to triangular fuzzy numbers, call the
underlying pyDecision fuzzy method, and convert the crisp scores back
to a :class:`~mcda_vista.relation.Relation`.

Typical usage with VISTA::

    from mcda_vista import generate_vista, plot_vista
    from mcda_vista.fuzzy_utils import fuzzy_topsis

    result = generate_vista(fuzzy_topsis, resolution=51, spread=0.10, delta=0.10)
    fig = plot_vista(result)

The fuzzification parameters ``spread`` and ``skew`` control the shape
of the triangular fuzzy numbers built around each crisp value:

- **spread** (float): Half-width of the triangular support.
  ``spread=0`` gives degenerate (crisp) numbers ``(x, x, x)``.
- **skew** (float): Asymmetry factor in ``[-1, 1]``.  ``0`` = symmetric,
  positive = right-skewed (wider upward), negative = left-skewed.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable

import numpy as np

from mcda_vista.relation import Relation
from mcda_vista.converters import relation_from_aggregates

__all__ = [
    "fuzzify_value",
    "fuzzify_matrix",
    "fuzzify_weights",
    "fuzzy_topsis",
    "fuzzy_vikor",
    "fuzzy_moora",
    "fuzzy_waspas",
    "fuzzy_edas",
    "fuzzy_copras",
]


# ---------------------------------------------------------------------------
# Fuzzification helpers
# ---------------------------------------------------------------------------


def fuzzify_value(
    x: float,
    spread: float,
    skew: float = 0.0,
    lo: float = 0.0,
    hi: float = 1.0,
) -> tuple[float, float, float]:
    """Convert a crisp value to a triangular fuzzy number ``(a, b, c)``.

    Parameters
    ----------
    x : float
        Crisp centre value.
    spread : float
        Base half-width.  The left arm is ``spread * (1 - skew)`` and the
        right arm is ``spread * (1 + skew)``.  When ``spread == 0`` the
        result is the degenerate number ``(x, x, x)``.
    skew : float
        Asymmetry factor in ``[-1, 1]``.  ``0`` = symmetric triangle,
        positive = wider to the right, negative = wider to the left.
    lo, hi : float
        Bounds for clamping.

    Returns
    -------
    tuple[float, float, float]
        ``(a, b, c)`` with ``a <= b <= c``.
    """
    left_arm = spread * (1.0 - skew)
    right_arm = spread * (1.0 + skew)
    a = max(lo, x - left_arm)
    c = min(hi, x + right_arm)
    b = max(a, min(c, x))
    return (a, b, c)


def fuzzify_matrix(
    dataset: np.ndarray,
    spread: float,
    skew: float = 0.0,
    lo: float = 0.0,
    hi: float = 1.0,
) -> list[list[tuple[float, float, float]]]:
    """Fuzzify a crisp decision matrix for pyDecision.

    Parameters
    ----------
    dataset : np.ndarray, shape (m, n)
        Crisp decision matrix (alternatives x criteria).
    spread, skew, lo, hi :
        Forwarded to :func:`fuzzify_value`.

    Returns
    -------
    list[list[tuple]]
        Nested list of ``(a, b, c)`` triangular fuzzy numbers,
        in the format expected by pyDecision fuzzy methods.
    """
    m, n = dataset.shape
    fuzzy = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(fuzzify_value(dataset[i, j], spread, skew, lo, hi))
        fuzzy.append(row)
    return fuzzy


def fuzzify_weights(
    weights: np.ndarray,
    spread: float,
    skew: float = 0.0,
    lo: float = 0.0,
    hi: float = float("inf"),
) -> list[list[tuple[float, float, float]]]:
    """Fuzzify a crisp weight vector for pyDecision.

    pyDecision expects fuzzy weights as ``[[(wa, wb, wc), ...]]``
    (a single-row nested list).

    Parameters
    ----------
    weights : np.ndarray, shape (n,)
        Crisp weight vector.
    spread, skew, lo, hi :
        Forwarded to :func:`fuzzify_value`.  Default ``hi`` is unbounded
        since weights are not necessarily in ``[0, 1]``.

    Returns
    -------
    list[list[tuple]]
        ``[[(wa, wb, wc), ...]]`` — single-row nested list.
    """
    row = [fuzzify_value(float(w), spread, skew, lo, hi) for w in weights]
    return [row]


# ---------------------------------------------------------------------------
# Warning handler (mirrors handle_pydecision_warnings from methods.base)
# ---------------------------------------------------------------------------


def _handle_warnings(
    func: Callable[..., Relation],
) -> Callable[..., Relation]:
    """Catch RuntimeWarning from pyDecision and return Relation.ERROR."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Relation:
        with warnings.catch_warnings(record=True) as wrn:
            warnings.simplefilter("always")
            result = func(*args, **kwargs)
            if len(wrn) > 0:
                for w in wrn:
                    if issubclass(w.category, RuntimeWarning):
                        return Relation.ERROR
                    else:
                        raise RuntimeError(
                            f"Unexpected warning in {func.__name__}: {w.message}"
                        )
            return result

    return wrapper


# ---------------------------------------------------------------------------
# VISTA-compatible fuzzy method wrappers
# ---------------------------------------------------------------------------


@_handle_warnings
def fuzzy_topsis(
    dataset: np.ndarray,
    weights: np.ndarray,
    *,
    spread: float = 0.10,
    skew: float = 0.0,
    delta: float = 0.10,
    weight_spread: float | None = None,
    **_kw: Any,
) -> Relation:
    """Fuzzy TOPSIS wrapper for VISTA.

    Parameters
    ----------
    dataset : np.ndarray, shape (m, n)
        Crisp decision matrix from VISTA (row 0 = reference, row 1 = test).
    weights : np.ndarray, shape (n,)
        Crisp weight vector.
    spread : float
        Fuzzification spread for the decision matrix.
    skew : float
        Fuzzification asymmetry.
    delta : float
        Indifference threshold for score comparison.
    weight_spread : float or None
        Fuzzification spread for weights.  Defaults to ``spread`` if *None*.
    """
    from pyDecision.algorithm import fuzzy_topsis_method

    n = dataset.shape[1]
    f_dataset = fuzzify_matrix(dataset, spread, skew)
    f_weights = fuzzify_weights(weights, weight_spread if weight_spread is not None else spread, skew)
    c_types = ["max"] * n

    scores = fuzzy_topsis_method(dataset=f_dataset, weights=f_weights, criterion_type=c_types, graph=False, verbose=False)
    return relation_from_aggregates(scores[0], scores[1], delta)


def fuzzy_vikor(
    dataset: np.ndarray,
    weights: np.ndarray,
    *,
    spread: float = 0.10,
    skew: float = 0.0,
    v: float = 0.5,
    delta: float = 0.10,
    weight_spread: float | None = None,
    **_kw: Any,
) -> Relation:
    """Fuzzy VIKOR wrapper for VISTA.

    Uses the Q-index (compromise ranking) for comparison.  When Q produces
    NaN (common with only 2 alternatives due to 0/0 normalisation) the
    wrapper falls back to the S-index.
    """
    from pyDecision.algorithm import fuzzy_vikor_method

    n = dataset.shape[1]
    f_dataset = fuzzify_matrix(dataset, spread, skew)
    f_weights = fuzzify_weights(weights, weight_spread if weight_spread is not None else spread, skew)
    c_types = ["max"] * n

    with warnings.catch_warnings():
        # VIKOR's Q normalisation divides by (S_worst - S_best) which is
        # often zero with only 2 alternatives — suppress that specific warning.
        warnings.filterwarnings("ignore", message="invalid value", category=RuntimeWarning)
        result = fuzzy_vikor_method(
            dataset=f_dataset, weights=f_weights, criterion_type=c_types,
            strategy_coefficient=v, graph=False, verbose=False,
        )

    # result = (flow_s, flow_r, flow_q, solution)
    # Each flow is a sorted array with columns [1-based alt_index, value].
    # Try Q first; fall back to S if Q contains NaN.
    for flow in (result[2], result[0]):
        scores: dict[int, float] = {}
        for row in flow:
            alt_idx = int(row[0]) - 1
            scores[alt_idx] = float(row[1])
        s0 = scores.get(0, 0.0)
        s1 = scores.get(1, 0.0)
        if not (np.isnan(s0) or np.isnan(s1)):
            # Lower S/Q is better → negate for relation_from_aggregates
            return relation_from_aggregates(-s0, -s1, delta)

    return Relation.ERROR


@_handle_warnings
def fuzzy_moora(
    dataset: np.ndarray,
    weights: np.ndarray,
    *,
    spread: float = 0.10,
    skew: float = 0.0,
    delta: float = 0.10,
    weight_spread: float | None = None,
    **_kw: Any,
) -> Relation:
    """Fuzzy MOORA wrapper for VISTA."""
    from pyDecision.algorithm import fuzzy_moora_method

    n = dataset.shape[1]
    f_dataset = fuzzify_matrix(dataset, spread, skew)
    f_weights = fuzzify_weights(weights, weight_spread if weight_spread is not None else spread, skew)
    c_types = ["max"] * n

    scores = fuzzy_moora_method(dataset=f_dataset, weights=f_weights, criterion_type=c_types, graph=False, verbose=False)
    return relation_from_aggregates(scores[0], scores[1], delta)


@_handle_warnings
def fuzzy_waspas(
    dataset: np.ndarray,
    weights: np.ndarray,
    *,
    spread: float = 0.10,
    skew: float = 0.0,
    delta: float = 0.10,
    weight_spread: float | None = None,
    **_kw: Any,
) -> Relation:
    """Fuzzy WASPAS wrapper for VISTA.

    Uses the combined WASPAS score for comparison.
    """
    from pyDecision.algorithm import fuzzy_waspas_method

    n = dataset.shape[1]
    f_dataset = fuzzify_matrix(dataset, spread, skew)
    f_weights = fuzzify_weights(weights, weight_spread if weight_spread is not None else spread, skew)
    c_types = ["max"] * n

    result = fuzzy_waspas_method(dataset=f_dataset, criterion_type=c_types, weights=f_weights, graph=False)
    # Returns (f_wsm, f_wpm, f_waspas) — use the combined WASPAS scores
    waspas_scores = result[2]
    return relation_from_aggregates(waspas_scores[0], waspas_scores[1], delta)


@_handle_warnings
def fuzzy_edas(
    dataset: np.ndarray,
    weights: np.ndarray,
    *,
    spread: float = 0.10,
    skew: float = 0.0,
    delta: float = 0.10,
    weight_spread: float | None = None,
    **_kw: Any,
) -> Relation:
    """Fuzzy EDAS wrapper for VISTA.

    EDAS (Evaluation based on Distance from Average Solution) ranks
    alternatives by their positive/negative distance to the average.
    """
    from pyDecision.algorithm import fuzzy_edas_method

    n = dataset.shape[1]
    f_dataset = fuzzify_matrix(dataset, spread, skew)
    f_weights = fuzzify_weights(weights, weight_spread if weight_spread is not None else spread, skew)
    c_types = ["max"] * n

    scores = fuzzy_edas_method(dataset=f_dataset, criterion_type=c_types, weights=f_weights, graph=False, verbose=False)
    return relation_from_aggregates(scores[0], scores[1], delta)


@_handle_warnings
def fuzzy_copras(
    dataset: np.ndarray,
    weights: np.ndarray,
    *,
    spread: float = 0.10,
    skew: float = 0.0,
    delta: float = 0.10,
    weight_spread: float | None = None,
    **_kw: Any,
) -> Relation:
    """Fuzzy COPRAS wrapper for VISTA.

    COPRAS (Complex Proportional Assessment) uses proportional ranking
    based on benefit and cost criteria aggregation.
    """
    from pyDecision.algorithm import fuzzy_copras_method

    n = dataset.shape[1]
    f_dataset = fuzzify_matrix(dataset, spread, skew)
    f_weights = fuzzify_weights(weights, weight_spread if weight_spread is not None else spread, skew)
    c_types = ["max"] * n

    scores = fuzzy_copras_method(dataset=f_dataset, weights=f_weights, criterion_type=c_types, graph=False, verbose=False)
    return relation_from_aggregates(scores[0], scores[1], delta)
