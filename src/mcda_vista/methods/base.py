"""MethodAdapter protocol and shared utilities for VISTA method adapters.

Every MCDA method adapter exposes a uniform interface so that the VISTA
engine can call any method without knowing its internals. The
:class:`MethodAdapter` protocol captures that interface.

The :func:`handle_pydecision_warnings` decorator factors out the
boilerplate "catch RuntimeWarning → return Relation.ERROR" pattern
that is common to all pyDecision-backed adapters.

:func:`is_self_comparison` supports the optional VISTA-side
self-indifference override described in
:meth:`mcda_vista.core.VistaGenerator.generate`.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from mcda_vista.relation import Relation

__all__ = ["MethodAdapter", "handle_pydecision_warnings", "is_self_comparison"]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MethodAdapter(Protocol):
    """Protocol that all VISTA method adapters must satisfy."""

    name: str
    """Short identifier, e.g. ``'saw'``, ``'topsis'``."""

    display_name: str
    """Human-readable name, e.g. ``'SAW'``, ``'TOPSIS'``."""

    def evaluate(
        self,
        dataset: np.ndarray,
        weights: np.ndarray,
        **params: Any,
    ) -> Relation:
        """Compare alternative at *dataset[1]* against reference at *dataset[0]*.

        Parameters
        ----------
        dataset : np.ndarray, shape (m, n)
            Decision matrix.  Row 0 is the reference, row 1 is the test point.
        weights : np.ndarray, shape (n,)
            Criteria weights.
        **params
            Method-specific parameters.

        Returns
        -------
        Relation
        """
        ...

    def default_params(self) -> dict[str, Any]:
        """Return default method-specific parameters."""
        ...

    def param_space(self) -> dict[str, dict]:
        """Return parameter ranges for sweeps and dashboard sliders.

        Returns a dict like::

            {
                'delta': {
                    'min': 0.0,
                    'max': 0.5,
                    'default': 0.1,
                    'step': 0.05,
                    'label': 'δ (indifference threshold)',
                },
            }
        """
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_self_comparison(dataset: np.ndarray) -> bool:
    """Return *True* when the test point coincides with the reference.

    Row 0 of *dataset* is the reference and row 1 the test point, so an
    exact match means the method is being asked to compare an
    alternative with itself.  Several pyDecision methods are numerically
    undefined in that situation — TOPSIS, for instance, places both rows
    at the ideal *and* the anti-ideal solution, producing a ``0/0``
    ratio that :func:`handle_pydecision_warnings` maps to
    :attr:`Relation.ERROR`.

    Parameters
    ----------
    dataset : np.ndarray, shape (m, n)
        Decision matrix with the reference in row 0.

    Returns
    -------
    bool
    """
    return bool(np.array_equal(dataset[0], dataset[1]))


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def handle_pydecision_warnings(
    func: Callable[..., Relation],
) -> Callable[..., Relation]:
    """Decorator that catches ``RuntimeWarning`` from pyDecision.

    If any :class:`RuntimeWarning` is emitted during the wrapped call the
    decorator returns :attr:`Relation.ERROR` instead of the normal result.
    Other warning categories trigger a :class:`RuntimeError` so they are
    not silently swallowed.
    """

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
