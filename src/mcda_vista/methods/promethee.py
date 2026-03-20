"""PROMETHEE I, II, and III method adapters for VISTA.

Uses pyDecision's ``promethee_i``, ``promethee_ii``, and ``promethee_iii``
with the VISTA monkey-patch that makes PROMETHEE I return flow vectors.

Ported from ``promethee_in_pyDecision.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mcda_vista.converters import relation_from_aggregates, relation_from_named_no_pminus
from mcda_vista.methods import register_method, handle_pydecision_warnings
# IMPORTANT: import _patches to ensure patches are applied
from mcda_vista.methods import _patches  # noqa: F401
from mcda_vista.relation import Relation

__all__ = ["PROMETHEEIAdapter", "PROMETHEEIIAdapter", "PROMETHEEIIIAdapter"]


# ---------------------------------------------------------------------------
# PROMETHEE I
# ---------------------------------------------------------------------------


@register_method("promethee_i")
class PROMETHEEIAdapter:
    """PROMETHEE I adapter — partial ranking from positive/negative flows."""

    name: str = "promethee_i"
    display_name: str = "PROMETHEE I"

    def evaluate(
        self,
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        f: str = "t4",
        q: float = 0.10,
        p: float = 0.20,
        s: float = 0.0,
        **_kw: Any,
    ) -> Relation:
        """Compare alternative at *dataset[1]* against reference at *dataset[0]*.

        Parameters
        ----------
        dataset : np.ndarray, shape (m, n)
            Decision matrix.
        weights : np.ndarray, shape (n,)
            Criteria weights.
        f : str
            Preference function type (``'t1'``–``'t7'``).
        q, p, s : float
            Indifference, preference, and Gaussian thresholds.
        """
        return self._run(dataset, weights, f=f, q=q, p=p, s=s)

    @staticmethod
    @handle_pydecision_warnings
    def _run(
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        f: str,
        q: float,
        p: float,
        s: float,
    ) -> Relation:
        from pyDecision.algorithm import promethee_i

        n = dataset.shape[1]
        F = [f] * n
        Q = [q] * n
        P = [p] * n
        S = [s] * n

        relMat, _, _ = promethee_i(
            dataset, W=weights, F=F, Q=Q, P=P, S=S, graph=False
        )
        return relation_from_named_no_pminus(relMat[1][0])

    def default_params(self) -> dict[str, Any]:
        return {"f": "t4", "q": 0.10, "p": 0.20, "s": 0.0}

    def param_space(self) -> dict[str, dict]:
        return {
            "f": {
                "choices": ["t1", "t2", "t3", "t4", "t5", "t6", "t7"],
                "default": "t4",
                "label": "Preference function type",
            },
            "q": {
                "min": 0.0,
                "max": 0.5,
                "default": 0.10,
                "step": 0.05,
                "label": "q (indifference)",
            },
            "p": {
                "min": 0.0,
                "max": 1.0,
                "default": 0.20,
                "step": 0.05,
                "label": "p (preference)",
            },
            "s": {
                "min": 0.0,
                "max": 1.0,
                "default": 0.0,
                "step": 0.05,
                "label": "s (Gaussian σ)",
            },
        }


# ---------------------------------------------------------------------------
# PROMETHEE II
# ---------------------------------------------------------------------------


@register_method("promethee_ii")
class PROMETHEEIIAdapter:
    """PROMETHEE II adapter — complete ranking from net flows."""

    name: str = "promethee_ii"
    display_name: str = "PROMETHEE II"

    def evaluate(
        self,
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        f: str = "t4",
        q: float = 0.10,
        p: float = 0.20,
        s: float = 0.0,
        delta: float = 0.10,
        **_kw: Any,
    ) -> Relation:
        """Compare alternative at *dataset[1]* against reference at *dataset[0]*.

        Parameters
        ----------
        dataset : np.ndarray, shape (m, n)
            Decision matrix.
        weights : np.ndarray, shape (n,)
            Criteria weights.
        f : str
            Preference function type (``'t1'``–``'t7'``).
        q, p, s : float
            Indifference, preference, and Gaussian thresholds.
        delta : float
            Indifference threshold for aggregate comparison.
        """
        return self._run(dataset, weights, f=f, q=q, p=p, s=s, delta=delta)

    @staticmethod
    @handle_pydecision_warnings
    def _run(
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        f: str,
        q: float,
        p: float,
        s: float,
        delta: float,
    ) -> Relation:
        from pyDecision.algorithm import promethee_ii

        n = dataset.shape[1]
        F = [f] * n
        Q = [q] * n
        P = [p] * n
        S = [s] * n

        flow = promethee_ii(
            dataset, W=weights, F=F, Q=Q, P=P, S=S, sort=False, topn=0, graph=False
        )
        return relation_from_aggregates(flow[0][1], flow[1][1], delta)

    def default_params(self) -> dict[str, Any]:
        return {"f": "t4", "q": 0.10, "p": 0.20, "s": 0.0, "delta": 0.10}

    def param_space(self) -> dict[str, dict]:
        return {
            "f": {
                "choices": ["t1", "t2", "t3", "t4", "t5", "t6", "t7"],
                "default": "t4",
                "label": "Preference function type",
            },
            "q": {
                "min": 0.0,
                "max": 0.5,
                "default": 0.10,
                "step": 0.05,
                "label": "q (indifference)",
            },
            "p": {
                "min": 0.0,
                "max": 1.0,
                "default": 0.20,
                "step": 0.05,
                "label": "p (preference)",
            },
            "s": {
                "min": 0.0,
                "max": 1.0,
                "default": 0.0,
                "step": 0.05,
                "label": "s (Gaussian σ)",
            },
            "delta": {
                "min": 0.0,
                "max": 0.5,
                "default": 0.10,
                "step": 0.05,
                "label": "δ (indifference threshold)",
            },
        }


# ---------------------------------------------------------------------------
# PROMETHEE III
# ---------------------------------------------------------------------------


@register_method("promethee_iii")
class PROMETHEEIIIAdapter:
    """PROMETHEE III adapter — interval-based partial ranking."""

    name: str = "promethee_iii"
    display_name: str = "PROMETHEE III"

    def evaluate(
        self,
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        f: str = "t4",
        q: float = 0.10,
        p: float = 0.20,
        s: float = 0.0,
        lmbd: float = 0.10,
        **_kw: Any,
    ) -> Relation:
        """Compare alternative at *dataset[1]* against reference at *dataset[0]*.

        Parameters
        ----------
        dataset : np.ndarray, shape (m, n)
            Decision matrix.
        weights : np.ndarray, shape (n,)
            Criteria weights.
        f : str
            Preference function type (``'t1'``–``'t7'``).
        q, p, s : float
            Indifference, preference, and Gaussian thresholds.
        lmbd : float
            λ parameter for PROMETHEE III interval width.
        """
        return self._run(dataset, weights, f=f, q=q, p=p, s=s, lmbd=lmbd)

    @staticmethod
    @handle_pydecision_warnings
    def _run(
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        f: str,
        q: float,
        p: float,
        s: float,
        lmbd: float,
    ) -> Relation:
        from pyDecision.algorithm import promethee_iii

        n = dataset.shape[1]
        F = [f] * n
        Q = [q] * n
        P = [p] * n
        S = [s] * n

        relMat = promethee_iii(
            dataset, W=weights, F=F, Q=Q, P=P, S=S, lmbd=lmbd, graph=False
        )
        return relation_from_named_no_pminus(relMat[1][0])

    def default_params(self) -> dict[str, Any]:
        return {"f": "t4", "q": 0.10, "p": 0.20, "s": 0.0, "lmbd": 0.10}

    def param_space(self) -> dict[str, dict]:
        return {
            "f": {
                "choices": ["t1", "t2", "t3", "t4", "t5", "t6", "t7"],
                "default": "t4",
                "label": "Preference function type",
            },
            "q": {
                "min": 0.0,
                "max": 0.5,
                "default": 0.10,
                "step": 0.05,
                "label": "q (indifference)",
            },
            "p": {
                "min": 0.0,
                "max": 1.0,
                "default": 0.20,
                "step": 0.05,
                "label": "p (preference)",
            },
            "s": {
                "min": 0.0,
                "max": 1.0,
                "default": 0.0,
                "step": 0.05,
                "label": "s (Gaussian σ)",
            },
            "lmbd": {
                "min": 0.0,
                "max": 0.5,
                "default": 0.10,
                "step": 0.05,
                "label": "λ (interval width)",
            },
        }
