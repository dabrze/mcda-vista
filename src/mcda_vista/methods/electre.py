"""ELECTRE III and ELECTRE IIIc method adapters for VISTA.

Uses pyDecision's ``electre_iii`` with VISTA monkey-patches that
parameterise the α / β distillation thresholds.

Ported from ``electre_in_pyDecision.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mcda_vista.converters import relation_from_credibilities, relation_from_named
from mcda_vista.methods import register_method, handle_pydecision_warnings
# IMPORTANT: import _patches to ensure patches are applied
from mcda_vista.methods import _patches  # noqa: F401
from mcda_vista.relation import Relation

__all__ = ["ELECTREIIIAdapter", "ELECTREIIIcAdapter"]


# ---------------------------------------------------------------------------
# ELECTRE III
# ---------------------------------------------------------------------------


@register_method("electre_iii")
class ELECTREIIIAdapter:
    """ELECTRE III adapter — distillation-based partial pre-order."""

    name: str = "electre_iii"
    display_name: str = "ELECTRE III"

    def evaluate(
        self,
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        q: float = 0.10,
        p: float = 0.20,
        v: float = 1.00,
        alpha: float = -0.15,
        beta: float = 0.30,
        **_kw: Any,
    ) -> Relation:
        """Compare alternative at *dataset[1]* against reference at *dataset[0]*.

        Parameters
        ----------
        dataset : np.ndarray, shape (m, n)
            Decision matrix.
        weights : np.ndarray, shape (n,)
            Criteria weights.
        q, p, v : float
            Indifference, preference, and veto thresholds (uniform across
            all criteria).
        alpha, beta : float
            Distillation cut-level parameters: λ_s = alpha · λ_max + beta.
        """
        return self._run(dataset, weights, q=q, p=p, v=v, alpha=alpha, beta=beta)

    @staticmethod
    @handle_pydecision_warnings
    def _run(
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        q: float,
        p: float,
        v: float,
        alpha: float,
        beta: float,
    ) -> Relation:
        from pyDecision.algorithm import electre_iii

        n = dataset.shape[1]
        Q = [q] * n
        P = [p] * n
        V = [v] * n

        _, _, _, _, _, rank_P = electre_iii(
            dataset, P=P, Q=Q, V=V, W=weights, alpha=alpha, beta=beta, graph=False
        )
        return relation_from_named(rank_P[1][0])

    def default_params(self) -> dict[str, Any]:
        return {"q": 0.10, "p": 0.20, "v": 1.00, "alpha": -0.15, "beta": 0.30}

    def param_space(self) -> dict[str, dict]:
        return {
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
            "v": {
                "min": 0.1,
                "max": 2.0,
                "default": 1.00,
                "step": 0.10,
                "label": "v (veto)",
            },
            "alpha": {
                "min": -0.50,
                "max": 0.0,
                "default": -0.15,
                "step": 0.05,
                "label": "α (cut-level slope)",
            },
            "beta": {
                "min": 0.0,
                "max": 1.0,
                "default": 0.30,
                "step": 0.05,
                "label": "β (cut-level intercept)",
            },
        }


# ---------------------------------------------------------------------------
# ELECTRE IIIc (credibility-based)
# ---------------------------------------------------------------------------


@register_method("electre_iiic")
class ELECTREIIIcAdapter:
    """ELECTRE IIIc adapter — credibility-based outranking relation."""

    name: str = "electre_iiic"
    display_name: str = "ELECTRE IIIc"

    def evaluate(
        self,
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        q: float = 0.10,
        p: float = 0.20,
        v: float = 1.00,
        lamb: float = 0.60,
        **_kw: Any,
    ) -> Relation:
        """Compare alternative at *dataset[1]* against reference at *dataset[0]*.

        Parameters
        ----------
        dataset : np.ndarray, shape (m, n)
            Decision matrix.
        weights : np.ndarray, shape (n,)
            Criteria weights.
        q, p, v : float
            Indifference, preference, and veto thresholds (uniform across
            all criteria).
        lamb : float
            Credibility cutting level (λ).
        """
        return self._run(dataset, weights, q=q, p=p, v=v, lamb=lamb)

    @staticmethod
    @handle_pydecision_warnings
    def _run(
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        q: float,
        p: float,
        v: float,
        lamb: float,
    ) -> Relation:
        from pyDecision.algorithm import electre_iii

        n = dataset.shape[1]
        Q = [q] * n
        P = [p] * n
        V = [v] * n

        _, credibility, _, _, _, _ = electre_iii(
            dataset, P=P, Q=Q, V=V, W=weights, alpha=-0.15, beta=0.30, graph=False
        )
        return relation_from_credibilities(credibility[1][0], credibility[0][1], lamb)

    def default_params(self) -> dict[str, Any]:
        return {"q": 0.10, "p": 0.20, "v": 1.00, "lamb": 0.60}

    def param_space(self) -> dict[str, dict]:
        return {
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
            "v": {
                "min": 0.1,
                "max": 2.0,
                "default": 1.00,
                "step": 0.10,
                "label": "v (veto)",
            },
            "lamb": {
                "min": 0.0,
                "max": 1.0,
                "default": 0.60,
                "step": 0.05,
                "label": "λ (cutting level)",
            },
        }
