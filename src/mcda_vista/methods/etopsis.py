"""E-TOPSIS method adapter for VISTA.

Custom implementation (not from pyDecision).  Decomposes the weighted
value space into projection and residual components relative to the
weight vector and derives a TOPSIS-like closeness coefficient.

Ported from ``etopsis_in_pyFile.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mcda_vista.converters import relation_from_aggregates
from mcda_vista.methods import register_method, handle_pydecision_warnings
from mcda_vista.relation import Relation

__all__ = ["ETOPSISAdapter"]


def _etopsis_scores(
    dataset: np.ndarray,
    weights: np.ndarray,
    aggregation: str,
    epsilon: float,
) -> tuple[np.ndarray, float]:
    """Compute E-TOPSIS scores and the weight-based divider.

    Returns
    -------
    scores : np.ndarray, shape (m,)
        Aggregated score per alternative.
    s : float
        Divider ``||w|| / mean(w)`` used to rescale the delta threshold.
    """
    n = dataset.shape[1]

    nw = np.linalg.norm(weights)
    mw = np.mean(weights)
    s = nw / mw

    B = weights.reshape((n, 1))
    iBtB = np.linalg.inv(B.T @ B)
    P = B @ iBtB @ B.T

    VS = dataset @ np.diag(weights)
    PS = VS @ P
    RS = VS @ (np.eye(n) - P)

    PSw = np.linalg.norm(PS, axis=1) / nw
    RSw = np.linalg.norm(RS, axis=1) / nw

    WMSD = np.column_stack([PSw, RSw]).astype(np.double)

    # Anti-ideal distance
    WMSD[:, 1] = WMSD[:, 1] / epsilon
    a = np.sqrt(np.sum(WMSD ** 2, axis=1))

    # Ideal distance
    WMSD[:, 0] = 1.0 - PSw
    i = np.sqrt(np.sum(WMSD ** 2, axis=1))

    if aggregation == "A":
        scores = a
    elif aggregation == "I":
        scores = i
    else:  # 'R' — ratio (default)
        scores = a / (a + i)

    return scores, s


@register_method("etopsis")
class ETOPSISAdapter:
    """E-TOPSIS adapter — projection-based closeness coefficient."""

    name: str = "etopsis"
    display_name: str = "E-TOPSIS"

    def evaluate(
        self,
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        aggregation: str = "R",
        epsilon: float = 1.0,
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
        aggregation : str
            Aggregation type: ``'A'`` (anti-ideal), ``'I'`` (ideal),
            or ``'R'`` (ratio, default).
        epsilon : float
            Scaling factor for the residual component (default 1.0).
        delta : float
            Indifference threshold (before rescaling by ``s``).
        """
        return self._run(
            dataset, weights, aggregation=aggregation, epsilon=epsilon, delta=delta
        )

    @staticmethod
    @handle_pydecision_warnings
    def _run(
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        aggregation: str,
        epsilon: float,
        delta: float,
    ) -> Relation:
        scores, s = _etopsis_scores(dataset, weights, aggregation, epsilon)
        return relation_from_aggregates(scores[0], scores[1], delta / s)

    def default_params(self) -> dict[str, Any]:
        return {"aggregation": "R", "epsilon": 1.0, "delta": 0.10}

    def param_space(self) -> dict[str, dict]:
        return {
            "aggregation": {
                "choices": ["A", "I", "R"],
                "default": "R",
                "label": "Aggregation type",
            },
            "epsilon": {
                "min": 0.1,
                "max": 5.0,
                "default": 1.0,
                "step": 0.1,
                "label": "ε (residual scaling)",
            },
            "delta": {
                "min": 0.0,
                "max": 0.5,
                "default": 0.10,
                "step": 0.05,
                "label": "δ (indifference threshold)",
            },
        }
