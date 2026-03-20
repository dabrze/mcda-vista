"""REGIME method adapter for VISTA.

Uses pyDecision's ``regime_method`` with the VISTA monkey-patch that
skips the ``po_ranking`` Matplotlib visualisation call.

Ported from ``regime_in_pyDecision.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mcda_vista.converters import relation_from_named_no_pminus
from mcda_vista.methods import register_method, handle_pydecision_warnings
# IMPORTANT: import _patches to ensure patches are applied
from mcda_vista.methods import _patches  # noqa: F401
from mcda_vista.relation import Relation

__all__ = ["REGIMEAdapter"]


@register_method("regime")
class REGIMEAdapter:
    """REGIME adapter — pairwise dominance-based outranking."""

    name: str = "regime"
    display_name: str = "REGIME"

    def evaluate(
        self,
        dataset: np.ndarray,
        weights: np.ndarray,
        **_kw: Any,
    ) -> Relation:
        """Compare alternative at *dataset[1]* against reference at *dataset[0]*.

        Parameters
        ----------
        dataset : np.ndarray, shape (m, n)
            Decision matrix.
        weights : np.ndarray, shape (n,)
            Criteria weights.
        """
        return self._run(dataset, weights)

    @staticmethod
    @handle_pydecision_warnings
    def _run(dataset: np.ndarray, weights: np.ndarray) -> Relation:
        from pyDecision.algorithm import regime_method

        n = dataset.shape[1]
        c_types = ["max"] * n

        rank = regime_method(dataset, weights, c_types)
        return relation_from_named_no_pminus(rank[1][0])

    def default_params(self) -> dict[str, Any]:
        return {}

    def param_space(self) -> dict[str, dict]:
        return {}
