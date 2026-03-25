from __future__ import annotations

from typing import Any

import numpy as np
from pyDecision.algorithm import oreste_method

from mcda_vista.converters import relation_from_ranks
from mcda_vista.methods import handle_pydecision_warnings, register_method
from mcda_vista.relation import Relation


@register_method("oreste")
class ORESTEAdapter:
    name: str = "oreste"
    display_name: str = "ORESTE"

    def evaluate(
        self,
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        alpha: float = 0.40,
        **_kw: Any,
    ) -> Relation:
        return self._run(dataset, weights, alpha)

    @staticmethod
    @handle_pydecision_warnings
    def _run(
        dataset: np.ndarray,
        weights: np.ndarray,
        alpha: float,
    ) -> Relation:
        n = dataset.shape[1]
        c_types = ["max"] * n
        ranks = oreste_method(
            dataset, weights, c_types, alpha=alpha,
            graph=False, verbose=False,
        )
        return relation_from_ranks(ranks[0], ranks[1])

    def default_params(self) -> dict[str, Any]:
        return {"alpha": 0.40}

    def param_space(self) -> dict[str, dict]:
        return {
            "alpha": {
                "min": 0.2,
                "max": 0.6,
                "default": 0.4,
                "step": 0.05,
                "label": "α (weighting factor)",
            },
        }
