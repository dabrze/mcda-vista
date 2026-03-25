from __future__ import annotations

from typing import Any

import numpy as np
from pyDecision.algorithm import topsis_method

from mcda_vista.converters import relation_from_aggregates
from mcda_vista.methods import handle_pydecision_warnings, register_method
from mcda_vista.relation import Relation


@register_method("topsis")
class TOPSISAdapter:
    name: str = "topsis"
    display_name: str = "TOPSIS"

    def evaluate(
        self,
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        delta: float = 0.10,
        **_kw: Any,
    ) -> Relation:
        return self._run(dataset, weights, delta)

    @staticmethod
    @handle_pydecision_warnings
    def _run(
        dataset: np.ndarray,
        weights: np.ndarray,
        delta: float,
    ) -> Relation:
        n = dataset.shape[1]
        c_types = ["max"] * n
        aggreg = topsis_method(
            dataset, weights=weights, criterion_type=c_types,
            graph=False, verbose=False,
        )
        return relation_from_aggregates(aggreg[0], aggreg[1], delta)

    def default_params(self) -> dict[str, Any]:
        return {"delta": 0.10}

    def param_space(self) -> dict[str, dict]:
        return {
            "delta": {
                "min": 0.0,
                "max": 0.5,
                "default": 0.1,
                "step": 0.05,
                "label": "δ (indifference threshold)",
            },
        }
