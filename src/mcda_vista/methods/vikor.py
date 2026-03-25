from __future__ import annotations

from typing import Any

import numpy as np
from pyDecision.algorithm import vikor_method

from mcda_vista.converters import relation_from_ranks
from mcda_vista.methods import handle_pydecision_warnings, register_method
from mcda_vista.relation import Relation


@register_method("vikor")
class VIKORAdapter:
    name: str = "vikor"
    display_name: str = "VIKOR"

    def evaluate(
        self,
        dataset: np.ndarray,
        weights: np.ndarray,
        *,
        v: float = 0.50,
        **_kw: Any,
    ) -> Relation:
        return self._run(dataset, weights, v)

    @staticmethod
    @handle_pydecision_warnings
    def _run(
        dataset: np.ndarray,
        weights: np.ndarray,
        v: float,
    ) -> Relation:
        n = dataset.shape[1]
        c_types = ["max"] * n
        s, r, q, x = vikor_method(
            dataset, weights, c_types, strategy_coefficient=v,
            graph=False, verbose=False,
        )
        return relation_from_ranks(q[0][0], q[1][0])

    def default_params(self) -> dict[str, Any]:
        return {"v": 0.50}

    def param_space(self) -> dict[str, dict]:
        return {
            "v": {
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "step": 0.25,
                "label": "v (strategy coefficient)",
            },
        }
