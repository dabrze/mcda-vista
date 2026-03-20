from __future__ import annotations

from pyDecision.algorithm import vikor_method

from mcda_vista.converters import relation_from_ranks
from mcda_vista.methods import handle_pydecision_warnings, register_method
from mcda_vista.relation import Relation


@register_method("vikor")
class VIKORAdapter:
    name = "vikor"
    display_name = "VIKOR"

    def evaluate(self, dataset, weights, *, v=0.50, **_kwargs):
        result = self._evaluate_inner(dataset, weights, v)
        return result

    @staticmethod
    @handle_pydecision_warnings
    def _evaluate_inner(dataset, weights, v):
        n = dataset.shape[1]
        c_types = ["max"] * n
        s, r, q, x = vikor_method(
            dataset, weights, c_types, strategy_coefficient=v,
            graph=False, verbose=False,
        )
        return relation_from_ranks(q[0][0], q[1][0])

    def default_params(self):
        return {"v": 0.50}

    def param_space(self):
        return {
            "v": {
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "step": 0.25,
                "label": "v (strategy coefficient)",
            },
        }
