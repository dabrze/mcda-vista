from __future__ import annotations

from pyDecision.algorithm import saw_method

from mcda_vista.converters import relation_from_aggregates
from mcda_vista.methods import handle_pydecision_warnings, register_method
from mcda_vista.relation import Relation


@register_method("saw")
class SAWAdapter:
    name = "saw"
    display_name = "SAW"

    def evaluate(self, dataset, weights, *, delta=0.10, **_kwargs):
        result = self._evaluate_inner(dataset, weights, delta)
        return result

    @staticmethod
    @handle_pydecision_warnings
    def _evaluate_inner(dataset, weights, delta):
        n = dataset.shape[1]
        c_types = ["max"] * n
        aggreg = saw_method(dataset, c_types, weights, graph=False, verbose=False)
        return relation_from_aggregates(aggreg[0][1], aggreg[1][1], delta)

    def default_params(self):
        return {"delta": 0.10}

    def param_space(self):
        return {
            "delta": {
                "min": 0.0,
                "max": 0.5,
                "default": 0.1,
                "step": 0.05,
                "label": "δ (indifference threshold)",
            },
        }
