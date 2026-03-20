from __future__ import annotations

from pyDecision.algorithm import oreste_method

from mcda_vista.converters import relation_from_ranks
from mcda_vista.methods import handle_pydecision_warnings, register_method
from mcda_vista.relation import Relation


@register_method("oreste")
class ORESTEAdapter:
    name = "oreste"
    display_name = "ORESTE"

    def evaluate(self, dataset, weights, *, alpha=0.40, **_kwargs):
        result = self._evaluate_inner(dataset, weights, alpha)
        return result

    @staticmethod
    @handle_pydecision_warnings
    def _evaluate_inner(dataset, weights, alpha):
        n = dataset.shape[1]
        c_types = ["max"] * n
        ranks = oreste_method(
            dataset, weights, c_types, alpha=alpha,
            graph=False, verbose=False,
        )
        return relation_from_ranks(ranks[0], ranks[1])

    def default_params(self):
        return {"alpha": 0.40}

    def param_space(self):
        return {
            "alpha": {
                "min": 0.2,
                "max": 0.6,
                "default": 0.4,
                "step": 0.05,
                "label": "α (weighting factor)",
            },
        }
