from __future__ import annotations

from pyDecision.algorithm import borda_method

from mcda_vista.converters import relation_from_ranks
from mcda_vista.methods import handle_pydecision_warnings, register_method
from mcda_vista.relation import Relation


@register_method("borda")
class BordaAdapter:
    name = "borda"
    display_name = "Borda"

    def evaluate(self, dataset, weights, **_kwargs):
        result = self._evaluate_inner(dataset)
        return result

    @staticmethod
    @handle_pydecision_warnings
    def _evaluate_inner(dataset):
        n = dataset.shape[1]
        c_types = ["max"] * n
        ranks = borda_method(dataset, c_types, graph=False, verbose=False)
        return relation_from_ranks(ranks[1], ranks[0])

    def default_params(self):
        return {}

    def param_space(self):
        return {}
