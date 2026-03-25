from __future__ import annotations

from typing import Any

import numpy as np
from pyDecision.algorithm import borda_method

from mcda_vista.converters import relation_from_ranks
from mcda_vista.methods import handle_pydecision_warnings, register_method
from mcda_vista.relation import Relation


@register_method("borda")
class BordaAdapter:
    name: str = "borda"
    display_name: str = "Borda"

    def evaluate(
        self,
        dataset: np.ndarray,
        weights: np.ndarray,
        **_kw: Any,
    ) -> Relation:
        return self._run(dataset)

    @staticmethod
    @handle_pydecision_warnings
    def _run(dataset: np.ndarray) -> Relation:
        n = dataset.shape[1]
        c_types = ["max"] * n
        ranks = borda_method(dataset, c_types, graph=False, verbose=False)
        return relation_from_ranks(ranks[1], ranks[0])

    def default_params(self) -> dict[str, Any]:
        return {}

    def param_space(self) -> dict[str, dict]:
        return {}
