"""Tests for VistaGenerator, VistaResult, and generate_vista."""
from __future__ import annotations

import numpy as np
import pytest

from mcda_vista.core import VistaGenerator, VistaResult, generate_vista
from mcda_vista.relation import Relation


# ── helpers ──────────────────────────────────────────────────────────────


def _dummy_always_better(dataset: np.ndarray, weights: np.ndarray, **kw) -> Relation:
    """Trivial MCDA method that always returns BETTER."""
    return Relation.BETTER


def _dummy_always_worse(dataset: np.ndarray, weights: np.ndarray, **kw) -> Relation:
    return Relation.WORSE


# ── VistaGenerator basic usage ───────────────────────────────────────────


class TestVistaGeneratorBasic:
    def test_generate_returns_vista_result(self):
        gen = VistaGenerator(method=_dummy_always_better, n_criteria=2, resolution=11)
        result = gen.generate(progress=False)
        assert isinstance(result, VistaResult)

    def test_grid_shape(self):
        gen = VistaGenerator(method=_dummy_always_better, n_criteria=2, resolution=11)
        result = gen.generate(progress=False)
        assert result.grid.shape == (121, 2)

    def test_relations_shape(self):
        gen = VistaGenerator(method=_dummy_always_better, n_criteria=2, resolution=11)
        result = gen.generate(progress=False)
        assert result.relations.shape == (121,)

    def test_all_relations_better(self):
        gen = VistaGenerator(method=_dummy_always_better, n_criteria=2, resolution=11)
        result = gen.generate(progress=False)
        assert all(r == Relation.BETTER for r in result.relations)

    def test_default_reference(self):
        gen = VistaGenerator(method=_dummy_always_better, n_criteria=2, resolution=11)
        result = gen.generate(progress=False)
        np.testing.assert_array_almost_equal(result.reference, [0.5, 0.5])

    def test_metadata_has_elapsed(self):
        gen = VistaGenerator(method=_dummy_always_better, n_criteria=2, resolution=5)
        result = gen.generate(progress=False)
        assert "elapsed_seconds" in result.metadata
        assert result.metadata["elapsed_seconds"] >= 0

    def test_custom_reference(self):
        gen = VistaGenerator(
            method=_dummy_always_better,
            n_criteria=2,
            resolution=5,
            reference=[0.3, 0.7],
        )
        result = gen.generate(progress=False)
        np.testing.assert_array_almost_equal(result.reference, [0.3, 0.7])

    def test_custom_weights(self):
        gen = VistaGenerator(
            method=_dummy_always_better,
            n_criteria=2,
            resolution=5,
            weights=[2.0, 3.0],
        )
        result = gen.generate(progress=False)
        np.testing.assert_array_almost_equal(result.weights, [2.0, 3.0])


# ── VistaGenerator validation ───────────────────────────────────────────


class TestVistaGeneratorValidation:
    def test_n_criteria_4_raises(self):
        with pytest.raises(ValueError, match="n_criteria must be 2 or 3"):
            VistaGenerator(method=_dummy_always_better, n_criteria=4)

    def test_n_criteria_1_raises(self):
        with pytest.raises(ValueError, match="n_criteria must be 2 or 3"):
            VistaGenerator(method=_dummy_always_better, n_criteria=1)

    def test_resolution_1_raises(self):
        with pytest.raises(ValueError, match="resolution must be >= 2"):
            VistaGenerator(method=_dummy_always_better, n_criteria=2, resolution=1)

    def test_reference_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="reference has shape"):
            VistaGenerator(
                method=_dummy_always_better,
                n_criteria=2,
                reference=[0.5, 0.5, 0.5],
            )

    def test_weights_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="weights has shape"):
            VistaGenerator(
                method=_dummy_always_better,
                n_criteria=2,
                weights=[1.0],
            )


# ── 3-criteria support ──────────────────────────────────────────────────


class TestVistaGenerator3D:
    def test_3_criteria_grid(self):
        gen = VistaGenerator(method=_dummy_always_better, n_criteria=3, resolution=5)
        result = gen.generate(progress=False)
        assert result.grid.shape == (125, 3)
        assert result.relations.shape == (125,)
        assert result.n_criteria == 3


# ── third_alternative ───────────────────────────────────────────────────


class TestVistaThirdAlternative:
    def test_third_alternative_is_set(self):
        gen = VistaGenerator(method=_dummy_always_better, n_criteria=2, resolution=5)
        result = gen.generate(third_alternative=[0.2, 0.8], progress=False)
        assert result.third_alternative is not None
        np.testing.assert_array_almost_equal(result.third_alternative, [0.2, 0.8])

    def test_no_third_alternative_is_none(self):
        gen = VistaGenerator(method=_dummy_always_better, n_criteria=2, resolution=5)
        result = gen.generate(progress=False)
        assert result.third_alternative is None


# ── generate_vista convenience function ─────────────────────────────────


class TestGenerateVista:
    def test_convenience_function(self):
        result = generate_vista(
            method=_dummy_always_better,
            n_criteria=2,
            resolution=5,
            progress=False,
        )
        assert isinstance(result, VistaResult)
        assert result.grid.shape == (25, 2)

    def test_with_third_alternative(self):
        result = generate_vista(
            method=_dummy_always_better,
            n_criteria=2,
            resolution=5,
            third_alternative=[0.1, 0.9],
            progress=False,
        )
        assert result.third_alternative is not None

    def test_method_name_from_callable(self):
        result = generate_vista(
            method=_dummy_always_better,
            n_criteria=2,
            resolution=5,
            progress=False,
        )
        assert result.method_name == "_dummy_always_better"
