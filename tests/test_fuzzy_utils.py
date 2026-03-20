"""Tests for the fuzzy_utils module."""
from __future__ import annotations

import numpy as np
import pytest

from mcda_vista.fuzzy_utils import (
    fuzzify_value,
    fuzzify_matrix,
    fuzzify_weights,
    fuzzy_topsis,
    fuzzy_vikor,
    fuzzy_moora,
    fuzzy_waspas,
)
from mcda_vista.relation import Relation

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

DATASET_EQUAL = np.array([[0.5, 0.5], [0.5, 0.5]])
DATASET_BETTER = np.array([[0.5, 0.5], [0.9, 0.9]])
DATASET_WORSE = np.array([[0.5, 0.5], [0.1, 0.1]])
WEIGHTS = np.array([1.0, 1.0])


# ---------------------------------------------------------------------------
# fuzzify_value
# ---------------------------------------------------------------------------


class TestFuzzifyValue:
    """Tests for fuzzify_value."""

    def test_fuzzify_value_zero_spread(self):
        """spread=0 returns (x, x, x)."""
        a, b, c = fuzzify_value(0.5, spread=0.0)
        assert (a, b, c) == (0.5, 0.5, 0.5)

    def test_fuzzify_value_symmetric(self):
        """spread=0.1, skew=0 gives symmetric triangle around x."""
        a, b, c = fuzzify_value(0.5, spread=0.1, skew=0.0)
        assert a == pytest.approx(0.4)
        assert b == pytest.approx(0.5)
        assert c == pytest.approx(0.6)

    def test_fuzzify_value_clamping_low(self):
        """Left bound is clamped to lo when x - spread < lo."""
        a, b, c = fuzzify_value(0.05, spread=0.2)
        assert a == pytest.approx(0.0)
        assert b == pytest.approx(0.05)
        assert c == pytest.approx(0.25)

    def test_fuzzify_value_clamping_high(self):
        """Right bound is clamped to hi when x + spread > hi."""
        a, b, c = fuzzify_value(0.95, spread=0.2)
        assert a == pytest.approx(0.75)
        assert b == pytest.approx(0.95)
        assert c == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "x, spread, skew",
        [
            (0.5, 0.1, 0.0),
            (0.0, 0.2, 0.0),
            (1.0, 0.2, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, -0.5),
            (0.1, 0.3, 0.8),
            (0.9, 0.3, -0.8),
            (0.5, 0.0, 0.0),
        ],
    )
    def test_fuzzify_value_ordering(self, x, spread, skew):
        """a <= b <= c for various inputs."""
        a, b, c = fuzzify_value(x, spread, skew)
        assert a <= b <= c

    def test_fuzzify_value_positive_skew(self):
        """skew > 0 makes the right arm wider than the left arm."""
        a, b, c = fuzzify_value(0.5, spread=0.2, skew=0.5)
        left_arm = b - a
        right_arm = c - b
        assert right_arm > left_arm

    def test_fuzzify_value_negative_skew(self):
        """skew < 0 makes the left arm wider than the right arm."""
        a, b, c = fuzzify_value(0.5, spread=0.2, skew=-0.5)
        left_arm = b - a
        right_arm = c - b
        assert left_arm > right_arm


# ---------------------------------------------------------------------------
# fuzzify_matrix
# ---------------------------------------------------------------------------


class TestFuzzifyMatrix:
    """Tests for fuzzify_matrix."""

    def test_fuzzify_matrix_shape(self):
        """Output has correct dimensions and each element is a 3-tuple."""
        data = np.array([[0.2, 0.8], [0.5, 0.6], [0.3, 0.9]])
        result = fuzzify_matrix(data, spread=0.1)

        assert isinstance(result, list)
        assert len(result) == 3
        for row in result:
            assert isinstance(row, list)
            assert len(row) == 2
            for elem in row:
                assert isinstance(elem, tuple)
                assert len(elem) == 3


# ---------------------------------------------------------------------------
# fuzzify_weights
# ---------------------------------------------------------------------------


class TestFuzzifyWeights:
    """Tests for fuzzify_weights."""

    def test_fuzzify_weights_format(self):
        """Output is [[(wa,wb,wc), ...]] single-row nested list."""
        w = np.array([1.0, 2.0, 3.0])
        result = fuzzify_weights(w, spread=0.1)

        assert isinstance(result, list)
        assert len(result) == 1
        inner = result[0]
        assert isinstance(inner, list)
        assert len(inner) == 3
        for elem in inner:
            assert isinstance(elem, tuple)
            assert len(elem) == 3


# ---------------------------------------------------------------------------
# Fuzzy MCDA method wrappers
# ---------------------------------------------------------------------------

_VALID_RELATIONS = set(Relation)


class TestFuzzyTopsis:
    """Tests for fuzzy_topsis."""

    def test_fuzzy_topsis_returns_relation(self):
        """Returns a valid Relation value."""
        result = fuzzy_topsis(
            DATASET_BETTER, WEIGHTS, spread=0.1, skew=0.0, delta=0.1,
        )
        assert result in _VALID_RELATIONS

    def test_fuzzy_topsis_discriminates(self):
        """Clearly better test point returns BETTER."""
        result = fuzzy_topsis(
            DATASET_BETTER, WEIGHTS, spread=0.05, skew=0.0, delta=0.1,
        )
        assert result == Relation.BETTER

    def test_fuzzy_topsis_symmetric_indifference(self):
        """Nearly equal alternatives return INDIFFERENT with generous delta."""
        # Exactly equal data triggers normalization warnings in pyDecision,
        # so we use slightly different values with a generous delta instead.
        dataset = np.array([[0.50, 0.50], [0.52, 0.48]])
        result = fuzzy_topsis(
            dataset, WEIGHTS, spread=0.05, skew=0.0, delta=0.5,
        )
        assert result == Relation.INDIFFERENT


class TestFuzzyVikor:
    """Tests for fuzzy_vikor."""

    def test_fuzzy_vikor_returns_relation(self):
        """Returns a valid Relation value (may be ERROR for degenerate)."""
        result = fuzzy_vikor(
            DATASET_BETTER, WEIGHTS, spread=0.1, skew=0.0, delta=0.1,
        )
        assert result in _VALID_RELATIONS


class TestFuzzyMoora:
    """Tests for fuzzy_moora."""

    def test_fuzzy_moora_returns_relation(self):
        """Returns a valid Relation value."""
        result = fuzzy_moora(
            DATASET_BETTER, WEIGHTS, spread=0.1, skew=0.0, delta=0.1,
        )
        assert result in _VALID_RELATIONS


class TestFuzzyWaspas:
    """Tests for fuzzy_waspas."""

    def test_fuzzy_waspas_returns_relation(self):
        """Returns a valid Relation value."""
        result = fuzzy_waspas(
            DATASET_BETTER, WEIGHTS, spread=0.1, skew=0.0, delta=0.1,
        )
        assert result in _VALID_RELATIONS
