"""Tests for piecewise-linear value (utility) function helpers."""
from __future__ import annotations

import pytest

from mcda_vista.value_functions import (
    generate_breakpoints,
    generate_values_from_01,
    utility_of_value,
    verify_breakpoints,
    weighted_utility_of_value,
)


# ── verify_breakpoints ──────────────────────────────────────────────────


class TestVerifyBreakpoints:
    def test_valid_linear(self):
        result = verify_breakpoints([(0, 0), (1, 1)])
        assert result == [(0, 0), (1, 1)]

    def test_valid_three_points(self):
        result = verify_breakpoints([(0, 0), (0.5, 0.5), (1, 1)])
        assert len(result) == 3

    def test_too_few_breakpoints(self):
        with pytest.raises(ValueError, match="At least 2"):
            verify_breakpoints([(0, 0)])

    def test_first_x_not_zero(self):
        with pytest.raises(ValueError, match="x-coordinate"):
            verify_breakpoints([(0.5, 0), (1, 1)])

    def test_last_x_not_one(self):
        with pytest.raises(ValueError, match="x-coordinate"):
            verify_breakpoints([(0, 0), (0.8, 1)])

    def test_first_y_not_zero(self):
        with pytest.raises(ValueError, match="y-coordinate"):
            verify_breakpoints([(0, 0.5), (1, 1)])

    def test_last_y_not_one(self):
        with pytest.raises(ValueError, match="y-coordinate"):
            verify_breakpoints([(0, 0), (1, 0.5)])

    def test_x_not_strictly_increasing(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            verify_breakpoints([(0, 0), (0.5, 0.5), (0.5, 0.8), (1, 1)])

    def test_y_decreasing(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            verify_breakpoints([(0, 0), (0.3, 0.9), (0.6, 0.5), (1, 1)])


class TestVerifyBreakpointsEdge:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            verify_breakpoints([])

    def test_non_tuple_elements(self):
        # Tuples of length != 2 should raise
        with pytest.raises(ValueError, match="exactly 2 elements"):
            verify_breakpoints([(0, 0, 0), (1, 1, 1)])


# ── generate_values_from_01 ─────────────────────────────────────────────


class TestGenerateValuesFrom01:
    def test_linear_5_points(self):
        vals = generate_values_from_01(5, ["L"])
        assert len(vals) == 5
        assert pytest.approx(vals) == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_linear_2_points(self):
        vals = generate_values_from_01(2, ["L"])
        assert pytest.approx(vals) == [0.0, 1.0]

    def test_starts_at_zero_ends_at_one(self):
        for code in ["L", "X", "V"]:
            vals = generate_values_from_01(5, [code])
            assert vals[0] == pytest.approx(0.0)
            assert vals[-1] == pytest.approx(1.0)

    def test_random_reproducible_with_seed(self):
        a = generate_values_from_01(5, ["R"], seed=42)
        b = generate_values_from_01(5, ["R"], seed=42)
        assert a == b

    def test_random_different_seeds(self):
        a = generate_values_from_01(5, ["R"], seed=1)
        b = generate_values_from_01(5, ["R"], seed=2)
        assert a != b

    def test_random_sorted(self):
        vals = generate_values_from_01(10, ["R"], seed=99)
        assert vals == sorted(vals)

    def test_k_less_than_2_raises(self):
        with pytest.raises(ValueError, match="K must be >= 2"):
            generate_values_from_01(1, ["L"])

    def test_unknown_form_raises(self):
        with pytest.raises(ValueError, match="Unrecognised form code"):
            generate_values_from_01(5, ["Z"])


# ── generate_breakpoints ────────────────────────────────────────────────


class TestGenerateBreakpoints:
    def test_gain_linear(self):
        bpts = generate_breakpoints("gain", 3, ["L"], ["L"])
        assert bpts == [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]

    def test_gain_5_points(self):
        bpts = generate_breakpoints("gain", 5, ["L"], ["L"])
        assert len(bpts) == 5
        assert bpts[0] == (0.0, 0.0)
        assert bpts[-1] == (1.0, 1.0)

    def test_cost_linear_raises_on_validation(self):
        # Cost inverts y: gain [(0,0),(0.5,0.5),(1,1)] → [(0,1),(0.5,0.5),(1,0)]
        # which violates verify_breakpoints (first y must be 0, last y must be 1).
        with pytest.raises(ValueError):
            generate_breakpoints("cost", 3, ["L"], ["L"])

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            generate_breakpoints("profit", 3, ["L"], ["L"])


# ── utility_of_value ────────────────────────────────────────────────────


class TestUtilityOfValue:
    def test_midpoint_linear(self):
        assert utility_of_value(0.5, [(0, 0), (1, 1)]) == pytest.approx(0.5)

    def test_at_zero(self):
        assert utility_of_value(0.0, [(0, 0), (1, 1)]) == pytest.approx(0.0)

    def test_at_one(self):
        assert utility_of_value(1.0, [(0, 0), (1, 1)]) == pytest.approx(1.0)

    def test_quarter(self):
        assert utility_of_value(0.25, [(0, 0), (1, 1)]) == pytest.approx(0.25)

    def test_piecewise(self):
        bpts = [(0, 0), (0.5, 0.8), (1, 1)]
        # In [0, 0.5]: slope = 0.8/0.5 = 1.6; at x=0.25 → 0.4
        assert utility_of_value(0.25, bpts) == pytest.approx(0.4)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="outside the breakpoint x-range"):
            utility_of_value(1.5, [(0, 0), (1, 1)])

    def test_below_range_raises(self):
        with pytest.raises(ValueError, match="outside the breakpoint x-range"):
            utility_of_value(-0.1, [(0, 0), (1, 1)])


# ── weighted_utility_of_value ───────────────────────────────────────────


class TestWeightedUtilityOfValue:
    def test_weight_multiplied(self):
        result = weighted_utility_of_value(0.5, 2.0, [(0, 0), (1, 1)])
        assert result == pytest.approx(1.0)

    def test_weight_zero(self):
        result = weighted_utility_of_value(0.5, 0.0, [(0, 0), (1, 1)])
        assert result == pytest.approx(0.0)

    def test_weight_one_same_as_utility(self):
        bpts = [(0, 0), (0.5, 0.5), (1, 1)]
        assert weighted_utility_of_value(0.3, 1.0, bpts) == pytest.approx(
            utility_of_value(0.3, bpts)
        )


# ── 'R' form reproducibility ───────────────────────────────────────────


class TestRandomFormReproducibility:
    def test_random_breakpoints_reproducible(self):
        a = generate_breakpoints("gain", 5, ["R"], ["R"], seed=123)
        b = generate_breakpoints("gain", 5, ["R"], ["R"], seed=123)
        assert a == b

    def test_random_breakpoints_differ_with_seed(self):
        a = generate_breakpoints("gain", 5, ["R"], ["R"], seed=1)
        b = generate_breakpoints("gain", 5, ["R"], ["R"], seed=2)
        assert a != b
