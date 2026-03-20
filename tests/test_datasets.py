"""Tests for dataset augmentation utilities."""
from __future__ import annotations

import numpy as np
import pytest

from mcda_vista.datasets import augment_random, augment_regular_grid


# ── augment_random ──────────────────────────────────────────────────────


class TestAugmentRandom:
    def test_output_shape(self):
        base = np.array([[0.5, 0.5]])
        result = augment_random(base, 3, seed=42)
        assert result.shape == (4, 2)

    def test_values_in_unit_interval(self):
        base = np.array([[0.5, 0.5]])
        result = augment_random(base, 3, seed=42)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_original_row_preserved(self):
        base = np.array([[0.5, 0.5]])
        result = augment_random(base, 3, seed=42)
        np.testing.assert_array_equal(result[0], [0.5, 0.5])

    def test_reproducibility_same_seed(self):
        base = np.array([[0.5, 0.5]])
        a = augment_random(base, 3, seed=42)
        b = augment_random(base, 3, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_result(self):
        base = np.array([[0.5, 0.5]])
        a = augment_random(base, 3, seed=42)
        b = augment_random(base, 3, seed=99)
        assert not np.array_equal(a, b)

    def test_zero_points_returns_copy(self):
        base = np.array([[0.5, 0.5]])
        result = augment_random(base, 0, seed=42)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result, base)

    def test_multi_criteria(self):
        base = np.array([[0.1, 0.2, 0.3]])
        result = augment_random(base, 5, seed=7)
        assert result.shape == (6, 3)


# ── augment_regular_grid ────────────────────────────────────────────────


class TestAugmentRegularGrid:
    def test_output_shape(self):
        base = np.array([[0.5, 0.5]])
        result = augment_regular_grid(base, (3, 3))
        # 1 original + 3*3 = 10
        assert result.shape == (10, 2)

    def test_original_row_preserved(self):
        base = np.array([[0.5, 0.5]])
        result = augment_regular_grid(base, (3, 3))
        np.testing.assert_array_equal(result[0], [0.5, 0.5])

    def test_grid_points_positions(self):
        base = np.array([[0.5, 0.5]])
        result = augment_regular_grid(base, (3, 3))
        grid_points = result[1:]  # skip original
        # Expected positions: i/(3+1) for i=1,2,3 → 0.25, 0.5, 0.75
        expected_ticks = np.array([0.25, 0.5, 0.75])
        for row in grid_points:
            assert any(np.isclose(row[0], expected_ticks))
            assert any(np.isclose(row[1], expected_ticks))

    def test_grid_values_in_unit_interval(self):
        base = np.array([[0.5, 0.5]])
        result = augment_regular_grid(base, (5, 5))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_shape_mismatch_raises(self):
        base = np.array([[0.5, 0.5]])  # 2 criteria
        with pytest.raises(ValueError, match="grid_shape has 3 elements"):
            augment_regular_grid(base, (3, 3, 3))

    def test_single_criterion_mismatch_raises(self):
        base = np.array([[0.5, 0.5]])  # 2 criteria
        with pytest.raises(ValueError):
            augment_regular_grid(base, (3,))

    def test_different_grid_shapes(self):
        base = np.array([[0.5, 0.5]])
        result = augment_regular_grid(base, (2, 4))
        # 1 + 2*4 = 9
        assert result.shape == (9, 2)
