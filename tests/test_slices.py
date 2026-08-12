"""Tests for n-dimensional pairwise VISTA slices."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mcda_vista.core import slice_vista
from mcda_vista.plotting import plot_vista, plot_vista_grid
from mcda_vista.relation import Relation


def _sum_method(dataset: np.ndarray, weights: np.ndarray, **kwargs: float) -> Relation:
    return Relation.BETTER if np.dot(dataset[1], weights) > np.dot(dataset[0], weights) else Relation.WORSE


def _context() -> dict:
    return {
        "resolution": 5,
        "reference": [0.5, 0.5, 0.5],
        "weights": [1 / 3, 1 / 3, 1 / 3],
    }


def test_slice_fixes_nonfree_coordinate():
    result = slice_vista(_sum_method, _context(), free=(1, 2), fixed={0: 0.2})
    assert result.grid.shape == (25, 3)
    assert np.all(result.grid[:, 0] == 0.2)
    assert result.metadata["free_indices"] == [1, 2]


def test_slice_rejects_missing_fixed_coordinate():
    with pytest.raises(ValueError, match="fixed"):
        slice_vista(_sum_method, _context(), free=(0, 1), fixed={})


def test_plot_slice_with_nondefault_dimensions():
    result = slice_vista(_sum_method, _context(), free=(1, 2), fixed={0: 0.2})
    fig = plot_vista(result)
    assert fig is not None
    plt.close(fig)


def test_plot_vista_grid_accepts_triangle_extra_marker():
    result = slice_vista(
        _sum_method,
        {**_context(), "extra_alternatives": [[0.3, 0.3, 0.3], [0.8, 0.8, 0.8]]},
        free=(0, 1),
        fixed={2: 0.5},
    )
    fig = plot_vista_grid(
        [[result]], row_labels=["toy"], col_labels=["slice"], extra_marker="triangle"
    )
    assert fig is not None
    plt.close(fig)


def test_plot_vista_rejects_unknown_extra_marker():
    result = slice_vista(_sum_method, _context(), free=(0, 1), fixed={2: 0.5})
    with pytest.raises(ValueError, match="extra_marker"):
        plot_vista(result, extra_marker="star")


def test_slice_veto_blocks_outranking_on_the_hidden_criterion():
    """ELECTRE III's veto must remove BETTER entirely from an extreme slice.

    This is the effect Experiment C's figure exists to show: with v = 0.3 the
    reference's 0.4 advantage on the fixed third criterion vetoes every
    outranking, which no single 2D vista would reveal.
    """
    context = {
        "resolution": 21,
        "reference": [0.5, 0.5, 0.5],
        "weights": [1 / 3, 1 / 3, 1 / 3],
        "method_params": {"q": 0.1, "p": 0.2, "v": 0.3},
    }
    blocked = slice_vista("electre_iii", context, free=(0, 1), fixed={2: 0.1})
    assert not np.any(blocked.relations == Relation.BETTER)

    # At the reference value of the hidden criterion the veto stays silent.
    neutral = slice_vista("electre_iii", context, free=(0, 1), fixed={2: 0.5})
    assert np.any(neutral.relations == Relation.BETTER)


def _relation_at_reference(result) -> Relation:
    centre = np.all(np.isclose(result.grid, [0.5, 0.5, 0.5]), axis=1)
    assert np.any(centre), "slice does not pass through the reference"
    return Relation(int(result.relations[centre][0]))


def test_identical_indifferent_overrides_the_self_comparison():
    """TOPSIS is undefined at a == b; the opt-in flag reports indifference."""
    base = {**_context(), "method_params": {"delta": 0.1}}

    default = slice_vista("topsis", base, free=(0, 1), fixed={2: 0.5})
    assert _relation_at_reference(default) is Relation.ERROR

    overridden = slice_vista(
        "topsis", {**base, "identical_indifferent": True}, free=(0, 1), fixed={2: 0.5},
    )
    assert _relation_at_reference(overridden) is Relation.INDIFFERENT

    # The override must touch nothing else.
    off_reference = ~np.all(np.isclose(default.grid, [0.5, 0.5, 0.5]), axis=1)
    assert np.array_equal(
        default.relations[off_reference], overridden.relations[off_reference]
    )
