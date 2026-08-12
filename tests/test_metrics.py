"""Tests for numeric VISTA checklist metrics."""

from __future__ import annotations

import numpy as np

from mcda_vista.metrics import quantify_checklist
from mcda_vista.relation import Relation


def _sum_method(dataset: np.ndarray, weights: np.ndarray, **kwargs: float) -> Relation:
    delta = kwargs.get("delta", 0.1)
    difference = float(np.dot(dataset[1], weights) - np.dot(dataset[0], weights))
    if abs(difference) <= delta:
        return Relation.INDIFFERENT
    return Relation.BETTER if difference > 0 else Relation.WORSE


def _indifferent_method(dataset: np.ndarray, weights: np.ndarray, **kwargs: float) -> Relation:
    return Relation.INDIFFERENT


def _incomparable_method(dataset: np.ndarray, weights: np.ndarray, **kwargs: float) -> Relation:
    return Relation.INCOMPARABLE


def _inverted_method(dataset: np.ndarray, weights: np.ndarray, **kwargs: float) -> Relation:
    """Deliberately misorders every dominance cone."""
    difference = float(np.dot(dataset[1], weights) - np.dot(dataset[0], weights))
    return Relation.WORSE if difference > 0 else Relation.BETTER


def _error_at_reference_method(
    dataset: np.ndarray, weights: np.ndarray, **kwargs: float,
) -> Relation:
    """Like :func:`_sum_method`, but numerically undefined at ``a == b``.

    Mirrors TOPSIS, whose vector normalisation degenerates to 0/0 when the
    compared alternatives coincide.
    """
    if np.array_equal(dataset[0], dataset[1]):
        return Relation.ERROR
    difference = float(np.dot(dataset[1], weights) - np.dot(dataset[0], weights))
    return Relation.BETTER if difference > 0 else Relation.WORSE


def _context(**overrides: object) -> dict:
    context = {
        "resolution": 11,
        "reference": [0.5, 0.5],
        "weights": [0.5, 0.5],
        "n_rays": 12,
        "method_params": {"delta": 0.1},
    }
    context.update(overrides)
    return context


def test_sum_method_has_analytic_checklist_basics():
    metrics = quantify_checklist(_sum_method, _context())
    assert metrics["self_indiff"] is True
    # One indifference-to-preference transition in each diagonal direction.
    assert metrics["diag_trans"] == 1
    assert metrics["diag_trans_per_ray"] == [1, 1]
    assert metrics["pct_incomp"] == 0.0
    assert metrics["iia_mean"] == 0.0
    assert metrics["region_denominator"] == 120


def test_sum_method_never_misorders_a_dominance_cone():
    """The additive toy method never reverses a preference inside a cone.

    It does place yellow inside the cones — points within δ of the reference
    still dominate it — but weakening to indifference is tolerated, so the
    violation rate must be exactly zero.
    """
    metrics = quantify_checklist(_sum_method, _context())
    assert metrics["dom_viol"] == 0.0
    assert metrics["dom_viol_count"] == 0


def test_sum_method_rays_cross_at_most_one_boundary():
    """Rays start *at* the reference, so a monotone method crosses once.

    Guards against sampling the raster instead of the method: snapping ray
    samples to grid cells makes boundary-tangent rays flip-flop and inflates
    this count into double digits.
    """
    metrics = quantify_checklist(_sum_method, _context(n_rays=360))
    assert metrics["ray_trans_max"] == 1
    assert metrics["ray_trans_max"] == metrics["ray_trans_max_raw"]
    assert metrics["ray_trans_mean"] <= 1.0


def test_diagonal_is_the_radial_check_over_two_directions():
    """The diagonal angles are two of the evenly spaced radial angles.

    Both checks count the same thing — relation changes moving away from the
    reference — so the radial maximum can never fall below the diagonal one.
    Pins the two to a single implementation.
    """
    for method in (_sum_method, _error_at_reference_method, _indifferent_method):
        metrics = quantify_checklist(method, _context(n_rays=360))
        assert metrics["ray_trans_max"] >= metrics["diag_trans"]
        assert metrics["diag_trans"] == max(metrics["diag_trans_per_ray"])
        assert len(metrics["diag_trans_per_ray"]) == 2


def test_rays_are_evaluated_off_grid():
    """Ray samples must not be restricted to grid coordinates."""
    seen: list[tuple[float, float]] = []

    def recording_method(dataset, weights, **kwargs):
        seen.append((float(dataset[1][0]), float(dataset[1][1])))
        return _sum_method(dataset, weights, **kwargs)

    quantify_checklist(recording_method, _context(resolution=11, n_rays=8))
    ticks = set(np.linspace(0.0, 1.0, 11).round(9))
    assert any(round(x, 9) not in ticks or round(y, 9) not in ticks for x, y in seen)


def test_error_relations_are_not_counted_as_a_colour():
    metrics = quantify_checklist(_error_at_reference_method, _context())
    assert metrics["self_indiff"] is None
    assert metrics["self_relation"] == "ERROR"
    assert metrics["n_error"] == 1
    # Both diagonal rays leave the ERROR reference cell straight into a
    # preference, so there is no colour change to count in either direction.
    assert metrics["diag_trans"] == 0
    assert metrics["diag_trans_per_ray"] == [0, 0]
    assert metrics["diag_trans_raw"] == 1


def test_identical_indifferent_repairs_the_reference_cell():
    metrics = quantify_checklist(
        _error_at_reference_method, _context(identical_indifferent=True),
    )
    assert metrics["self_indiff"] is True
    assert metrics["n_error"] == 0
    assert metrics["diag_trans"] == 1


def test_dominance_tolerates_indifference_and_incomparability():
    """Neither weakening counts against a method."""
    context = {"resolution": 5, "n_rays": 4}
    assert quantify_checklist(_indifferent_method, context)["dom_viol"] == 0.0
    assert quantify_checklist(_incomparable_method, context)["dom_viol"] == 0.0


def test_dominance_counts_outright_reversals():
    """Without this, nothing proves the metric can ever be nonzero."""
    metrics = quantify_checklist(_inverted_method, {"resolution": 5, "n_rays": 4})
    assert metrics["dom_viol"] == 1.0
    assert metrics["dom_viol_count"] == metrics["dom_cone_total"] > 0
