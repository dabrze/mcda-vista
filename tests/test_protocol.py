"""Tests for the VISTA protocol module."""
from __future__ import annotations

import numpy as np

from mcda_vista.core import VistaResult, generate_vista
from mcda_vista.protocol import (
    CheckResult,
    check_diagonal_preference,
    check_dominance,
    check_preference_ratio,
    check_radial_preference,
    check_self_indifference,
    check_third_alternative_stability,
    plot_protocol_report,
    run_protocol,
)
from mcda_vista.relation import Relation


# ── Test helpers ────────────────────────────────────────────────────────


def _dummy_saw(dataset: np.ndarray, weights: np.ndarray, **kw) -> Relation:
    """Simple SAW-like method: weighted sum comparison with delta threshold."""
    delta = kw.get("delta", 0.10)
    score_ref = np.dot(dataset[0], weights)
    score_test = np.dot(dataset[1], weights)
    diff = score_test - score_ref
    if abs(diff) < delta:
        return Relation.INDIFFERENT
    return Relation.BETTER if diff > 0 else Relation.WORSE


def _dummy_always_better(dataset: np.ndarray, weights: np.ndarray, **kw) -> Relation:
    return Relation.BETTER


def _dummy_always_error(dataset: np.ndarray, weights: np.ndarray, **kw) -> Relation:
    return Relation.ERROR


def _dummy_always_indifferent(dataset: np.ndarray, weights: np.ndarray, **kw) -> Relation:
    return Relation.INDIFFERENT


def _dummy_erratic(dataset: np.ndarray, weights: np.ndarray, **kw) -> Relation:
    """Concentric rings of different relations — multiple radial transitions."""
    point = dataset[1]
    ref = dataset[0]
    dist = np.linalg.norm(point - ref)
    if dist < 0.15:
        return Relation.INDIFFERENT
    elif dist < 0.30:
        return Relation.BETTER
    elif dist < 0.45:
        return Relation.WORSE
    else:
        return Relation.BETTER


def _make_result(method, resolution=11, **kw) -> VistaResult:
    return generate_vista(method, resolution=resolution, progress=False, **kw)


# ── CheckResult dataclass ──────────────────────────────────────────────


class TestCheckResult:
    def test_fields(self):
        cr = CheckResult(name="test", passed=True, message="ok")
        assert cr.name == "test"
        assert cr.passed is True
        assert cr.message == "ok"
        assert cr.detail == {}
        assert cr.vista_results == []


# ── Check 1: Dominance ─────────────────────────────────────────────────


class TestCheckDominance:
    def test_saw_passes(self):
        result = _make_result(_dummy_saw, delta=0.10)
        check = check_dominance(result)
        assert check.passed is True
        assert check.detail["worse_in_dominating"] == 0
        assert check.detail["better_in_dominated"] == 0

    def test_all_better_fails(self):
        result = _make_result(_dummy_always_better)
        check = check_dominance(result)
        assert check.passed is False
        assert check.detail["better_in_dominated"] > 0

    def test_3d_inconclusive(self):
        result = _make_result(_dummy_saw, n_criteria=3, resolution=5, delta=0.10)
        check = check_dominance(result)
        assert check.passed is None


# ── Check 2: Self-indifference ─────────────────────────────────────────


class TestCheckSelfIndifference:
    def test_saw_passes(self):
        result = _make_result(_dummy_saw, resolution=21, delta=0.10)
        check = check_self_indifference(result)
        assert check.passed is True
        assert check.detail["nearest_relation"] == "INDIFFERENT"
        assert check.detail["is_numerical_error"] is False

    def test_all_better_fails(self):
        result = _make_result(_dummy_always_better)
        check = check_self_indifference(result)
        assert check.passed is False
        assert check.detail["is_numerical_error"] is False

    def test_all_indifferent_passes(self):
        result = _make_result(_dummy_always_indifferent)
        check = check_self_indifference(result)
        assert check.passed is True

    def test_error_flags_numerical_issue(self):
        result = _make_result(_dummy_always_error)
        check = check_self_indifference(result)
        assert check.passed is False
        assert check.detail["is_numerical_error"] is True
        assert "numerical" in check.message.lower()


# ── Check 3: Diagonal preference change ────────────────────────────────


class TestCheckDiagonalPreference:
    def test_saw_passes(self):
        result = _make_result(_dummy_saw, resolution=21, delta=0.10)
        check = check_diagonal_preference(result)
        assert check.passed is True
        assert check.detail["upper_transitions"] <= 1
        assert check.detail["lower_transitions"] <= 1

    def test_all_indifferent_passes(self):
        result = _make_result(_dummy_always_indifferent)
        check = check_diagonal_preference(result)
        assert check.passed is True
        assert check.detail["upper_transitions"] == 0
        assert check.detail["lower_transitions"] == 0

    def test_3d_inconclusive(self):
        result = _make_result(_dummy_saw, n_criteria=3, resolution=5, delta=0.10)
        check = check_diagonal_preference(result)
        assert check.passed is None


# ── Check 4: Radial preference change ──────────────────────────────────


class TestCheckRadialPreference:
    def test_saw_passes(self):
        result = _make_result(_dummy_saw, resolution=21, delta=0.10)
        check = check_radial_preference(result, n_rays=12)
        assert check.passed is True
        assert check.detail["worst_transitions"] <= 1

    def test_erratic_fails(self):
        result = _make_result(_dummy_erratic, resolution=21)
        check = check_radial_preference(result, n_rays=12)
        assert check.passed is False
        assert check.detail["worst_transitions"] > 1

    def test_3d_inconclusive(self):
        result = _make_result(_dummy_saw, n_criteria=3, resolution=5, delta=0.10)
        check = check_radial_preference(result)
        assert check.passed is None


# ── Check 5: Preference ratio ─────────────────────────────────────────


class TestCheckPreferenceRatio:
    def test_saw_balanced(self):
        result = _make_result(_dummy_saw, resolution=21, delta=0.10)
        check = check_preference_ratio(result)
        assert check.passed is None  # informational, never pass/fail
        assert check.detail["balanced"] is True
        ratio = check.detail["ratio"]
        assert 0.8 <= ratio <= 1.2

    def test_all_better_imbalanced(self):
        result = _make_result(_dummy_always_better)
        check = check_preference_ratio(result)
        assert check.passed is None  # informational
        assert check.detail["balanced"] is False
        assert check.detail["ratio"] == float("inf")

    def test_custom_bounds(self):
        result = _make_result(_dummy_saw, resolution=21, delta=0.10)
        check = check_preference_ratio(result, ratio_bounds=(0.99, 1.01))
        assert check.passed is None
        assert isinstance(check.detail["balanced"], bool)

    def test_message_contains_percentages(self):
        result = _make_result(_dummy_saw, resolution=21, delta=0.10)
        check = check_preference_ratio(result)
        assert "worse" in check.message.lower()
        assert "better" in check.message.lower()


# ── Check 6: Third alternative stability ───────────────────────────────


class TestCheckThirdAlternativeStability:
    def test_saw_stable(self):
        baseline = _make_result(_dummy_saw, resolution=11, delta=0.10)
        with_third = generate_vista(
            _dummy_saw,
            resolution=11,
            third_alternative=[0.25, 0.75],
            progress=False,
            delta=0.10,
        )
        check = check_third_alternative_stability(baseline, [with_third])
        # SAW doesn't use normalization so third alt shouldn't matter
        assert check.passed is True
        assert check.detail["max_change"] == 0.0

    def test_high_threshold_passes(self):
        baseline = _make_result(_dummy_saw, resolution=11, delta=0.10)
        with_third = _make_result(_dummy_always_better, resolution=11)
        # This comparison is between different methods so relations differ heavily
        check = check_third_alternative_stability(
            baseline, [with_third], threshold=1.0
        )
        assert check.passed is True

    def test_zero_threshold_any_change_fails(self):
        baseline = _make_result(_dummy_saw, resolution=11, delta=0.10)
        with_third = _make_result(_dummy_always_better, resolution=11)
        check = check_third_alternative_stability(
            baseline, [with_third], threshold=0.0
        )
        assert check.passed is False


# ── ProtocolReport ─────────────────────────────────────────────────────


class TestProtocolReport:
    def test_summary_string(self):
        report = run_protocol(
            _dummy_saw, resolution=11, progress=False, delta=0.10
        )
        s = report.summary()
        assert "VISTA Protocol Report" in s
        assert report.method_name in s

    def test_all_passed_property(self):
        report = run_protocol(
            _dummy_saw, resolution=11, progress=False, delta=0.10
        )
        assert isinstance(report.all_passed, bool)

    def test_has_six_checks(self):
        report = run_protocol(
            _dummy_saw, resolution=11, progress=False, delta=0.10
        )
        assert len(report.checks) == 6

    def test_baseline_is_set(self):
        report = run_protocol(
            _dummy_saw, resolution=11, progress=False, delta=0.10
        )
        assert report.baseline is not None
        assert report.baseline.resolution == 11


# ── run_protocol ───────────────────────────────────────────────────────


class TestRunProtocol:
    def test_custom_third_alternatives(self):
        report = run_protocol(
            _dummy_saw,
            resolution=11,
            third_alternatives=[[0.3, 0.7]],
            progress=False,
            delta=0.10,
        )
        stability = report.checks[5]
        assert "[0.3, 0.7]" in stability.message

    def test_extra_weights(self):
        report = run_protocol(
            _dummy_saw,
            resolution=11,
            extra_weights=[[0.25, 0.75], [0.75, 0.25]],
            progress=False,
            delta=0.10,
        )
        assert len(report.weight_sensitivity) == 2
        for w, vista, checks in report.weight_sensitivity:
            assert len(checks) == 5
            assert vista.resolution == 11

    def test_metadata_has_elapsed(self):
        report = run_protocol(
            _dummy_saw, resolution=11, progress=False, delta=0.10
        )
        assert "elapsed_seconds" in report.metadata
        assert report.metadata["elapsed_seconds"] >= 0


# ── plot_protocol_report ───────────────────────────────────────────────


class TestPlotProtocolReport:
    def test_returns_figure(self):
        import matplotlib

        matplotlib.use("Agg")

        report = run_protocol(
            _dummy_saw, resolution=11, progress=False, delta=0.10
        )
        fig = plot_protocol_report(report)
        assert fig is not None
        plt_module = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_module.close(fig)

    def test_custom_figsize(self):
        import matplotlib

        matplotlib.use("Agg")

        report = run_protocol(
            _dummy_saw, resolution=11, progress=False, delta=0.10
        )
        fig = plot_protocol_report(report, figsize=(12, 8))
        assert fig is not None
        plt_module = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_module.close(fig)
