"""Smoke tests for plotting (verify no exceptions are raised)."""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mcda_vista.core import VistaResult
from mcda_vista.plotting import plot_vista, plot_vista_comparison


@pytest.fixture
def mock_vista_result():
    """A small VistaResult with random but valid data."""
    rng = np.random.default_rng(42)
    n_points = 25
    grid = rng.random((n_points, 2))
    relations = rng.choice([0, 1, 2, 3, 4], size=n_points).astype(np.uint8)

    return VistaResult(
        grid=grid,
        relations=relations,
        reference=np.array([0.5, 0.5]),
        weights=np.array([1.0, 1.0]),
        method_name="test_method",
        params={},
        resolution=5,
        n_criteria=2,
        metadata={"elapsed_seconds": 0.01},
    )


class TestPlotVista:
    def test_returns_figure(self, mock_vista_result):
        fig = plot_vista(mock_vista_result)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_no_exception_basic(self, mock_vista_result):
        fig = plot_vista(mock_vista_result)
        plt.close(fig)

    def test_custom_labels(self, mock_vista_result):
        fig = plot_vista(
            mock_vista_result,
            xlabel="Cost",
            ylabel="Quality",
            title="Test Plot",
        )
        plt.close(fig)

    def test_no_legend(self, mock_vista_result):
        fig = plot_vista(mock_vista_result, show_legend=False)
        plt.close(fig)

    def test_no_reference(self, mock_vista_result):
        fig = plot_vista(mock_vista_result, show_reference=False)
        plt.close(fig)

    def test_with_third_alternative(self):
        rng = np.random.default_rng(7)
        result = VistaResult(
            grid=rng.random((9, 2)),
            relations=rng.choice([1, 2, 3], size=9).astype(np.uint8),
            reference=np.array([0.5, 0.5]),
            weights=np.array([1.0, 1.0]),
            method_name="with_third",
            params={},
            resolution=3,
            n_criteria=2,
            third_alternative=np.array([0.3, 0.7]),
            metadata={},
        )
        fig = plot_vista(result)
        plt.close(fig)


class TestPlotVistaComparison:
    def test_returns_figure(self, mock_vista_result):
        fig = plot_vista_comparison([mock_vista_result, mock_vista_result])
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_single_result(self, mock_vista_result):
        fig = plot_vista_comparison([mock_vista_result])
        plt.close(fig)

    def test_multiple_results(self, mock_vista_result):
        results = [mock_vista_result] * 4
        fig = plot_vista_comparison(results, ncols=2)
        plt.close(fig)

    def test_with_title(self, mock_vista_result):
        fig = plot_vista_comparison(
            [mock_vista_result, mock_vista_result],
            title="Comparison",
        )
        plt.close(fig)
