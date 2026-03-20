"""Matplotlib-based visualisation for VISTA results.

Ports the R ``ggplot2`` visualisations from *viz.Rmd* to Python,
matching the original colour scheme and layout conventions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, cast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from mcda_vista.relation import Relation

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from mcda_vista.core import VistaResult

__all__ = [
    "plot_vista",
    "plot_vista_grid",
    "plot_vista_comparison",
    "vista_colormap",
]

# ── Canonical ordering used for legends ──────────────────────────────────

_LEGEND_ORDER: list[Relation] = [
    Relation.BETTER,
    Relation.INDIFFERENT,
    Relation.WORSE,
    Relation.INCOMPARABLE,
]


def _auto_point_size(resolution: int) -> float:
    """Compute a scatter marker area that fills the plot nicely."""
    return max(0.1, 20.0 / (resolution / 10))


# ── Public helpers ───────────────────────────────────────────────────────


def vista_colormap() -> dict[int, str]:
    """Return ``{Relation.value: hex_color}`` mapping for external use."""
    return Relation.color_map()


# ── Internal drawing helper ──────────────────────────────────────────────


def _draw_vista_on_ax(
    ax: Axes,
    result: VistaResult,
    point_size: float | None = None,
    show_reference: bool = True,
) -> None:
    """Render a single VISTA scatter plot onto *ax*.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    result : VistaResult
        Pre-computed VISTA sweep result (must be 2-criteria).
    point_size : float or None
        Marker area passed to ``ax.scatter``.  When *None* the size is
        automatically derived from ``result.resolution``.
    show_reference : bool
        If *True*, draw dashed crosshair lines and an open-circle marker
        at the reference point.
    """
    if point_size is None:
        point_size = _auto_point_size(result.resolution)
    grid = result.grid
    relations = result.relations
    ref = result.reference

    # Plot each relation category in legend order so the z-order is
    # consistent and the legend entries are predictable.
    for rel in _LEGEND_ORDER:
        mask = relations == rel.value
        if not np.any(mask):
            continue
        ax.scatter(
            grid[mask, 0],
            grid[mask, 1],
            c=rel.color,
            s=point_size,
            marker="s",
            label=rel.label,
            linewidths=0,
            rasterized=True,
        )

    # Handle ERROR points (if any) separately.
    err_mask = relations == Relation.ERROR.value
    if np.any(err_mask):
        ax.scatter(
            grid[err_mask, 0],
            grid[err_mask, 1],
            c=Relation.ERROR.color,
            s=point_size,
            marker="s",
            label=Relation.ERROR.label,
            linewidths=0,
            rasterized=True,
        )

    # Reference point decoration.
    if show_reference:
        ax.axvline(ref[0], linestyle="--", color="#555555", linewidth=0.5)
        ax.axhline(ref[1], linestyle="--", color="#555555", linewidth=0.5)
        ax.plot(
            ref[0],
            ref[1],
            marker="o",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor="#555555",
            markeredgewidth=0.8,
            zorder=5,
        )

    # Optional third alternative (triangle marker).
    if result.third_alternative is not None:
        alt = result.third_alternative
        ax.plot(
            alt[0],
            alt[1],
            marker="^",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor="#555555",
            markeredgewidth=0.8,
            zorder=5,
        )

    # Optional extra alternatives (diamond markers).
    if result.extra_alternatives is not None:
        for row in result.extra_alternatives:
            ax.plot(
                row[0],
                row[1],
                marker="D",
                markersize=4,
                markerfacecolor="none",
                markeredgecolor="#555555",
                markeredgewidth=0.8,
                zorder=5,
            )

    ax.set_aspect("equal")


def _shared_legend_handles() -> list[mpatches.Patch]:
    """Build legend handles for all four main relation types.

    Always returns all four entries so the legend is stable even when some
    categories are absent from a particular plot.
    """
    return [
        mpatches.Patch(color=rel.color, label=rel.label)
        for rel in _LEGEND_ORDER
    ]


# ── Public plot functions ────────────────────────────────────────────────


def plot_vista(
    result: VistaResult,
    ax: Axes | None = None,
    xlabel: str = "Criterion 1",
    ylabel: str = "Criterion 2",
    title: str | None = None,
    show_reference: bool = True,
    show_legend: bool = True,
    point_size: float | None = None,
) -> Figure:
    """Plot a single VISTA result.

    Parameters
    ----------
    result : VistaResult
        A 2-criteria VISTA sweep result.
    ax : Axes or None
        If *None* a new figure and axes are created.
    xlabel, ylabel : str
        Axis labels.
    title : str or None
        Optional title above the plot.
    show_reference : bool
        Draw crosshair + reference-point marker.
    show_legend : bool
        Attach a legend listing all relation types.
    point_size : float or None
        Marker area for scatter points.  Auto-computed from resolution
        when *None*.

    Returns
    -------
    Figure
        The matplotlib :class:`~matplotlib.figure.Figure` containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = cast(Figure, ax.get_figure())

    _draw_vista_on_ax(ax, result, point_size=point_size, show_reference=show_reference)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if show_legend:
        ax.legend(
            handles=_shared_legend_handles(),
            loc="best",
            framealpha=0.9,
            fontsize="small",
            markerscale=1.5,
        )

    fig.tight_layout()
    return fig


def plot_vista_grid(
    results: list[list[VistaResult | None]] | dict[tuple[int, int], VistaResult],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    show_legend: bool = True,
    point_size: float | None = None,
) -> Figure:
    """Create a grid of VISTA subplots (rows × columns).

    Mimics ``facet_grid`` in R/ggplot2: row labels appear on the left,
    column labels on top, and a single shared legend sits at the bottom.

    Parameters
    ----------
    results : 2-D list or dict
        Either a nested list ``results[row][col]`` or a dict mapping
        ``(row_idx, col_idx)`` → :class:`VistaResult`.  ``None`` entries
        produce blank panels.
    row_labels, col_labels : sequence of str
        Human-readable labels for each row / column.
    title : str or None
        Super-title for the whole figure.
    figsize : tuple or None
        Figure size in inches; auto-computed if *None*.
    show_legend : bool
        Show shared legend at the bottom.
    point_size : float or None
        Marker area for scatter points.  Auto-computed from resolution
        when *None*.

    Returns
    -------
    Figure
    """
    nrows = len(row_labels)
    ncols = len(col_labels)
    if figsize is None:
        figsize = (2.5 * ncols, 2.5 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
    )

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]

            # Retrieve the result for this cell.
            if isinstance(results, dict):
                res = results.get((r, c))
            else:
                res = results[r][c]

            if res is not None:
                _draw_vista_on_ax(ax, res, point_size=point_size)

            # Column labels on top row.
            if r == 0:
                ax.set_title(col_labels[c], fontsize="small", fontweight="bold")

            # Row labels on the leftmost column.
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize="small", fontweight="bold")
            else:
                ax.set_ylabel("")

            # Suppress tick labels for a clean grid.
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(length=0)

    if title is not None:
        fig.suptitle(title, fontweight="bold", y=1.02)

    if show_legend:
        fig.legend(
            handles=_shared_legend_handles(),
            loc="lower center",
            ncol=len(_LEGEND_ORDER),
            frameon=False,
            fontsize="small",
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.tight_layout()
    if show_legend:
        fig.subplots_adjust(bottom=0.08)

    return fig


def plot_vista_comparison(
    results: Sequence[VistaResult],
    ncols: int = 4,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    show_legend: bool = True,
    point_size: float | None = None,
) -> Figure:
    """Side-by-side comparison of multiple VISTA results (facet-wrap).

    Each subplot is titled with the corresponding
    :pyattr:`VistaResult.method_name`.

    Parameters
    ----------
    results : sequence of VistaResult
        Results to compare.
    ncols : int
        Maximum number of columns per row.
    figsize : tuple or None
        Figure size; auto-computed if *None*.
    title : str or None
        Super-title for the figure.
    show_legend : bool
        Shared legend at the bottom.
    point_size : float or None
        Marker area for scatter points.  Auto-computed from resolution
        when *None*.

    Returns
    -------
    Figure
    """
    n = len(results)
    nrows = max(1, int(np.ceil(n / ncols)))
    actual_ncols = min(n, ncols)

    if figsize is None:
        figsize = (2.8 * actual_ncols, 2.8 * nrows)

    fig, axes = plt.subplots(nrows, actual_ncols, figsize=figsize, squeeze=False)

    for idx, res in enumerate(results):
        r, c = divmod(idx, actual_ncols)
        ax = axes[r, c]
        _draw_vista_on_ax(ax, res, point_size=point_size)
        ax.set_title(res.method_name, fontsize="small", fontweight="bold")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)

    # Hide unused axes.
    for idx in range(n, nrows * actual_ncols):
        r, c = divmod(idx, actual_ncols)
        axes[r, c].set_visible(False)

    if title is not None:
        fig.suptitle(title, fontweight="bold", y=1.02)

    if show_legend:
        fig.legend(
            handles=_shared_legend_handles(),
            loc="lower center",
            ncol=len(_LEGEND_ORDER),
            frameon=False,
            fontsize="small",
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.tight_layout()
    if show_legend:
        fig.subplots_adjust(bottom=0.08)

    return fig
