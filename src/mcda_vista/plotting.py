"""Matplotlib-based visualisation for VISTA results.

Ports the R ``ggplot2`` visualisations from *viz.Rmd* to Python,
matching the original colour scheme and layout conventions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, cast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from mcda_vista._constants import (
    COMPARISON_FIGURE_SCALE,
    GRID_FIGURE_SCALE,
    LEGEND_BOTTOM_MARGIN,
    POINT_SIZE_BASE,
    POINT_SIZE_DIVISOR,
    POINT_SIZE_MIN,
)
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
    return max(POINT_SIZE_MIN, POINT_SIZE_BASE / (resolution / POINT_SIZE_DIVISOR))


# ── Public helpers ───────────────────────────────────────────────────────


def vista_colormap() -> dict[int, str]:
    """Return ``{Relation.value: hex_color}`` mapping for external use."""
    return Relation.color_map()


# ── Internal drawing helper ──────────────────────────────────────────────

# Marker styles available for ``extra_alternatives`` (background sets).
_EXTRA_MARKER_STYLES: dict[str, dict[str, Any]] = {
    "diamond": {
        "marker": "D",
        "markersize": 4,
        "markeredgecolor": "#555555",
        "markeredgewidth": 0.8,
    },
    "triangle": {
        "marker": "^",
        "markersize": 5,
        "markeredgecolor": "#FFFFFF",
        "markeredgewidth": 1.2,
    },
}


def _draw_vista_on_ax(
    ax: Axes,
    result: VistaResult,
    point_size: float | None = None,
    show_reference: bool = True,
    extra_marker: str = "diamond",
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
    extra_marker : {"diamond", "triangle"}
        Style for ``result.extra_alternatives``.  ``"diamond"`` (default)
        draws small grey diamonds; ``"triangle"`` draws white triangles
        matching the third-alternative marker, which reads better when a
        whole background set sits on top of the coloured regions.
    """
    if extra_marker not in _EXTRA_MARKER_STYLES:
        raise ValueError(
            f"extra_marker must be one of {sorted(_EXTRA_MARKER_STYLES)}, "
            f"got {extra_marker!r}"
        )
    if point_size is None:
        point_size = _auto_point_size(result.resolution)
    grid = result.grid
    relations = result.relations
    ref = result.reference
    free_indices = tuple(result.metadata.get("free_indices", (0, 1)))
    x_index, y_index = free_indices

    # Plot each relation category in legend order so the z-order is
    # consistent and the legend entries are predictable.
    for rel in _LEGEND_ORDER:
        mask = relations == rel.value
        if not np.any(mask):
            continue
        ax.scatter(
            grid[mask, x_index],
            grid[mask, y_index],
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
            grid[err_mask, x_index],
            grid[err_mask, y_index],
            c=Relation.ERROR.color,
            s=point_size,
            marker="s",
            label=Relation.ERROR.label,
            linewidths=0,
            rasterized=True,
        )

    # Reference point decoration.
    if show_reference:
        ax.axvline(ref[x_index], linestyle="--", color="#555555", linewidth=0.5)
        ax.axhline(ref[y_index], linestyle="--", color="#555555", linewidth=0.5)
        ax.plot(
            ref[x_index],
            ref[y_index],
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
            alt[x_index],
            alt[y_index],
            marker="^",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor="#FFFFFF",
            markeredgewidth=1.2,
            zorder=5,
        )

    # Optional extra alternatives.
    if result.extra_alternatives is not None:
        style = _EXTRA_MARKER_STYLES[extra_marker]
        for row in result.extra_alternatives:
            ax.plot(
                row[x_index],
                row[y_index],
                markerfacecolor="none",
                zorder=5,
                **style,
            )

    ax.set_aspect("equal")


def _shared_legend_handles() -> list[mpatches.Patch]:
    """Build legend handles for all four main relation types.

    Always returns all four entries so the legend is stable even when some
    categories are absent from a particular plot.
    """
    return [mpatches.Patch(color=rel.color, label=rel.label) for rel in _LEGEND_ORDER]


def _finalize_figure(fig: Figure, *, title: str | None, show_legend: bool) -> None:
    """Apply shared legend and layout adjustments to a multi-panel figure."""
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
        fig.subplots_adjust(bottom=LEGEND_BOTTOM_MARGIN)


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
    extra_marker: str = "diamond",
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
    extra_marker : {"diamond", "triangle"}
        Marker style for background alternatives.

    Returns
    -------
    Figure
        The matplotlib :class:`~matplotlib.figure.Figure` containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = cast(Figure, ax.get_figure())

    _draw_vista_on_ax(
        ax,
        result,
        point_size=point_size,
        show_reference=show_reference,
        extra_marker=extra_marker,
    )

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
    extra_marker: str = "diamond",
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
    extra_marker : {"diamond", "triangle"}
        Marker style for background alternatives in every panel.

    Returns
    -------
    Figure
    """
    nrows = len(row_labels)
    ncols = len(col_labels)
    if figsize is None:
        figsize = (GRID_FIGURE_SCALE * ncols, GRID_FIGURE_SCALE * nrows)

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
                _draw_vista_on_ax(
                    ax, res, point_size=point_size, extra_marker=extra_marker
                )

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

    _finalize_figure(fig, title=title, show_legend=show_legend)

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
        figsize = (
            COMPARISON_FIGURE_SCALE * actual_ncols,
            COMPARISON_FIGURE_SCALE * nrows,
        )

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

    _finalize_figure(fig, title=title, show_legend=show_legend)

    return fig
