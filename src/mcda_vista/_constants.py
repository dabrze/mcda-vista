"""Shared numeric and layout constants for VISTA visualisation."""

from __future__ import annotations

# ── Scatter point sizing ─────────────────────────────────────────────────

POINT_SIZE_BASE: float = 20.0
"""Numerator in the auto-point-size formula: ``base / (resolution / divisor)``."""

POINT_SIZE_DIVISOR: float = 10.0
"""Denominator scale in the auto-point-size formula."""

POINT_SIZE_MIN: float = 0.1
"""Floor for the computed marker area so points never become invisible."""

# ── Figure scaling ───────────────────────────────────────────────────────

GRID_FIGURE_SCALE: float = 2.5
"""Inches per cell when auto-sizing ``plot_vista_grid`` figures."""

COMPARISON_FIGURE_SCALE: float = 2.8
"""Inches per cell when auto-sizing ``plot_vista_comparison`` figures."""

LEGEND_BOTTOM_MARGIN: float = 0.08
"""Bottom margin reserved for a shared figure legend via ``subplots_adjust``."""
