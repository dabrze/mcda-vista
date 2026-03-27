"""Grid comparison view — methods × parameter sweep matrix."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

from mcda_vista.core import VistaResult
from mcda_vista.methods import get_method, list_methods
from mcda_vista.plotting import plot_vista_grid

from mcda_vista.app._helpers import (
    auto_point_size,
    cached_generate_vista,
    fig_to_png_bytes,
    hashable_params,
)


_AXIS_CHOICES = ["Methods", "Weight ratio (w₁:w₂)", "Reference x", "Reference y"]


def render_grid(
    resolution: int,
    reference: list[float],
    weights: list[float],
    third_alt: list[float] | None,
    method_name: str,
    method_params: dict[str, Any],
    point_size: float | None = None,
) -> None:
    """Render a grid of vistas varying two axes."""

    available = list_methods()

    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        row_var = st.selectbox("Row variable", options=_AXIS_CHOICES, index=0, key="grid_row_var")
        row_values, row_labels = _axis_config(row_var, available, method_name, method_params, prefix="row")

    with cfg_col2:
        remaining = [c for c in _AXIS_CHOICES if c != row_var]
        col_var = st.selectbox("Column variable", options=remaining, index=0, key="grid_col_var")
        col_values, col_labels = _axis_config(col_var, available, method_name, method_params, prefix="col")

    if not row_values or not col_values:
        st.info("Configure at least one value for each axis.")
        return

    n_total = len(row_values) * len(col_values)
    if n_total > 36:
        st.warning(f"Grid has {n_total} cells — this may take a while.")

    if st.button("Generate grid", key="grid_generate"):
        progress = st.progress(0, text="Computing vistas…")
        results: list[list[VistaResult | None]] = []
        done = 0

        for r_idx, r_val in enumerate(row_values):
            row: list[VistaResult | None] = []
            for c_idx, c_val in enumerate(col_values):
                ref, w, mname, mparams = _resolve_cell(
                    row_var, r_val, col_var, c_val,
                    reference, weights, method_name, method_params,
                )
                try:
                    result = cached_generate_vista(
                        method=mname,
                        resolution=resolution,
                        reference=tuple(ref),
                        weights=tuple(w),
                        n_criteria=2,
                        third_alternative=tuple(third_alt) if third_alt else None,
                        _params_key=hashable_params(**mparams),
                        **mparams,
                    )
                    row.append(result)
                except Exception:
                    row.append(None)
                done += 1
                progress.progress(done / n_total, text=f"Computed {done}/{n_total}")
            results.append(row)
        progress.empty()

        if point_size is None:
            point_size = auto_point_size(resolution)
        fig = plot_vista_grid(
            results,
            row_labels=row_labels,
            col_labels=col_labels,
            point_size=point_size,
        )

        png_bytes = fig_to_png_bytes(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.download_button(
            "Download grid PNG",
            data=png_bytes,
            file_name="vista_grid.png",
            mime="image/png",
            key="grid_dl",
        )


def _axis_config(
    var_name: str,
    available: list[str],
    method_name: str,
    method_params: dict[str, Any],
    prefix: str,
) -> tuple[list[Any], list[str]]:
    """Return (values, labels) for a given axis variable."""

    if var_name == "Methods":
        selected = st.multiselect(
            f"{prefix.title()} methods",
            options=available,
            default=[available[0]] if available else [],
            key=f"grid_{prefix}_methods",
        )
        labels = [get_method(m).display_name for m in selected]
        return selected, labels

    if var_name == "Weight ratio (w₁:w₂)":
        ratios_str = st.text_input(
            f"{prefix.title()} weight ratios (comma-separated, e.g. 0.5,1.0,2.0)",
            value="0.5, 1.0, 2.0",
            key=f"grid_{prefix}_weights",
        )
        try:
            ratios = [float(x.strip()) for x in ratios_str.split(",") if x.strip()]
        except ValueError:
            st.error("Invalid weight ratios.")
            return [], []
        values = [[r, 1.0] for r in ratios]
        labels = [f"w₁:w₂ = {r:.2f}:1.00" for r in ratios]
        return values, labels

    if var_name == "Reference x":
        vals_str = st.text_input(
            f"{prefix.title()} reference x values (comma-separated)",
            value="0.25, 0.50, 0.75",
            key=f"grid_{prefix}_refx",
        )
        try:
            vals = [float(x.strip()) for x in vals_str.split(",") if x.strip()]
        except ValueError:
            st.error("Invalid values.")
            return [], []
        labels = [f"x={v:.2f}" for v in vals]
        return vals, labels

    if var_name == "Reference y":
        vals_str = st.text_input(
            f"{prefix.title()} reference y values (comma-separated)",
            value="0.25, 0.50, 0.75",
            key=f"grid_{prefix}_refy",
        )
        try:
            vals = [float(x.strip()) for x in vals_str.split(",") if x.strip()]
        except ValueError:
            st.error("Invalid values.")
            return [], []
        labels = [f"y={v:.2f}" for v in vals]
        return vals, labels

    return [], []


def _resolve_cell(
    row_var: str,
    row_val: Any,
    col_var: str,
    col_val: Any,
    base_ref: list[float],
    base_weights: list[float],
    base_method: str,
    base_params: dict[str, Any],
) -> tuple[list[float], list[float], str, dict[str, Any]]:
    """Resolve per-cell reference, weights, method, and params."""
    ref = list(base_ref)
    w = list(base_weights)
    mname = base_method
    mparams = dict(base_params)

    for var, val in [(row_var, row_val), (col_var, col_val)]:
        if var == "Methods":
            mname = val
            adapter = get_method(mname)
            mparams = adapter.default_params()
        elif var == "Weight ratio (w₁:w₂)":
            w = list(val)
        elif var == "Reference x":
            ref[0] = val
        elif var == "Reference y":
            ref[1] = val

    return ref, w, mname, mparams
