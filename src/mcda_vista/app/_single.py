"""Single-vista view — compact version of the original dashboard."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time

from mcda_vista.core import VistaResult
from mcda_vista.methods import get_method
from mcda_vista.plotting import plot_vista
from mcda_vista.relation import Relation

from mcda_vista.app._helpers import (
    auto_point_size,
    cached_generate_vista,
    fig_to_png_bytes,
    hashable_params,
    result_to_csv,
)


def render_single(
    method_name: str,
    resolution: int,
    reference: list[float],
    weights: list[float],
    third_alt: list[float] | None,
    method_params: dict[str, Any],
    point_size: float | None = None,
) -> None:
    """Render a single vista with info panel and download buttons."""
    adapter = get_method(method_name)

    t0 = time.perf_counter()
    result = cached_generate_vista(
        method=method_name,
        resolution=resolution,
        reference=tuple(reference),
        weights=tuple(weights),
        n_criteria=2,
        third_alternative=tuple(third_alt) if third_alt else None,
        _params_key=hashable_params(**method_params),
        **method_params,
    )
    elapsed = time.perf_counter() - t0

    if point_size is None:
        point_size = auto_point_size(resolution)
    fig = plot_vista(
        result,
        title=f"{adapter.display_name} VISTA",
        point_size=point_size,
    )
    plt.tight_layout()
    png_bytes = fig_to_png_bytes(fig)

    col_plot, col_info = st.columns([2, 1])

    with col_plot:
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with col_info:
        st.markdown(f"**Method:** {adapter.display_name}")
        st.markdown(f"**Resolution:** {resolution}\u00d7{resolution} = {resolution**2:,} pts")
        st.markdown(f"**Reference:** ({reference[0]:.2f}, {reference[1]:.2f})")
        st.markdown(f"**Weights:** ({weights[0]:.2f}, {weights[1]:.2f})")
        if third_alt:
            st.markdown(f"**Third alt:** ({third_alt[0]:.2f}, {third_alt[1]:.2f})")
        if method_params:
            st.markdown("**Parameters:**")
            for k, v in method_params.items():
                st.markdown(f"- `{k}` = {v}")
        st.markdown(f"**Compute time:** {elapsed:.3f} s")

        _render_distribution(result)

    _render_downloads(result, png_bytes, method_name)


def _render_distribution(result: VistaResult) -> None:
    """Show relation distribution as coloured counts."""
    with st.expander("Distribution", expanded=True):
        for rel in Relation:
            count = int(np.sum(result.relations == rel.value))
            if count > 0:
                pct = count / len(result.relations) * 100
                st.markdown(
                    f"<span style='color:{rel.color}'>\u25a0</span> "
                    f"**{rel.label}**: {count:,} ({pct:.1f}%)",
                    unsafe_allow_html=True,
                )


def _render_downloads(
    result: VistaResult, png_bytes: bytes, method_name: str
) -> None:
    """Download buttons for PNG and CSV."""
    col1, col2, _ = st.columns([1, 1, 3])
    with col1:
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name=f"vista_{method_name}.png",
            mime="image/png",
        )
    with col2:
        st.download_button(
            "Download CSV",
            data=result_to_csv(result),
            file_name=f"vista_{method_name}.csv",
            mime="text/csv",
        )
