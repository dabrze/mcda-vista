"""Pairwise comparison view — two vistas side-by-side."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time

from mcda_vista.methods import get_method, list_methods
from mcda_vista.plotting import plot_vista
from mcda_vista.relation import Relation

from mcda_vista.app._helpers import (
    auto_point_size,
    cached_generate_vista,
    fig_to_png_bytes,
    hashable_params,
    render_method_params,
)


def render_pair(
    resolution: int,
    reference: list[float],
    weights: list[float],
    third_alt: list[float] | None,
    point_size: float | None = None,
) -> None:
    """Render two vistas side-by-side for comparison."""
    available = list_methods()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Vista A")
        method_a = st.selectbox(
            "Method",
            options=available,
            format_func=lambda m: f"{get_method(m).display_name} ({m})",
            key="pair_method_a",
        )
        adapter_a = get_method(method_a)
        params_a = render_method_params(adapter_a, key_prefix="pair_a_")

    with col_b:
        st.markdown("#### Vista B")
        default_b = min(1, len(available) - 1)
        method_b = st.selectbox(
            "Method",
            options=available,
            index=default_b,
            format_func=lambda m: f"{get_method(m).display_name} ({m})",
            key="pair_method_b",
        )
        adapter_b = get_method(method_b)
        params_b = render_method_params(adapter_b, key_prefix="pair_b_")

    if point_size is None:
        point_size = auto_point_size(resolution)
    ref_t = tuple(reference)
    w_t = tuple(weights)
    third_t = tuple(third_alt) if third_alt else None

    t0 = time.perf_counter()
    result_a = cached_generate_vista(
        method=method_a,
        resolution=resolution,
        reference=ref_t,
        weights=w_t,
        n_criteria=2,
        third_alternative=third_t,
        _params_key=hashable_params(**params_a),
        **params_a,
    )
    result_b = cached_generate_vista(
        method=method_b,
        resolution=resolution,
        reference=ref_t,
        weights=w_t,
        n_criteria=2,
        third_alternative=third_t,
        _params_key=hashable_params(**params_b),
        **params_b,
    )
    elapsed = time.perf_counter() - t0

    fig_a = plot_vista(result_a, title=adapter_a.display_name, point_size=point_size)
    fig_b = plot_vista(result_b, title=adapter_b.display_name, point_size=point_size)
    plt.tight_layout()

    png_a = fig_to_png_bytes(fig_a)
    png_b = fig_to_png_bytes(fig_b)

    plot_a, plot_b = st.columns(2)
    with plot_a:
        st.pyplot(fig_a, use_container_width=True)
        plt.close(fig_a)
        _compact_distribution(result_a)
    with plot_b:
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)
        _compact_distribution(result_b)

    st.caption(f"Total compute time: {elapsed:.3f} s")

    dl1, dl2, _ = st.columns([1, 1, 3])
    with dl1:
        st.download_button(
            f"Download {adapter_a.display_name} PNG",
            data=png_a,
            file_name=f"vista_{method_a}.png",
            mime="image/png",
            key="pair_dl_a",
        )
    with dl2:
        st.download_button(
            f"Download {adapter_b.display_name} PNG",
            data=png_b,
            file_name=f"vista_{method_b}.png",
            mime="image/png",
            key="pair_dl_b",
        )


def _compact_distribution(result: Any) -> None:
    """One-line distribution summary."""
    parts: list[str] = []
    for rel in Relation:
        count = int(np.sum(result.relations == rel.value))
        if count > 0:
            pct = count / len(result.relations) * 100
            parts.append(
                f"<span style='color:{rel.color}'>\u25a0</span> "
                f"{rel.label}: {pct:.1f}%"
            )
    st.markdown(" &nbsp; ".join(parts), unsafe_allow_html=True)
