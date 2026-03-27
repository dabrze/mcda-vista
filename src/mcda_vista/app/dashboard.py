"""VISTA interactive dashboard.

Launch with::

    streamlit run -m mcda_vista.app.dashboard
    # or after installing with [app] extra:
    mcda-vista-app
"""

from __future__ import annotations

import streamlit as st

from mcda_vista.methods import get_method, list_methods

from mcda_vista.app._grid import render_grid
from mcda_vista.app._helpers import auto_point_size, render_method_params
from mcda_vista.app._pair import render_pair
from mcda_vista.app._protocol import render_protocol
from mcda_vista.app._single import render_single
from mcda_vista.app._style import inject_compact_css


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="VISTA Dashboard",
        layout="wide",
    )

    inject_compact_css()

    st.markdown("### VISTA — VISualization of relation Topologies of Alternatives")

    # ------------------------------------------------------------------
    # Sidebar — shared controls
    # ------------------------------------------------------------------
    available_methods = list_methods()

    with st.sidebar:
        st.header("Configuration")

        st.subheader("Method")
        method_name = st.selectbox(
            "MCDA method",
            options=available_methods,
            format_func=lambda m: f"{get_method(m).display_name}  ({m})",
        )

        adapter = get_method(method_name)

        st.subheader("Grid")
        resolution = st.slider(
            "Resolution (per axis)",
            min_value=11,
            max_value=201,
            value=51,
            step=2,
            help="Higher values give finer plots but take longer to compute.",
        )

        auto_ps = auto_point_size(resolution)
        point_size = st.slider(
            "Point size",
            min_value=0.1,
            max_value=20.0,
            value=auto_ps,
            step=0.1,
            help="Marker area for scatter points. Auto-derived from resolution by default.",
            key="point_size",
        )

        st.subheader("Reference point")
        ref_x = st.slider("x (criterion 1)", 0.0, 1.0, 0.5, 0.01, key="ref_x")
        ref_y = st.slider("y (criterion 2)", 0.0, 1.0, 0.5, 0.01, key="ref_y")

        st.subheader("Weights")
        w1 = st.number_input("w\u2081", min_value=0.01, value=1.0, step=0.1, key="w1")
        w2 = st.number_input("w\u2082", min_value=0.01, value=1.0, step=0.1, key="w2")

        st.subheader(f"{adapter.display_name} parameters")
        method_params = render_method_params(adapter, key_prefix="mp_")

        st.subheader("Third alternative")
        use_third = st.checkbox("Enable third alternative", key="use_third")
        third_alt: list[float] | None = None
        if use_third:
            t_x = st.slider("Third x", 0.0, 1.0, 0.3, 0.01, key="third_x")
            t_y = st.slider("Third y", 0.0, 1.0, 0.7, 0.01, key="third_y")
            third_alt = [t_x, t_y]

    # ------------------------------------------------------------------
    # Shared derived values
    # ------------------------------------------------------------------
    reference = [ref_x, ref_y]
    weights = [w1, w2]

    # ------------------------------------------------------------------
    # View tabs
    # ------------------------------------------------------------------
    tab_single, tab_pair, tab_grid, tab_protocol = st.tabs(
        ["Single", "Pair", "Grid", "Protocol"]
    )

    with tab_single:
        render_single(
            method_name=method_name,
            resolution=resolution,
            reference=reference,
            weights=weights,
            third_alt=third_alt,
            method_params=method_params,
            point_size=point_size,
        )

    with tab_pair:
        render_pair(
            resolution=resolution,
            reference=reference,
            weights=weights,
            third_alt=third_alt,
            point_size=point_size,
        )

    with tab_grid:
        render_grid(
            resolution=resolution,
            reference=reference,
            weights=weights,
            third_alt=third_alt,
            method_name=method_name,
            method_params=method_params,
            point_size=point_size,
        )

    with tab_protocol:
        render_protocol(
            method_name=method_name,
            resolution=resolution,
            method_params=method_params,
            point_size=point_size,
        )


if __name__ == "__main__":
    main()
