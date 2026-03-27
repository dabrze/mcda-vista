"""Protocol report view — run and display the VISTA protocol."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

from mcda_vista.methods import get_method
from mcda_vista.protocol import ProtocolReport, plot_protocol_report, run_protocol

from mcda_vista.app._helpers import fig_to_png_bytes


def render_protocol(
    method_name: str,
    resolution: int,
    method_params: dict[str, Any],
    point_size: float | None = None,
) -> None:
    """Run the VISTA protocol and display results."""
    adapter = get_method(method_name)

    with st.expander("Protocol settings", expanded=False):
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            stability_threshold = st.slider(
                "Stability threshold",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                key="proto_stability",
            )
            n_rays = st.slider(
                "Number of rays (check 4)",
                min_value=8,
                max_value=72,
                value=36,
                step=4,
                key="proto_nrays",
            )
        with p_col2:
            ratio_lo = st.number_input(
                "Ratio lower bound",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.05,
                key="proto_ratio_lo",
            )
            ratio_hi = st.number_input(
                "Ratio upper bound",
                min_value=1.0,
                max_value=5.0,
                value=1.2,
                step=0.05,
                key="proto_ratio_hi",
            )

        use_weight_sens = st.checkbox("Include weight sensitivity", key="proto_wsens")
        extra_weights: list[list[float]] | None = None
        if use_weight_sens:
            w_str = st.text_input(
                "Extra weight pairs (semicolon-separated, e.g. 0.3,0.7; 0.7,0.3)",
                value="0.3,0.7; 0.7,0.3",
                key="proto_extra_w",
            )
            try:
                extra_weights = [
                    [float(v.strip()) for v in pair.split(",")]
                    for pair in w_str.split(";")
                    if pair.strip()
                ]
            except ValueError:
                st.error("Invalid weight format.")
                extra_weights = None

    _SESSION_KEY = "protocol_report"

    if st.button(
        f"Run protocol for {adapter.display_name}",
        key="proto_run",
    ):
        with st.spinner("Running protocol (this may take a moment)…"):
            new_report = run_protocol(
                method=method_name,
                resolution=resolution,
                stability_threshold=stability_threshold,
                ratio_bounds=(ratio_lo, ratio_hi),
                n_rays=n_rays,
                extra_weights=extra_weights,
                progress=False,
                **method_params,
            )
        st.session_state[_SESSION_KEY] = new_report

    report: ProtocolReport | None = st.session_state.get(_SESSION_KEY)
    if report is None:
        st.info("Click the button above to run the protocol.")
        return

    elapsed = report.metadata.get("elapsed_seconds", 0)
    st.caption(f"Protocol completed in {elapsed:.1f} s")

    # Check results summary
    for check in report.checks:
        if check.passed is True:
            st.success(f"**{check.name}:** {check.message}")
        elif check.passed is False:
            st.error(f"**{check.name}:** {check.message}")
        else:
            st.info(f"**{check.name}:** {check.message}")

    # Full report figure
    fig = plot_protocol_report(report, point_size=point_size)
    png_bytes = fig_to_png_bytes(fig)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Text summary in expander
    with st.expander("Full text summary"):
        st.code(report.summary(), language=None)

    st.download_button(
        "Download protocol report PNG",
        data=png_bytes,
        file_name=f"protocol_{method_name}.png",
        mime="image/png",
        key="proto_dl",
    )
