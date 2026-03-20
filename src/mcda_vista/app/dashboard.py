"""VISTA interactive dashboard.

Launch with::

    streamlit run -m mcda_vista.app.dashboard
    # or after installing with [app] extra:
    mcda-vista-app
"""

from __future__ import annotations

import io
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from mcda_vista.core import generate_vista
from mcda_vista.methods import get_method, list_methods
from mcda_vista.plotting import plot_vista, plot_vista_comparison
from mcda_vista.relation import Relation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hashable_params(**kwargs: object) -> tuple[tuple[str, object], ...]:
    """Convert keyword arguments to a hashable, sorted tuple of pairs."""
    return tuple(sorted(kwargs.items()))


def _render_method_params(
    adapter: object,
    key_prefix: str = "",
) -> dict[str, object]:
    """Auto-generate sidebar widgets from a method's ``param_space()``."""
    space = adapter.param_space()
    defaults = adapter.default_params()
    params: dict[str, object] = {}

    if not space:
        st.caption("_No tuneable parameters for this method._")
        return params

    for pname, spec in space.items():
        label = spec.get("label", pname)
        key = f"{key_prefix}{pname}"

        if "choices" in spec:
            params[pname] = st.selectbox(
                label,
                options=spec["choices"],
                index=spec["choices"].index(spec.get("default", spec["choices"][0])),
                key=key,
            )
        elif "min" in spec and "max" in spec:
            params[pname] = st.slider(
                label,
                min_value=float(spec["min"]),
                max_value=float(spec["max"]),
                value=float(spec.get("default", defaults.get(pname, spec["min"]))),
                step=float(spec.get("step", 0.01)),
                key=key,
            )
        else:
            params[pname] = st.number_input(
                label,
                value=float(spec.get("default", defaults.get(pname, 0.0))),
                key=key,
            )

    return params


def _fig_to_png_bytes(fig: plt.Figure) -> bytes:
    """Render a Matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _result_to_csv(result: object) -> str:
    """Convert a VistaResult to CSV text."""
    header_parts = [
        f"c{i + 1}" for i in range(result.grid.shape[1])
    ]
    header_parts.append("relation")
    lines = [",".join(header_parts)]
    for row, rel in zip(result.grid, result.relations):
        vals = ",".join(f"{v:.6f}" for v in row)
        lines.append(f"{vals},{int(rel)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="VISTA Dashboard",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 VISTA — VISualization of relation Topologies of Alternatives")

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    available_methods = list_methods()

    with st.sidebar:
        st.header("⚙️ Configuration")

        # --- Method selector ---
        st.subheader("Method")
        method_name = st.selectbox(
            "MCDA method",
            options=available_methods,
            format_func=lambda m: f"{get_method(m).display_name}  ({m})",
        )

        adapter = get_method(method_name)

        # --- Resolution ---
        st.subheader("Grid")
        resolution = st.slider(
            "Resolution (per axis)",
            min_value=11,
            max_value=201,
            value=51,
            step=2,
            help="Higher values give finer plots but take longer to compute.",
        )

        # --- Reference point ---
        st.subheader("📍 Reference point")
        ref_x = st.slider("x (criterion 1)", 0.0, 1.0, 0.5, 0.01, key="ref_x")
        ref_y = st.slider("y (criterion 2)", 0.0, 1.0, 0.5, 0.01, key="ref_y")

        # --- Weights ---
        st.subheader("⚖️ Weights")
        w1 = st.number_input("w₁", min_value=0.01, value=1.0, step=0.1, key="w1")
        w2 = st.number_input("w₂", min_value=0.01, value=1.0, step=0.1, key="w2")

        # --- Method-specific parameters ---
        st.subheader(f"🔧 {adapter.display_name} parameters")
        method_params = _render_method_params(adapter, key_prefix="mp_")

        # --- Third alternative ---
        st.subheader("🔺 Third alternative")
        use_third = st.checkbox("Enable third alternative", key="use_third")
        third_alt: list[float] | None = None
        if use_third:
            t_x = st.slider("Third x", 0.0, 1.0, 0.3, 0.01, key="third_x")
            t_y = st.slider("Third y", 0.0, 1.0, 0.7, 0.01, key="third_y")
            third_alt = [t_x, t_y]

        # --- Comparison mode ---
        st.divider()
        st.subheader("📊 Comparison mode")
        compare = st.checkbox("Compare multiple methods", key="compare")
        compare_methods: list[str] = []
        if compare:
            compare_methods = st.multiselect(
                "Select methods to compare",
                options=available_methods,
                default=[method_name],
                key="compare_methods",
            )

    # ------------------------------------------------------------------
    # Generate VISTA
    # ------------------------------------------------------------------
    reference = [ref_x, ref_y]
    weights = [w1, w2]

    if not compare:
        # ---------- Single-method view ----------
        t0 = time.perf_counter()
        result = generate_vista(
            method=method_name,
            resolution=resolution,
            reference=reference,
            weights=weights,
            n_criteria=2,
            third_alternative=third_alt,
            progress=False,
            **method_params,
        )
        elapsed = time.perf_counter() - t0

        fig = plot_vista(
            result,
            title=f"{adapter.display_name} VISTA",
            point_size=max(0.1, 20.0 / (resolution / 10)),
        )
        plt.tight_layout()

        # Pre-render PNG for download before closing the figure
        png_bytes = _fig_to_png_bytes(fig)

        # ---- Display ----
        col_plot, col_info = st.columns([3, 1])

        with col_plot:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_info:
            st.subheader("ℹ️ Info")
            st.markdown(f"**Method:** {adapter.display_name}")
            st.markdown(f"**Resolution:** {resolution}×{resolution} = {resolution**2:,} pts")
            st.markdown(f"**Reference:** ({ref_x:.2f}, {ref_y:.2f})")
            st.markdown(f"**Weights:** ({w1:.2f}, {w2:.2f})")
            if third_alt:
                st.markdown(f"**Third alt:** ({third_alt[0]:.2f}, {third_alt[1]:.2f})")
            if method_params:
                st.markdown("**Parameters:**")
                for k, v in method_params.items():
                    st.markdown(f"- `{k}` = {v}")
            st.markdown(f"**Compute time:** {elapsed:.3f} s")

            # Relation distribution
            st.subheader("📈 Distribution")
            for rel in Relation:
                count = int(np.sum(result.relations == rel.value))
                if count > 0:
                    pct = count / len(result.relations) * 100
                    st.markdown(
                        f"<span style='color:{rel.color}'>■</span> "
                        f"**{rel.label}**: {count:,} ({pct:.1f}%)",
                        unsafe_allow_html=True,
                    )

        # ---- Downloads ----
        st.divider()
        dl_col1, dl_col2, _ = st.columns([1, 1, 3])
        with dl_col1:
            st.download_button(
                "📥 Download PNG",
                data=png_bytes,
                file_name=f"vista_{method_name}.png",
                mime="image/png",
            )
        with dl_col2:
            st.download_button(
                "📥 Download CSV",
                data=_result_to_csv(result),
                file_name=f"vista_{method_name}.csv",
                mime="text/csv",
            )

    else:
        # ---------- Comparison view ----------
        if len(compare_methods) == 0:
            st.info("Select at least one method in the sidebar to compare.")
            return

        st.subheader(f"Comparing {len(compare_methods)} methods")

        results: list[object] = []
        progress_bar = st.progress(0, text="Generating VISTAs…")
        for i, mname in enumerate(compare_methods):
            m_adapter = get_method(mname)
            m_params = m_adapter.default_params()
            result = generate_vista(
                method=mname,
                resolution=resolution,
                reference=reference,
                weights=weights,
                n_criteria=2,
                third_alternative=third_alt,
                progress=False,
                **m_params,
            )
            results.append(result)
            progress_bar.progress(
                (i + 1) / len(compare_methods),
                text=f"Generated {m_adapter.display_name} ({i + 1}/{len(compare_methods)})",
            )
        progress_bar.empty()

        ncols = min(4, len(results))
        fig = plot_vista_comparison(
            results,
            ncols=ncols,
            title="VISTA Method Comparison",
            point_size=max(0.1, 15.0 / (resolution / 10)),
        )

        st.pyplot(fig, use_container_width=True)

        # Render PNG bytes before closing the figure
        comparison_png = _fig_to_png_bytes(fig)
        plt.close(fig)

        # Download comparison PNG
        st.download_button(
            "📥 Download comparison PNG",
            data=comparison_png,
            file_name="vista_comparison.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()
