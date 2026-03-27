"""Shared helpers for the VISTA dashboard."""

from __future__ import annotations

import io
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

from mcda_vista._constants import POINT_SIZE_BASE, POINT_SIZE_DIVISOR, POINT_SIZE_MIN
from mcda_vista.core import VistaResult, generate_vista
from mcda_vista.methods.base import MethodAdapter


def hashable_params(**kwargs: object) -> tuple[tuple[str, object], ...]:
    """Convert keyword arguments to a hashable, sorted tuple of pairs."""
    return tuple(sorted(kwargs.items()))


@st.cache_data(show_spinner=False)
def cached_generate_vista(
    method: str,
    resolution: int,
    reference: tuple[float, ...],
    weights: tuple[float, ...],
    n_criteria: int = 2,
    third_alternative: tuple[float, ...] | None = None,
    _params_key: tuple[tuple[str, object], ...] = (),
    **method_params: Any,
) -> VistaResult:
    """Thin caching wrapper around :func:`generate_vista`.

    *reference*, *weights* and *third_alternative* are accepted as tuples
    (hashable) and converted to lists before forwarding.
    """
    return generate_vista(
        method=method,
        resolution=resolution,
        reference=list(reference),
        weights=list(weights),
        n_criteria=n_criteria,
        third_alternative=list(third_alternative) if third_alternative else None,
        progress=False,
        **method_params,
    )

def render_method_params(
    adapter: MethodAdapter,
    key_prefix: str = "",
) -> dict[str, Any]:
    """Auto-generate sidebar widgets from a method's ``param_space()``."""
    space = adapter.param_space()
    defaults = adapter.default_params()
    params: dict[str, Any] = {}

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
            raw = [
                spec["min"],
                spec["max"],
                spec.get("step", 0.01),
                spec.get("default", defaults.get(pname, spec["min"])),
            ]
            use_int = all(
                isinstance(v, int) or (isinstance(v, float) and v == int(v))
                for v in raw
            )
            cast = int if use_int else float
            params[pname] = st.slider(
                label,
                min_value=cast(spec["min"]),
                max_value=cast(spec["max"]),
                value=cast(spec.get("default", defaults.get(pname, spec["min"]))),
                step=cast(spec.get("step", 1 if use_int else 0.01)),
                key=key,
            )
        else:
            params[pname] = st.number_input(
                label,
                value=float(spec.get("default", defaults.get(pname, 0.0))),
                key=key,
            )

    return params


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    """Render a Matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def result_to_csv(result: VistaResult) -> str:
    """Convert a VistaResult to CSV text."""
    header_parts = [f"c{i + 1}" for i in range(result.grid.shape[1])]
    header_parts.append("relation")
    lines = [",".join(header_parts)]
    for row, rel in zip(result.grid, result.relations):
        vals = ",".join(f"{v:.6f}" for v in row)
        lines.append(f"{vals},{int(rel)}")
    return "\n".join(lines)


def auto_point_size(resolution: int) -> float:
    """Compute scatter marker area from resolution."""
    return max(POINT_SIZE_MIN, POINT_SIZE_BASE / (resolution / POINT_SIZE_DIVISOR))
