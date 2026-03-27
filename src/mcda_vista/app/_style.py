"""CSS injection for reduced whitespace and cleaner layout."""

from __future__ import annotations

import streamlit as st

_COMPACT_CSS = """
<style>
    /* Reduce top padding of the main block */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    /* Tighten sidebar section spacing */
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }
    /* Reduce spacing after headers in sidebar */
    [data-testid="stSidebar"] h2 {
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
    }
    [data-testid="stSidebar"] h3 {
        margin-top: 0.4rem;
        margin-bottom: 0.15rem;
    }
    /* Reduce tab-content top gap */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.5rem;
    }
</style>
"""


def inject_compact_css() -> None:
    """Inject CSS that reduces Streamlit's default whitespace."""
    st.markdown(_COMPACT_CSS, unsafe_allow_html=True)
