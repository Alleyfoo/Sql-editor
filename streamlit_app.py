"""Query Studio — Streamlit multipage entry point.

Registers pages under ``st.navigation`` and runs the selected one.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Query Studio",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded",
)

page = st.navigation(
    [
        st.Page(
            "src/streamlit_app/pages/studio.py",
            title="Studio",
            default=True,
        ),
        st.Page(
            "src/streamlit_app/pages/llm_assistant.py",
            title="LLM SQL Assistant",
        ),
    ]
)
page.run()
