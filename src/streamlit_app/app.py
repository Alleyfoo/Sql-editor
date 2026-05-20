from __future__ import annotations

import streamlit as st

from src.streamlit_app import state
from src.streamlit_app.styles import inject_css
from src.streamlit_app.components import (
    ask,
    assistant,
    composer,
    header,
    results,
    sidebar,
    sql_preview,
)


def _render_statusbar() -> None:
    from src import history
    from pathlib import Path
    conn = st.session_state.get("conn")
    route = st.session_state.get("last_route", "—")
    hist_path = str(history.DEFAULT_HISTORY_PATH)
    connected = bool(conn)
    dot_style = "" if connected else 'style="background:#B91C1C;"'
    conn_label = "Connected to in-memory SQLite" if connected else "No dataset loaded"

    st.markdown(
        f"""
        <div class="app-statusbar">
          <span class="dot" {dot_style}></span>
          <span>{conn_label}</span>
          <span>route: <strong>{route}</strong></span>
          <span>history: <strong>{hist_path}</strong></span>
          <div class="sb-right">
            <span>v0.5 · Phase 3</span>
            <span>built with Streamlit</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run() -> None:
    st.set_page_config(
        page_title="Query Studio",
        page_icon="▪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    state.init()

    sidebar.render()
    header.render()
    ask.render()
    assistant.render()

    composer_col, preview_col = st.columns([1.15, 1], gap="medium")
    with composer_col:
        st.markdown(
            '<div style="font-size:11px;font-weight:600;letter-spacing:.07em;'
            'text-transform:uppercase;color:#8E867B;margin-bottom:8px;">Query Composer</div>',
            unsafe_allow_html=True,
        )
        composer.render()

    with preview_col:
        sql_preview.render()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;font-weight:600;letter-spacing:.07em;'
        'text-transform:uppercase;color:#8E867B;margin-bottom:8px;">Results</div>',
        unsafe_allow_html=True,
    )
    results.render()

    _render_statusbar()
