"""Query Studio — main tab (data + query app).

One of three panels in the top-level tabbed app
(``streamlit_app.py``).  Behaviour is identical to the previous
single-page body.  Session state is shared with the LLM SQL
Assistant and Workflow tabs.
"""

from __future__ import annotations

import streamlit as st

from src.streamlit_app import state
from src.streamlit_app.styles import inject_css
from src.streamlit_app.components import (
    ask,
    assistant,
    header,
    results,
    schema_strip,
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
            <span>Query Studio · demo</span>
            <span>built with Streamlit</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    inject_css()
    state.init()

    sidebar.render()
    header.render()

    center_col, chat_col = st.columns([2, 1], gap="medium")

    with center_col:
        schema_strip.render()
        results.render()
        sql_preview.render()

    with chat_col:
        with st.container(key="asst_rail"):
            with st.container(border=True, key="chat_panel"):
                st.markdown(
                    '<div class="chat-panel-header">💬 Assistant</div>',
                    unsafe_allow_html=True,
                )
                ask.render()
                assistant.render()

    _render_statusbar()
