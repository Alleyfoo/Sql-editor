from __future__ import annotations

import time

import streamlit as st

from src.executor import ExecutionError, execute
from src import history


def render() -> None:
    sql = st.session_state.get("last_sql", "")
    has_conn = bool(st.session_state.get("conn"))
    sql_valid = sql and not sql.startswith("--")

    # Estimate row count from last results
    results_df = st.session_state.get("results_df")
    n_rows_est = f"{len(results_df):,}" if results_df is not None else "—"

    st.markdown(
        f"""
        <div class="sql-toolbar">
          <span class="dot"></span>
          <span class="stat">SELECT-only</span>
          <span class="sep">·</span>
          <span class="stat"><strong>{n_rows_est}</strong> rows</span>
          <span class="sep">·</span>
          <span class="stat">SQLite in-memory</span>
          <div class="toolbar-actions">
            <button class="ghost-btn" onclick="
              var code = document.querySelector('[data-testid=stCode] code');
              if(code) navigator.clipboard.writeText(code.innerText);
            ">Copy</button>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.code(
        sql if sql else "-- compose a query above --",
        language="sql",
        line_numbers=True,
    )

    run_col, explain_col, badge_col = st.columns([0.22, 0.18, 0.60])
    with run_col:
        run = st.button(
            "▶ Run query",
            type="primary",
            use_container_width=True,
            disabled=not (has_conn and sql_valid),
            key="run_button",
        )
    with explain_col:
        explain = st.button(
            "Explain",
            use_container_width=True,
            disabled=not sql_valid,
            key="explain_button",
        )
    with badge_col:
        st.markdown(
            '<div style="padding-top:6px;display:flex;align-items:center;gap:12px;">'
            '<span class="safety-badge">&#128274; SELECT-only &nbsp;·&nbsp; read-only conn</span>'
            '<span style="font-size:11px;color:#8E867B;font-family:\'IBM Plex Mono\',monospace;">'
            '⌘ ↵ to run</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    if run:
        _run_query(sql)

    if explain:
        with st.expander("Query explanation", expanded=True):
            _explain_sql(sql)


def _run_query(sql: str) -> None:
    conn = st.session_state.conn
    try:
        with st.spinner("Running…"):
            t0 = time.perf_counter()
            df = execute(conn, sql)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        st.session_state.results_df = df
        st.session_state.last_exec_ms = elapsed_ms
        history.log_query(sql, len(df))
        st.rerun()
    except ExecutionError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")


def _explain_sql(sql: str) -> None:
    lines = sql.split()
    clauses = []
    current = []
    keywords = {"SELECT", "FROM", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT"}
    for word in lines:
        if word.upper() in keywords and current:
            clauses.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        clauses.append(" ".join(current))

    descriptions = {
        "SELECT": "Chooses which columns to return.",
        "FROM": "Specifies the source table.",
        "WHERE": "Filters rows before grouping.",
        "GROUP BY": "Groups rows sharing the same values.",
        "HAVING": "Filters groups after aggregation.",
        "ORDER BY": "Sorts the result set.",
        "LIMIT": "Caps the number of rows returned.",
    }

    for clause in clauses:
        keyword = clause.split()[0].upper()
        desc = descriptions.get(keyword, "")
        st.markdown(f"**`{keyword}`** — {desc}")
        st.code(clause, language="sql")
