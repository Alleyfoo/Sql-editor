from __future__ import annotations

import time

import streamlit as st

from src.executor import ExecutionError, execute
from src import history
from src.streamlit_app.sql_highlight import render_sql_block


def render() -> None:
    sql = st.session_state.get("last_sql", "")
    has_conn = bool(st.session_state.get("conn"))
    sql_valid = bool(sql) and not sql.startswith("--")

    results_df = st.session_state.get("results_df")
    n_rows_est = f"{len(results_df):,}" if results_df is not None else "—"

    # Dark SQL panel — toolbar + hand-rolled highlighted code in one HTML block
    st.markdown(
        f"""
        <div class="sql-panel">
          <div class="sql-toolbar">
            <span class="dot"></span>
            <span class="stat">SELECT-only</span>
            <span class="sep">·</span>
            <span class="stat"><strong>{n_rows_est}</strong> rows</span>
            <span class="sep">·</span>
            <span class="stat">SQLite in-memory</span>
            <div class="toolbar-actions">
              <button class="ghost-btn"
                onclick="navigator.clipboard.writeText(
                  document.querySelector('.sql-block').innerText
                )">Copy</button>
            </div>
          </div>
          {render_sql_block(sql)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Run bar — native Streamlit widgets below the panel
    run_col, explain_col, badge_col = st.columns([0.22, 0.18, 0.60])
    with run_col:
        run = st.button(
            "▶ Run query",
            type="primary",
            width='stretch',
            disabled=not (has_conn and sql_valid),
            key="run_button",
        )
    with explain_col:
        explain = st.button(
            "Explain",
            width='stretch',
            disabled=not sql_valid,
            key="explain_button",
        )
    with badge_col:
        st.markdown(
            '<div style="padding-top:6px;display:flex;align-items:center;gap:14px;">'
            '<span class="safety-badge">&#128274; SELECT-only &nbsp;·&nbsp; read-only conn</span>'
            '<span style="font-size:11px;color:#8E867B;font-family:\'IBM Plex Mono\',monospace;">'
            '&#8984; &#8629; to run</span>'
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
    """Produce a specific plain-English breakdown of the current query.

    Uses the QueryModel from session state when available so the explanation
    refers to actual column names, values, and aggregations rather than
    generic clause descriptions.
    """
    model = st.session_state.get("model")
    if model is not None and hasattr(model, "table"):
        _explain_from_model(model, sql)
    else:
        _explain_from_sql_text(sql)


def _explain_from_model(model, sql: str) -> None:
    """Generate explanation from the structured QueryModel."""
    steps = []

    # ── Source ──────────────────────────────────────────────────────────────
    dataset_meta = st.session_state.get("dataset_meta", {})
    row_count = dataset_meta.get("rows")
    row_hint = f" ({row_count:,} rows)" if row_count else ""
    steps.append(("📂 Source", f"Reading from **`{model.table}`**{row_hint}."))

    # ── Filters (WHERE) ──────────────────────────────────────────────────────
    if model.filters:
        parts = []
        for f in model.filters:
            op = f.operator.upper()
            if op in ("IS NULL", "IS NOT NULL"):
                parts.append(f"`{f.column}` {op.lower()}")
            elif op == "BETWEEN" and isinstance(f.value, tuple):
                parts.append(f"`{f.column}` between `{f.value[0]}` and `{f.value[1]}`")
            elif op == "LIKE":
                parts.append(f"`{f.column}` matches `{f.value}`")
            else:
                parts.append(f"`{f.column}` {op} `{f.value}`")
        joined = f" **{model.filters[1].logical}** ".join(parts) if len(parts) > 1 else parts[0]
        steps.append(("🔍 Filter", f"Keep only rows where {joined}."))

    # ── Aggregations + GROUP BY ──────────────────────────────────────────────
    if model.aggregations:
        agg_parts = []
        for a in model.aggregations:
            fn = a.function.upper()
            col = a.column
            alias = a.alias
            if fn == "COUNT" and col == "*":
                label = "count all rows"
            elif fn == "COUNT DISTINCT":
                label = f"count unique **`{col}`**"
            else:
                label = f"{fn.lower()} of **`{col}`**"
            if alias:
                label += f" → `{alias}`"
            agg_parts.append(label)

        agg_desc = "Calculate: " + ", ".join(agg_parts) + "."
        if model.group_by:
            agg_desc += f" Grouped by **`{'`, `'.join(model.group_by)}`**."
        steps.append(("∑ Aggregate", agg_desc))

    elif model.group_by:
        steps.append(("⬡ Group", f"Group rows by **`{'`, `'.join(model.group_by)}`**."))

    # ── Selected columns ─────────────────────────────────────────────────────
    if model.selected_columns:
        non_grouped = [
            c for c in model.selected_columns
            if c not in (model.group_by or [])
            and not any(
                (a.alias or a.display_name) == c for a in (model.aggregations or [])
            )
        ]
        if non_grouped:
            steps.append(("📋 Columns", f"Return **`{'`, `'.join(non_grouped)}`**."))

    # ── HAVING ───────────────────────────────────────────────────────────────
    if model.having:
        having_parts = []
        for h in model.having:
            having_parts.append(f"`{h.column}` {h.operator} `{h.value}`")
        steps.append(("▽ Having", f"After grouping, keep only where {', '.join(having_parts)}."))

    # ── ORDER BY ─────────────────────────────────────────────────────────────
    if model.order_by:
        sort_parts = [
            f"**`{col}`** {'↑ ascending' if d.upper() == 'ASC' else '↓ descending'}"
            for col, d in model.order_by
        ]
        steps.append(("↕ Sort", "Sort by " + ", then ".join(sort_parts) + "."))

    # ── LIMIT ─────────────────────────────────────────────────────────────────
    if model.limit:
        steps.append(("✂ Limit", f"Return at most **{model.limit:,}** rows."))

    # ── Render ───────────────────────────────────────────────────────────────
    for icon_label, desc in steps:
        icon, label = icon_label.split(" ", 1)
        st.markdown(
            f'<div style="display:flex;gap:10px;align-items:flex-start;'
            f'margin-bottom:8px;padding:8px 12px;background:#F7F5F0;'
            f'border-radius:6px;border-left:3px solid #C2410C;">'
            f'<span style="font-size:15px;line-height:1.4;">{icon}</span>'
            f'<div><span style="font-size:10px;font-weight:700;letter-spacing:.07em;'
            f'text-transform:uppercase;color:#8E867B;">{label}</span>'
            f'<div style="font-size:13px;color:#1A1714;margin-top:2px;">{desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


def _explain_from_sql_text(sql: str) -> None:
    """Fallback: split by SQL keywords and show each clause."""
    lines = sql.split()
    clauses, current = [], []
    keywords = {"SELECT", "FROM", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT"}
    for word in lines:
        if word.upper() in keywords and current:
            clauses.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        clauses.append(" ".join(current))

    for clause in clauses:
        keyword = clause.split()[0].upper()
        st.code(clause, language="sql")
