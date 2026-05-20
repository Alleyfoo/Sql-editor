from __future__ import annotations

from typing import List, Optional

import streamlit as st

from src.heuristic_nl import HeuristicResult, parse_heuristic
from src.query_model import QueryModel
from src.streamlit_app import state
from src.streamlit_app.llm_health import probe_ollama
from src.streamlit_app.quick_queries import QuickQuery, build_quick_queries


def render() -> None:
    has_data = bool(st.session_state.get("conn"))
    schema = st.session_state.get("schema", {})

    # Pre-fill from follow-up chip click (legacy NL flow)
    prefill = st.session_state.get("nl_prefill", "")
    if prefill:
        st.session_state["nl_prefill"] = ""

    with st.container(border=True):
        # Header line: pulse dot + label + model info
        transcript = st.session_state.get("transcript", [])
        n_turns = sum(1 for m in transcript if m.get("role") == "user")
        st.markdown(
            f'<div class="ask-model-line">'
            f'<span class="pulse"></span>'
            f'<strong style="color:var(--ink);font-size:12px;">Ask your data</strong>'
            f'<span style="margin-left:auto;opacity:0.7;">'
            f'local model · {n_turns} turn{"s" if n_turns != 1 else ""} of context'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_input, col_ask, col_analyze = st.columns([1, 0.13, 0.21])
        with col_input:
            text = st.text_input(
                "ask",
                value=prefill,
                label_visibility="collapsed",
                placeholder="Ask in plain English…  e.g. show top 10 by revenue",
                disabled=not has_data,
                key="nl_text_input",
            )
        with col_ask:
            ask_clicked = st.button(
                "Ask",
                use_container_width=True,
                disabled=not has_data,
                key="btn_ask",
            )
        with col_analyze:
            analyze_clicked = st.button(
                "Ask + Analyze",
                use_container_width=True,
                type="primary",
                disabled=not has_data,
                key="btn_analyze",
            )

        # Quick queries · runs offline (no LLM dependency)
        _render_quick_queries(schema, has_data)

    if (ask_clicked or analyze_clicked) and text.strip():
        _handle_ask(text.strip(), run_and_analyze=analyze_clicked)


def _render_quick_queries(schema: dict, has_data: bool) -> None:
    quicks: List[QuickQuery] = build_quick_queries(schema) if has_data else []

    st.markdown(
        '<div class="ask-model-line" style="margin-top:6px;">'
        '<span style="opacity:0.7;">Quick queries · runs offline</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    if not has_data:
        st.caption(
            "Open a CSV (or load the demo dataset) to see quick-query templates."
        )
        return
    if not quicks:
        st.caption("No quick-query templates fit this schema.")
        return

    # Render up to 4 buttons per row.
    per_row = 4
    for i in range(0, len(quicks), per_row):
        batch = quicks[i : i + per_row]
        cols = st.columns(len(batch))
        for col, qq in zip(cols, batch):
            with col:
                if st.button(
                    qq.label,
                    key=qq.key,
                    help=qq.description,
                    use_container_width=True,
                ):
                    _apply_quick_query(qq, schema)


def _apply_quick_query(qq: QuickQuery, schema: dict) -> None:
    """Build a QueryModel from the template and sync it into the composer.

    Does not execute. The user reviews the SQL preview and clicks Run.
    """
    try:
        model: QueryModel = qq.build(schema)
    except Exception as exc:
        st.toast(f"Could not apply quick query: {exc}", icon="⚠️")
        return

    ss = st.session_state
    ss.model = model

    # Sync sub-row session state used by the composer UI.
    ss.where_rows = [
        {
            "column": f.column,
            "operator": f.operator,
            "value": f.value,
            "logical": f.logical or "AND",
        }
        for f in model.filters
    ]
    ss.having_rows = [
        {
            "column": f.column,
            "operator": f.operator,
            "value": f.value,
            "logical": f.logical or "AND",
        }
        for f in model.having
    ]
    ss.order_rows = [
        {"col": col, "dir": direction} for col, direction in model.order_by
    ]
    ss.agg_rows = [
        {"fn": agg.function, "col": agg.column, "alias": agg.alias}
        for agg in model.aggregations
    ]

    # Refresh SQL preview; clear stale results.
    try:
        ss.last_sql = model.to_sql()
    except ValueError as exc:
        ss.last_sql = f"-- {exc}"
    ss.results_df = None
    ss.last_exec_ms = None

    # Surface what we did in the transcript so the user has context.
    state.append_transcript(
        {
            "role": "assistant",
            "reply": (model.reply or qq.label) + " — review the SQL and press Run.",
            "sql": ss.last_sql if not ss.last_sql.startswith("-- ") else "",
            "analysis": None,
            "error": None,
            "source": "quick_query",
        }
    )
    st.rerun()


def _handle_ask(text: str, run_and_analyze: bool) -> None:
    from src.config import load_config
    from src.llm.natural_language import LLMError, RouteToPythonError, nl_to_query_model
    from src.streamlit_app import state

    schema = st.session_state.schema
    history = st.session_state.nl_history
    model = st.session_state.model

    state.append_transcript({"role": "user", "text": text})

    # Offline fast-path: when the cached probe says Ollama is unreachable,
    # try the deterministic heuristic parser before bothering the network.
    if not probe_ollama().ok:
        if _try_heuristic(text, schema, run_and_analyze):
            return
        # Heuristic didn't catch it either — fall through to the LLM call so
        # the user still gets a clear "Ollama unreachable" error message
        # rather than a silent failure.

    try:
        cfg = load_config()
        with st.spinner("Thinking…"):
            result_model = nl_to_query_model(
                text,
                schema,
                selected_columns=list(model.selected_columns),
                history=history,
            )
    except RouteToPythonError as exc:
        state.append_transcript(
            {
                "role": "assistant",
                "reply": str(exc),
                "sql": "",
                "analysis": None,
                "error": None,
                "routed": True,
            }
        )
        st.rerun()
        return
    except LLMError as exc:
        state.append_transcript(
            {
                "role": "assistant",
                "reply": "",
                "sql": "",
                "analysis": None,
                "error": str(exc),
            }
        )
        st.toast(str(exc), icon="⚠️")
        st.rerun()
        return

    st.session_state.model = result_model
    reply = result_model.reply or "Query updated."

    try:
        sql = result_model.to_sql()
        st.session_state.last_sql = sql
    except ValueError as exc:
        state.append_transcript(
            {
                "role": "assistant",
                "reply": reply,
                "sql": "",
                "analysis": None,
                "error": f"Could not generate SQL: {exc}",
            }
        )
        state.push_nl_history(text, reply)
        st.rerun()
        return

    analysis = None
    if run_and_analyze:
        analysis = _run_and_analyze(text, sql, schema)

    state.append_transcript(
        {
            "role": "assistant",
            "reply": reply,
            "sql": sql,
            "analysis": analysis,
            "error": None,
        }
    )
    state.push_nl_history(text, reply)
    st.rerun()


def _try_heuristic(text: str, schema: dict, run_and_analyze: bool) -> bool:
    """Attempt to satisfy ``text`` using the offline heuristic parser.

    Returns ``True`` if a model was produced and applied (the caller should
    then return early); ``False`` if confidence was too low and the LLM
    path should still be tried.
    """
    result: HeuristicResult = parse_heuristic(text, schema)
    if not result.parsed or result.model is None:
        # Surface why the heuristic gave up, so the eventual LLM error
        # makes more sense in context.
        state.append_transcript(
            {
                "role": "assistant",
                "reply": "",
                "sql": "",
                "analysis": None,
                "error": (
                    "LLM offline and the offline heuristic couldn't parse this "
                    f"(confidence {result.confidence:.2f}). "
                    "Try a Quick query below or simplify the request."
                ),
                "source": "heuristic",
                "heuristic_reasoning": result.reasoning,
            }
        )
        st.toast("Heuristic couldn't parse — see assistant log.", icon="⚠️")
        st.rerun()
        return True  # we surfaced an error; caller should not double-report

    model: QueryModel = result.model
    ss = st.session_state
    ss.model = model
    ss.where_rows = [
        {
            "column": f.column,
            "operator": f.operator,
            "value": f.value,
            "logical": f.logical or "AND",
        }
        for f in model.filters
    ]
    ss.having_rows = [
        {
            "column": f.column,
            "operator": f.operator,
            "value": f.value,
            "logical": f.logical or "AND",
        }
        for f in model.having
    ]
    ss.order_rows = [
        {"col": col, "dir": direction} for col, direction in model.order_by
    ]
    ss.agg_rows = [
        {"fn": agg.function, "col": agg.column, "alias": agg.alias}
        for agg in model.aggregations
    ]

    try:
        sql = model.to_sql()
        ss.last_sql = sql
    except ValueError as exc:
        state.append_transcript(
            {
                "role": "assistant",
                "reply": "",
                "sql": "",
                "analysis": None,
                "error": f"Heuristic produced invalid SQL: {exc}",
                "source": "heuristic",
            }
        )
        st.rerun()
        return True

    analysis = None
    if run_and_analyze:
        # Execute via the same read-only path; skip LLM analysis since
        # we're explicitly offline.
        analysis = _run_only(sql)

    state.append_transcript(
        {
            "role": "assistant",
            "reply": (
                f"{model.reply or 'Heuristic match'} "
                f"(confidence {result.confidence:.2f})."
            ),
            "sql": sql,
            "analysis": analysis,
            "error": None,
            "source": "heuristic",
            "heuristic_reasoning": result.reasoning,
        }
    )
    state.push_nl_history(text, model.reply or "Heuristic match.")
    st.rerun()
    return True


def _run_only(sql: str):
    """Execute ``sql`` against the read-only connection, no LLM analysis."""
    import time
    from src.executor import execute
    from src import history

    conn = st.session_state.conn
    try:
        with st.spinner("Running…"):
            t0 = time.perf_counter()
            df = execute(conn, sql)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        st.session_state.results_df = df
        st.session_state.last_exec_ms = elapsed_ms
        history.log_query(sql, len(df))
    except Exception as exc:
        st.toast(f"Execution error: {exc}", icon="⚠️")
    return None


def _run_and_analyze(text: str, sql: str, schema: dict):
    import time
    from src.executor import execute
    from src import history
    from src.llm.result_analysis import analyze_result_with_llm, AnalysisError

    conn = st.session_state.conn
    try:
        with st.spinner("Running…"):
            t0 = time.perf_counter()
            df = execute(conn, sql)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        st.session_state.results_df = df
        st.session_state.last_exec_ms = elapsed_ms
        history.log_query(sql, len(df))
    except Exception as exc:
        st.toast(f"Execution error: {exc}", icon="⚠️")
        return None

    try:
        with st.spinner("Analyzing…"):
            return analyze_result_with_llm(text, sql, df, schema)
    except (AnalysisError, Exception):
        return None
