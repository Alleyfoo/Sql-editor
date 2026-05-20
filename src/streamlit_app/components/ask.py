from __future__ import annotations

from typing import List, Optional

import streamlit as st

from src.streamlit_app import state


_GENERIC_CHIPS = [
    "Show all columns",
    "Count rows",
    "Show top 10 rows",
]


def render() -> None:
    has_data = bool(st.session_state.get("conn"))
    schema = st.session_state.get("schema", {})

    # Handle chip click via query param
    chip_param = st.query_params.get("chip")
    if chip_param is not None:
        chips_for_param = list(_GENERIC_CHIPS)
        if schema:
            text_cols = [c for c, t in schema.items() if t == "text"]
            num_cols = [c for c, t in schema.items() if t == "numeric"]
            if text_cols and num_cols:
                chips_for_param.append(f"top 10 {text_cols[0]} by {num_cols[0]}")
        try:
            idx = int(chip_param)
            if 0 <= idx < len(chips_for_param):
                st.session_state["nl_prefill"] = chips_for_param[idx]
        except (ValueError, TypeError):
            pass
        del st.query_params["chip"]
        st.rerun()

    # Pre-fill from follow-up chip click
    prefill = st.session_state.get("nl_prefill", "")
    if prefill:
        st.session_state["nl_prefill"] = ""

    # Model indicator line
    transcript = st.session_state.get("transcript", [])
    n_turns = sum(1 for m in transcript if m.get("role") == "user")
    st.markdown(
        f'<div class="ask-model-line">'
        f'<span class="pulse"></span>Ask your data'
        f'<span style="margin-left:12px;opacity:0.7;">local model · {n_turns} turn{"s" if n_turns != 1 else ""} of context</span>'
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

    # Suggestion chips — inline HTML pills using ?chip=N query param
    chips = list(_GENERIC_CHIPS)
    if schema:
        text_cols = [c for c, t in schema.items() if t == "text"]
        num_cols = [c for c, t in schema.items() if t == "numeric"]
        if text_cols and num_cols:
            chips.append(f"top 10 {text_cols[0]} by {num_cols[0]}")

    disabled_class = " disabled" if not has_data else ""
    chip_links = "".join(
        f'<a href="?chip={i}" class="ask-chip{disabled_class}">↳ {chip}</a>'
        for i, chip in enumerate(chips)
    )
    st.markdown(
        f'<div class="ask-chips-row">{chip_links}</div>',
        unsafe_allow_html=True,
    )

    if (ask_clicked or analyze_clicked) and text.strip():
        _handle_ask(text.strip(), run_and_analyze=analyze_clicked)


def _handle_ask(text: str, run_and_analyze: bool) -> None:
    from src.config import load_config
    from src.llm.natural_language import LLMError, RouteToPythonError, nl_to_query_model
    from src.streamlit_app import state

    schema = st.session_state.schema
    history = st.session_state.nl_history
    model = st.session_state.model

    state.append_transcript({"role": "user", "text": text})

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
        state.append_transcript({
            "role": "assistant",
            "reply": str(exc),
            "sql": "",
            "analysis": None,
            "error": None,
            "routed": True,
        })
        st.rerun()
        return
    except LLMError as exc:
        state.append_transcript({
            "role": "assistant",
            "reply": "",
            "sql": "",
            "analysis": None,
            "error": str(exc),
        })
        st.toast(str(exc), icon="⚠️")
        st.rerun()
        return

    st.session_state.model = result_model
    reply = result_model.reply or "Query updated."

    try:
        sql = result_model.to_sql()
        st.session_state.last_sql = sql
    except ValueError as exc:
        state.append_transcript({
            "role": "assistant",
            "reply": reply,
            "sql": "",
            "analysis": None,
            "error": f"Could not generate SQL: {exc}",
        })
        state.push_nl_history(text, reply)
        st.rerun()
        return

    analysis = None
    if run_and_analyze:
        analysis = _run_and_analyze(text, sql, schema)

    state.append_transcript({
        "role": "assistant",
        "reply": reply,
        "sql": sql,
        "analysis": analysis,
        "error": None,
    })
    state.push_nl_history(text, reply)
    st.rerun()


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
