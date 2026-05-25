from __future__ import annotations

from typing import List, Optional

import streamlit as st

from src.heuristic_nl import (
    HEURISTIC_FAST_PATH_THRESHOLD,
    HeuristicResult,
    parse_heuristic,
)
from src.query_model import QueryModel
from src.streamlit_app import state
from src.streamlit_app.llm_health import probe_ollama
from src.streamlit_app.quick_queries import QuickQuery, build_quick_queries, build_supply_chain_quick_queries


def render() -> None:
    has_data = bool(st.session_state.get("conn"))
    schema = st.session_state.get("schema", {})

    # Follow-up chip buttons set nl_text_input + nl_auto_submit directly in
    # session state before rerun, so we just read auto_submit here.
    auto_submit = st.session_state.pop("nl_auto_submit", False)

    with st.container(border=True):
        st.markdown('<span class="ask-anchor" style="display:none"></span>',
                    unsafe_allow_html=True)
        # Header line: pulse dot + label + model info
        transcript = st.session_state.get("transcript", [])
        n_turns = sum(1 for m in transcript if m.get("role") == "user")
        # Provider label: read config.yaml once per session, cache in session state.
        # header.py invalidates "_cached_provider" whenever the provider changes.
        if "_cached_provider" not in st.session_state:
            from src.config import load_config as _load_config
            from src.llm.natural_language import load_llm_config as _load_llm_cfg
            st.session_state["_cached_provider"] = (
                _load_llm_cfg(_load_config() or {}).provider or "ollama"
            ).lower()
        _provider_label = (
            "remote Ollama"
            if st.session_state["_cached_provider"] == "ollama_remote"
            else "local Ollama"
        )
        st.markdown(
            f'<div class="ask-model-line">'
            f'<span class="pulse"></span>'
            f'<strong style="color:var(--ink);font-size:12px;">Ask your data</strong>'
            f'<span style="margin-left:auto;opacity:0.7;">'
            f'{_provider_label} · {n_turns} turn{"s" if n_turns != 1 else ""} of context'
            f"</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        col_input, col_ask, col_analyze = st.columns([1, 0.13, 0.21])
        with col_input:
            text = st.text_input(
                "ask",
                label_visibility="collapsed",
                placeholder=(
                    "e.g. monthly revenue trend 2024, "
                    "top 10 customers by margin, "
                    "or compare EMEA vs AMER year-over-year"
                ),
                disabled=not has_data,
                key="nl_text_input",
            )
        with col_ask:
            ask_clicked = st.button(
                "Ask",
                width='stretch',
                disabled=not has_data,
                key="btn_ask",
            )
        with col_analyze:
            analyze_clicked = st.button(
                "Ask + Analyze",
                width='stretch',
                type="primary",
                disabled=not has_data,
                key="btn_analyze",
            )

    # Quick queries outside the ask card so the card stays compact
    _render_quick_queries(schema, has_data)

    if (ask_clicked or analyze_clicked or auto_submit) and text.strip():
        _handle_ask(text.strip(), run_llm_analysis=analyze_clicked)


def _render_quick_queries(schema: dict, has_data: bool) -> None:
    if has_data and st.session_state.get("tables"):
        quicks: List[QuickQuery] = build_supply_chain_quick_queries()
    elif has_data:
        quicks: List[QuickQuery] = build_quick_queries(schema)
    else:
        quicks: List[QuickQuery] = []

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
                    width='stretch',
                ):
                    _apply_quick_query(qq, schema)


def _apply_quick_query(qq: QuickQuery, schema: dict) -> None:
    """Build a QueryModel from the template and sync into the composer.

    Does not execute — user reviews SQL and clicks Run.
    For raw-SQL quick queries (qq.sql is set), bypasses the QueryModel
    and puts the SQL directly into the preview.
    """
    # Raw SQL path — used for JOIN / multi-table queries
    if qq.sql:
        ss = st.session_state
        ss.last_sql = qq.sql
        ss.results_df = None
        ss.last_exec_ms = None
        state.append_transcript({
            "role": "assistant",
            "reply": f"{qq.label} — review the SQL and press Run.",
            "sql": qq.sql,
            "det_analysis": None,
            "error": None,
            "source": "quick_query",
        })
        st.rerun()
        return

    try:
        model: QueryModel = qq.build(schema)
    except Exception as exc:
        st.toast(f"Could not apply quick query: {exc}", icon="⚠️")
        return

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
        ss.last_sql = model.to_sql()
    except ValueError as exc:
        ss.last_sql = f"-- {exc}"
    ss.results_df = None
    ss.last_exec_ms = None

    state.append_transcript(
        {
            "role": "assistant",
            "reply": (model.reply or qq.label) + " — review the SQL and press Run.",
            "sql": ss.last_sql if not ss.last_sql.startswith("-- ") else "",
            "det_analysis": None,
            "error": None,
            "source": "quick_query",
        }
    )
    st.rerun()


def _handle_ask(text: str, run_llm_analysis: bool) -> None:
    from src.config import load_config
    from src.llm.natural_language import LLMError, RouteToPythonError, nl_to_query_model
    from src.streamlit_app import state

    # For multi-table datasets, scope NL queries to the primary table so the
    # LLM doesn't hallucinate cross-table columns into a single "data" table.
    tables: dict = st.session_state.get("tables", {})
    if tables:
        primary_table = next(iter(tables))
        schema = tables[primary_table]
    else:
        primary_table = "data"
        schema = st.session_state.schema
    history = st.session_state.nl_history
    model = st.session_state.model

    state.append_transcript({"role": "user", "text": text})

    # Heuristic fast-path before LLM.  Force a fresh probe on each Ask
    # so the status pill reflects current Ollama state, not a stale cache.
    probe = probe_ollama(force=True)
    fast_path = parse_heuristic(text, schema)
    if fast_path.parsed and fast_path.confidence >= HEURISTIC_FAST_PATH_THRESHOLD:
        _apply_heuristic_result(
            text, fast_path, run_llm_analysis=run_llm_analysis, llm_available=probe.ok
        )
        return

    if not probe.ok:
        if _try_heuristic(text, schema, run_llm_analysis):
            return

    try:
        from src.llm.natural_language import load_llm_config
        cfg_data = load_config()
        llm_cfg = load_llm_config(cfg_data)
        with st.spinner("Thinking…"):
            result_model = nl_to_query_model(
                text,
                schema,
                selected_columns=list(model.selected_columns),
                history=history,
                config=llm_cfg,
                table_name=primary_table,
            )
    except RouteToPythonError as exc:
        state.append_transcript(
            {
                "role": "assistant",
                "reply": str(exc),
                "sql": "",
                "det_analysis": None,
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
                "det_analysis": None,
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
                "det_analysis": None,
                "error": f"Could not generate SQL: {exc}",
            }
        )
        state.push_nl_history(text, reply)
        st.rerun()
        return

    det_analysis = _execute_and_compute(
        sql, result_model,
        text=text, schema=schema,
        run_llm_analysis=run_llm_analysis,
    )

    state.append_transcript(
        {
            "role": "assistant",
            "reply": reply,
            "sql": sql,
            "det_analysis": det_analysis,
            "error": None,
        }
    )
    state.push_nl_history(text, reply)
    st.rerun()


def _try_heuristic(text: str, schema: dict, run_llm_analysis: bool) -> bool:
    result: HeuristicResult = parse_heuristic(text, schema)
    _apply_heuristic_result(text, result, run_llm_analysis=run_llm_analysis, llm_available=False)
    return True


def _apply_heuristic_result(
    text: str,
    result: HeuristicResult,
    *,
    run_llm_analysis: bool,
    llm_available: bool,
) -> None:
    if not result.parsed or result.model is None:
        state.append_transcript(
            {
                "role": "assistant",
                "reply": "",
                "sql": "",
                "det_analysis": None,
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
        return

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
                "det_analysis": None,
                "error": f"Heuristic produced invalid SQL: {exc}",
                "source": "heuristic",
            }
        )
        st.rerun()
        return

    det_analysis = _execute_and_compute(
        sql, model,
        text=text,
        schema=st.session_state.schema,
        run_llm_analysis=(run_llm_analysis and llm_available),
    )

    state.append_transcript(
        {
            "role": "assistant",
            "reply": (
                f"{model.reply or 'Heuristic match'} "
                f"(via heuristic, confidence {result.confidence:.2f})."
            ),
            "sql": sql,
            "det_analysis": det_analysis,
            "error": None,
            "source": "heuristic",
            "heuristic_reasoning": result.reasoning,
        }
    )
    state.push_nl_history(text, model.reply or "Heuristic match.")
    st.rerun()


def _execute_and_compute(
    sql: str,
    model: QueryModel,
    *,
    text: str = "",
    schema: Optional[dict] = None,
    run_llm_analysis: bool = False,
):
    """Execute SQL, compute deterministic insights, optionally run deep analysis.

    Returns a DeterministicAnalysis (possibly enriched), or None on execution failure.
    When run_llm_analysis=True, AnalysisCoordinator runs a full analysis pipeline:
    profiling → analysis plan → LLM insight report → chart specs.
    Chart specs are stored in session state for the results Chart tab.
    Falls back silently to deterministic-only results on any failure.
    """
    import time
    from src.executor import execute
    from src import history
    from src.streamlit_app.insight_engine import compute_insights

    conn = st.session_state.conn
    source_row_count = st.session_state.get("dataset_meta", {}).get("rows")

    # Clear chart specs from any previous analysis run
    st.session_state.pop("last_chart_specs", None)

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
        det = compute_insights(df, model, source_row_count=source_row_count)
    except Exception:
        det = None

    # Deep analysis via AnalysisCoordinator — only on "Ask + Analyze"
    if run_llm_analysis and df is not None and len(df) > 0:
        try:
            from src.config import load_config
            from src.ingestion import infer_schema
            from src.llm.natural_language import load_llm_config
            from src.analysis_lane import AnalysisCoordinator

            with st.spinner("Analysing…"):
                cfg = load_llm_config(load_config())
                result_schema = infer_schema(df)
                coordinator = AnalysisCoordinator(llm_config=cfg)
                run = coordinator.run(
                    question=text or sql,
                    distilled_df=df,
                    schema=result_schema,
                    prior_analysis_handle=None,
                    expected_followup_from=None,
                )

            # Store chart specs for the results Chart tab
            if run.charts:
                st.session_state["last_chart_specs"] = run.charts

            # Enrich the DeterministicAnalysis with AnalysisRun output
            det = _merge_analysis_run(det, run)
        except Exception:
            pass  # coordinator failure never blocks the UI

    return det


def _merge_analysis_run(det, run) -> "DeterministicAnalysis":
    """Merge AnalysisCoordinator output into a DeterministicAnalysis.

    - prose  ← InsightReport summary (replaces any prior LLM narrative)
    - next_questions ← extended with follow-up hints from insights
    - warnings ← extended with guardrail errors (capped at 2)
    """
    from src.streamlit_app.insight_engine import DeterministicAnalysis

    prose = (run.report.summary or "").strip() or None

    # Extract follow-up question hints from insight evidence fields
    extra_qs: List[str] = []
    for ins in run.report.insights[:3]:
        ef = ins.evidence_fields
        if ef and len(ef) > 0:
            extra_qs.append(f"Break down {ef[0]} further")
    # Also surface blocked claims as informational follow-ups
    for bc in run.report.blocked_claims[:1]:
        extra_qs.append(bc)

    questions = list(det.next_questions if det else [])
    for q in extra_qs:
        if q and q not in questions:
            questions.append(q)
    questions = questions[:5]

    warnings = list(det.warnings if det else [])
    warnings.extend(run.guardrail_errors[:2])

    if det is None:
        return DeterministicAnalysis(
            prose=prose,
            next_questions=questions,
            warnings=warnings,
        )
    return DeterministicAnalysis(
        headline=det.headline,
        insights=det.insights,
        next_questions=questions,
        warnings=warnings,
        prose=prose,
        pattern=det.pattern,
    )
