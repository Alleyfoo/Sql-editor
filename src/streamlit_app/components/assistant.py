from __future__ import annotations

from typing import List, Optional

import streamlit as st


def render() -> None:
    transcript: List[dict] = st.session_state.get("transcript", [])
    if not transcript:
        return

    n_user = sum(1 for e in transcript if e.get("role") == "user")
    dataset_name = st.session_state.get("dataset_name", "")
    fq_idx = 0

    with st.container(key="assistant_panel"):
        # ── Header strip ──────────────────────────────────────────────────
        head_col, clear_col = st.columns([1, 0.22])
        with head_col:
            st.markdown(
                f'<div class="asst-head">'
                f'<span class="asst-head-label">Assistant</span>'
                f'<span class="asst-head-count">'
                f'{n_user} turn{"s" if n_user != 1 else ""}'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with clear_col:
            if st.button("Clear", key="assistant_clear", width='stretch'):
                st.session_state.transcript = []
                st.session_state["_followup_bank"] = {}
                st.rerun()

        # ── Scrollable transcript ─────────────────────────────────────────
        with st.container(key="assistant_scroll"):
            for entry_idx, entry in enumerate(transcript):
                role = entry.get("role", "user")
                if role == "user":
                    with st.chat_message("user"):
                        st.markdown(f"**{entry.get('text', '')}**")
                        ref_html = (
                            f' · <strong>1 reference:</strong> {dataset_name}'
                            if dataset_name else ""
                        )
                        st.markdown(
                            f'<div class="msg-meta">just now{ref_html}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    with st.chat_message("assistant"):
                        error = entry.get("error")
                        routed = entry.get("routed", False)
                        reply = entry.get("reply", "")
                        sql = entry.get("sql", "")
                        det_analysis = entry.get("det_analysis")
                        old_analysis = entry.get("analysis")

                        if error:
                            st.error(error)
                        elif routed:
                            st.markdown(reply)
                            st.markdown(
                                '<span class="pill routed">↷ routed to Python</span>',
                                unsafe_allow_html=True,
                            )
                        else:
                            if reply:
                                st.markdown(reply)
                            if sql:
                                is_quick = entry.get("source") == "quick_query"
                                with st.expander("SQL", expanded=is_quick):
                                    from src.streamlit_app.sql_highlight import render_sql_block
                                    st.markdown(
                                        f'<div style="background:var(--code-bg);border:1px solid #2a2520;'
                                        f'border-radius:var(--r-2);overflow:hidden;margin:2px 0 4px;">'
                                        f'{render_sql_block(sql)}'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )
                                # Inline run button — avoids scrolling to the SQL panel
                                run_col, _spacer = st.columns([0.28, 0.72])
                                with run_col:
                                    if st.button(
                                        "▶ Run",
                                        key=f"inline_run_{entry_idx}",
                                        type="primary",
                                        width="stretch",
                                        disabled=not bool(st.session_state.get("conn")),
                                    ):
                                        _inline_run(sql)

                        if det_analysis is not None and not det_analysis.is_empty:
                            fq_idx = _render_det_analysis(det_analysis, entry_idx, fq_idx)
                        elif old_analysis:
                            fq_idx = _render_legacy_analysis(old_analysis, entry_idx, fq_idx)


def _render_det_analysis(det, entry_idx: int, fq_idx: int) -> int:
    """Render a DeterministicAnalysis into headline + cards + prose + chips.

    Returns the updated fq_idx after rendering follow-up buttons.
    """
    if det.headline:
        st.markdown(
            f'<div class="headline-callout">{det.headline.text}</div>',
            unsafe_allow_html=True,
        )

    insights = det.insights[:3]
    if insights:
        cards_html = ""
        for ins in insights:
            delta_cls = {
                "up":      "delta-up",
                "down":    "delta-down",
                "neutral": "delta-neutral",
            }.get(ins.direction, "delta-neutral")
            cards_html += (
                f'<div class="insight-card">'
                f'<div class="insight-label">{ins.label}</div>'
                f'<div class="insight-value">{ins.value}</div>'
                f'<div class="insight-delta {delta_cls}">{ins.delta}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div class="insights-grid insights-{len(insights)}">{cards_html}</div>',
            unsafe_allow_html=True,
        )

    # Phase 4c: LLM narrative sits between the cards and the follow-up chips
    if det.prose:
        st.markdown(
            f'<div class="analysis-prose">{det.prose}</div>',
            unsafe_allow_html=True,
        )

    if det.warnings:
        for w in det.warnings:
            st.warning(w)

    questions = det.next_questions[:5] if det.next_questions else []
    return _render_followup_chips(questions, entry_idx, fq_idx)


def _render_followup_chips(questions: List[str], entry_idx: int, fq_idx: int) -> int:
    """Render follow-up questions as Streamlit buttons (not <a href> links).

    Returns the updated fq_idx after rendering all buttons.
    Using st.button avoids the URL query-param round-trip that causes
    session-state timing issues and broken chips.
    """
    if not questions:
        return fq_idx
    for i, q in enumerate(questions):
        btn_key = f"fq_{entry_idx}_{fq_idx + i}"
        if st.button(f"→ {q}", key=btn_key):
            st.session_state["nl_text_input"] = q
            st.session_state["nl_auto_submit"] = True
            st.rerun()
    return fq_idx + len(questions)


def _inline_run(sql: str) -> None:
    """Execute SQL from an inline assistant Run button and update results."""
    import time
    from src.executor import ExecutionError, execute
    from src import history

    conn = st.session_state.get("conn")
    if not conn:
        st.toast("No dataset loaded.", icon="⚠️")
        return
    try:
        with st.spinner("Running…"):
            t0 = time.perf_counter()
            df = execute(conn, sql)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        st.session_state.last_sql = sql
        st.session_state.results_df = df
        st.session_state.last_exec_ms = elapsed_ms
        history.log_query(sql, len(df))
        st.rerun()
    except ExecutionError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")


def _render_legacy_analysis(analysis, entry_idx: int, fq_idx: int) -> int:
    """Render the old ResultAnalysis shape (backward compat for pre-4b entries).

    Returns the updated fq_idx after rendering follow-up buttons.
    """
    insights = analysis.insights[:3] if analysis.insights else []
    if insights:
        cols = st.columns(min(len(insights), 3))
        for i, insight in enumerate(insights):
            with cols[i]:
                if ":" in insight:
                    label, rest = insight.split(":", 1)
                    value_part = rest.strip()
                else:
                    label = f"Insight {i + 1}"
                    value_part = insight
                st.markdown(
                    f'<div class="insight-card">'
                    f'<div class="insight-label">{label.strip().upper()}</div>'
                    f'<div class="insight-value">{value_part}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    if analysis.summary:
        st.caption(analysis.summary)

    if analysis.warnings:
        for w in analysis.warnings:
            st.warning(w)

    questions = analysis.next_questions[:3] if analysis.next_questions else []
    return _render_followup_chips(questions, entry_idx, fq_idx)
