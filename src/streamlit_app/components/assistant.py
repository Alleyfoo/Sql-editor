from __future__ import annotations

import urllib.parse
from typing import List, Optional

import streamlit as st


def render() -> None:
    transcript: List[dict] = st.session_state.get("transcript", [])
    if not transcript:
        return

    # Handle follow-up chip click via ?fq=N query param
    fq = st.query_params.get("fq")
    if fq is not None:
        bank: dict = st.session_state.get("_followup_bank", {})
        text = bank.get(fq)
        if text:
            st.session_state["nl_prefill"] = text
        del st.query_params["fq"]
        st.rerun()

    n_user = sum(1 for e in transcript if e.get("role") == "user")
    dataset_name = st.session_state.get("dataset_name", "")
    followup_bank: dict = {}
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
            if st.button("Clear", key="assistant_clear", use_container_width=True):
                st.session_state.transcript = []
                st.session_state["_followup_bank"] = {}
                st.rerun()

        # ── Scrollable transcript ─────────────────────────────────────────
        with st.container(key="assistant_scroll"):
            for entry in transcript:
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
                                with st.expander("SQL", expanded=False):
                                    from src.streamlit_app.sql_highlight import render_sql_block
                                    st.markdown(
                                        f'<div style="background:var(--code-bg);border:1px solid #2a2520;'
                                        f'border-radius:var(--r-2);overflow:hidden;margin:2px 0 4px;">'
                                        f'{render_sql_block(sql)}'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                        if det_analysis is not None and not det_analysis.is_empty:
                            followups = _render_det_analysis(det_analysis, fq_idx)
                            for i, q in enumerate(followups):
                                followup_bank[str(fq_idx + i)] = q
                            fq_idx += len(followups)
                        elif old_analysis:
                            followups = _render_legacy_analysis(old_analysis, fq_idx)
                            for i, q in enumerate(followups):
                                followup_bank[str(fq_idx + i)] = q
                            fq_idx += len(followups)

        st.session_state["_followup_bank"] = followup_bank


def _render_det_analysis(det, fq_start: int = 0) -> List[str]:
    """Render a DeterministicAnalysis into headline + cards + prose + chips."""
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
    _render_followup_chips(questions, fq_start)
    return questions


def _render_followup_chips(questions: List[str], fq_start: int) -> None:
    if not questions:
        return
    chips_html = "".join(
        f'<a href="?fq={fq_start + i}" class="followup-chip">'
        f'→ {urllib.parse.escape(q)}'
        f'</a>'
        for i, q in enumerate(questions)
    )
    st.markdown(
        f'<div class="followups-row">{chips_html}</div>',
        unsafe_allow_html=True,
    )


def _render_legacy_analysis(analysis, fq_start: int = 0) -> List[str]:
    """Render the old ResultAnalysis shape (backward compat for pre-4b entries)."""
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
    _render_followup_chips(questions, fq_start)
    return questions
