from __future__ import annotations

import urllib.parse
from typing import List

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

    # Build follow-up bank while rendering (collected this pass, used next)
    followup_bank: dict = {}
    fq_idx = 0

    # Count user messages for the badge
    n_user = sum(1 for e in transcript if e.get("role") == "user")
    dataset_name = st.session_state.get("dataset_name", "")

    st.markdown("---")

    # Section header
    head_col, clear_col = st.columns([1, 0.22])
    with head_col:
        count_html = f'<span class="count">{n_user}</span>' if n_user else ""
        st.markdown(
            f'<div class="assistant-head">'
            f'<span class="label">Assistant</span>'
            f'{count_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with clear_col:
        if st.button("Clear", key="assistant_clear", use_container_width=True):
            st.session_state.transcript = []
            st.session_state["_followup_bank"] = {}
            st.rerun()

    # Transcript entries
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
                analysis = entry.get("analysis")

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
                            st.code(sql, language="sql")

                if analysis:
                    followups = _render_analysis(analysis, fq_idx)
                    for i, q in enumerate(followups):
                        followup_bank[str(fq_idx + i)] = q
                    fq_idx += len(followups)

    st.session_state["_followup_bank"] = followup_bank


def _render_analysis(analysis, fq_start: int = 0) -> List[str]:
    """Render insight cards + follow-up chips. Returns follow-up question texts."""
    insights = analysis.insights[:3] if analysis.insights else []
    if insights:
        cols = st.columns(min(len(insights), 3))
        for i, insight in enumerate(insights):
            with cols[i]:
                # Try to split "Label: Value · detail" for richer display
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
    if questions:
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

    return questions
