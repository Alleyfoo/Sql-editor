from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

from src.history import DEFAULT_HISTORY_PATH


def render() -> None:
    schema: Dict[str, str] = st.session_state.get("schema", {})
    meta: dict = st.session_state.get("dataset_meta", {})
    dataset_name: Optional[str] = st.session_state.get("dataset_name")

    with st.sidebar:
        _render_dataset_card(dataset_name, meta, schema)
        st.divider()
        if schema:
            _render_schema_profile(schema)
            st.divider()
        _render_recent_runs()


def _render_dataset_card(name, meta, schema) -> None:
    if not name:
        st.markdown(
            '<div style="padding:16px;background:#F4F1EA;border-radius:6px;'
            'border:1px dashed #E4DFD2;text-align:center;color:#8E867B;">'
            '<div style="font-size:22px;margin-bottom:6px;">📂</div>'
            '<div style="font-size:13px;font-weight:500;">No dataset loaded</div>'
            '<div style="font-size:12px;margin-top:4px;">Open a CSV to begin</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    rows = meta.get("rows", "—")
    cols = meta.get("cols", "—")
    size_kb = round(meta.get("size_bytes", 0) / 1024, 1)
    loaded = meta.get("loaded_at", "")[:16].replace("T", " ")

    st.markdown(
        f'<div style="padding:4px 0 8px 0;">'
        f'<div style="font-weight:600;font-size:13px;margin-bottom:6px;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>'
        f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;">'
        f'<span class="pill connected">&#9679; Connected</span>'
        f'<span class="pill readonly">read-only</span>'
        f'</div>'
        f'<div style="font-size:12px;color:#57514A;line-height:1.8;">'
        f'<b>{rows:,}</b> rows &nbsp;·&nbsp; <b>{cols}</b> columns'
        f'&nbsp;·&nbsp; {size_kb} KB<br/>Loaded {loaded}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _render_schema_profile(schema: Dict[str, str]) -> None:
    st.markdown(
        '<div style="font-size:11px;font-weight:600;letter-spacing:.07em;'
        'text-transform:uppercase;color:#8E867B;margin-bottom:8px;">Columns</div>',
        unsafe_allow_html=True,
    )
    model = st.session_state.get("model")
    df = st.session_state.get("dataset_df")
    profiles = st.session_state.get("_col_profiles", {})

    if df is not None and not profiles:
        from src.streamlit_app.profile import profile_from_df
        file_hash = st.session_state.get("dataset_meta", {}).get("file_hash", "x")
        profiles = profile_from_df(df, schema, file_hash)
        st.session_state["_col_profiles"] = profiles

    chip_colors = {
        "numeric": ("var(--info-soft)", "var(--info)"),
        "text":    ("var(--good-soft)", "var(--good)"),
        "date":    ("var(--accent-soft)", "var(--accent-ink)"),
    }

    for col, dtype in schema.items():
        bg, fg = chip_colors.get(dtype, ("#eee", "#333"))
        p = profiles.get(col, {})
        pct = p.get("pct_complete", 100)
        stats_html = _stats_html(dtype, p)
        extra_html = _extra_col_html(dtype, p)

        selected = model and col in (model.selected_columns or [])

        cb_col, label_col = st.columns([0.08, 0.92], gap="small")
        with cb_col:
            st.checkbox(
                label=col,
                value=bool(selected),
                key=f"sel_{col}",
                label_visibility="collapsed",
                on_change=_toggle_column,
                args=(col,),
            )
        with label_col:
            st.markdown(
                f'<div class="schema-row" style="background:{"rgba(194,65,12,0.04)" if selected else "transparent"};'
                f'border-radius:4px;padding:3px 6px;">'
                f'<div class="schema-row-head">'
                f'<span class="col-name" style="font-weight:{"500" if selected else "400"}">{col}</span>'
                f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
                f'padding:1px 5px;border-radius:3px;font-weight:500;'
                f'background:{bg};color:{fg};">{dtype}</span>'
                f'</div>'
                f'<div class="schema-row-stats">{stats_html}</div>'
                f'<div style="margin-top:2px;">'
                f'<div class="completeness-bar-wrap">'
                f'<div class="completeness-bar-fill" style="width:{pct}%;"></div>'
                f'</div></div>'
                f'{extra_html}'
                f'</div>',
                unsafe_allow_html=True,
            )


def _toggle_column(col: str) -> None:
    model = st.session_state.get("model")
    if model is None:
        return
    cols = list(model.selected_columns)
    if col in cols:
        cols.remove(col)
    else:
        cols.append(col)
    model.selected_columns = cols


def _stats_html(dtype: str, p: dict) -> str:
    if dtype == "numeric":
        mn, mx = p.get("min"), p.get("max")
        if mn is not None:
            return f"{mn:,.2g} – {mx:,.2g} &nbsp;·&nbsp; {p.get('pct_complete', 100)}% complete"
    elif dtype == "text":
        u = p.get("unique_count")
        if u is not None:
            return f"{u} unique &nbsp;·&nbsp; {p.get('pct_complete', 100)}% complete"
    elif dtype == "date":
        mn = p.get("min_date", "")
        mx = p.get("max_date", "")
        if mn:
            return f"{mn} – {mx}"
    return f"{p.get('pct_complete', 100)}% complete"


def _extra_col_html(dtype: str, p: dict) -> str:
    if dtype == "numeric":
        counts = p.get("hist_counts", [])
        if not counts:
            return ""
        max_c = max(counts) if counts else 1
        bar_parts = []
        for c in counts:
            cls = ' class="hi"' if c == max_c else ""
            pct = max(round(c / max_c * 100), 4)
            bar_parts.append(f'<span{cls} style="height:{pct}%;"></span>')
        bars = "".join(bar_parts)
        return f'<div class="sparkline-wrap">{bars}</div>'
    elif dtype == "text":
        top = p.get("top_values", {})
        if not top:
            return ""
        chips = "".join(
            f'<span class="sample-v">{v}</span>'
            for v in list(top.keys())[:3]
        )
        return f'<div class="sample-row">{chips}</div>'
    return ""


def _render_recent_runs() -> None:
    st.markdown(
        '<div style="font-size:11px;font-weight:600;letter-spacing:.07em;'
        'text-transform:uppercase;color:#8E867B;margin-bottom:8px;">Recent runs</div>',
        unsafe_allow_html=True,
    )
    path = DEFAULT_HISTORY_PATH
    if not Path(path).exists():
        st.markdown('<div style="font-size:12px;color:#B8B0A2;">No queries yet.</div>',
                    unsafe_allow_html=True)
        return

    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    entries = list(reversed(entries[-8:]))

    for entry in entries:
        sql_short = entry.get("sql", "")[:60].replace("\n", " ")
        if len(entry.get("sql", "")) > 60:
            sql_short += "…"
        rows = entry.get("rows", "?")
        ts = entry.get("ts", "")[:16].replace("T", " ")
        st.markdown(
            f'<div style="padding:5px 0;border-bottom:1px solid #E4DFD2;font-size:12px;">'
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
            f'color:#57514A;margin-bottom:2px;">{sql_short}</div>'
            f'<div style="color:#8E867B;">{ts} &nbsp;·&nbsp; {rows} rows</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
