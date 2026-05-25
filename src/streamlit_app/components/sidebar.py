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
        st.markdown(
            '<div class="schema-section-head"><span>Active Dataset</span></div>',
            unsafe_allow_html=True,
        )
        _render_dataset_card(dataset_name, meta, schema)
        st.divider()
        tables: Dict[str, Dict[str, str]] = st.session_state.get("tables", {})
        if tables:
            _render_multi_table_schema(tables)
            st.divider()
        elif schema:
            _render_schema_profile(schema)
            st.divider()
        _render_recent_runs()


def _render_dataset_card(name, meta, schema) -> None:
    if not name:
        st.markdown(
            '<div class="ds-empty">'
            '<div class="ds-empty-ico">&#128194;</div>'
            '<div class="ds-empty-title">No dataset loaded</div>'
            '<div class="ds-empty-sub">Open a CSV to begin</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    rows = meta.get("rows", "—")
    cols = meta.get("cols", "—")
    size_bytes = meta.get("size_bytes", 0)
    size_str = f"{round(size_bytes / 1024, 1)} KB" if size_bytes else "—"
    loaded = meta.get("loaded_at", "")[:16].replace("T", " ") or "—"

    st.markdown(
        f'<div class="ds-card">'
        f'<div class="ds-name">'
        f'<div class="file-ico"></div>'
        f'<div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{name}</div>'
        f'</div>'
        f'<div class="ds-meta">'
        f'<div class="kv"><div class="k">Rows</div><div class="v"><span class="mono">{rows:,}</span></div></div>'
        f'<div class="kv"><div class="k">Columns</div><div class="v"><span class="mono">{cols}</span></div></div>'
        f'<div class="kv"><div class="k">Size</div><div class="v"><span class="mono">{size_str}</span></div></div>'
        f'<div class="kv"><div class="k">Loaded</div><div class="v"><span class="mono">{loaded}</span></div></div>'
        f'</div>'
        f'<div class="badge-row">'
        f'<span class="ds-badge connected"><span class="dot"></span>Connected</span>'
        f'<span class="ds-badge read-only">read-only</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_multi_table_schema(tables: Dict[str, Dict[str, str]]) -> None:
    """Render schema sections for each table in a multi-table dataset."""
    st.markdown(
        '<div class="schema-section-head"><span>Schema</span></div>',
        unsafe_allow_html=True,
    )
    type_chip_class = {"numeric": "type-num", "text": "type-text", "date": "type-date"}
    for table_name, schema in tables.items():
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;letter-spacing:.06em;'
            f'text-transform:uppercase;color:#C2410C;margin:10px 0 4px 0;">'
            f'{table_name}</div>',
            unsafe_allow_html=True,
        )
        for col, dtype in schema.items():
            chip_cls = type_chip_class.get(dtype, "type-text")
            chip_label = {"numeric": "num", "text": "text", "date": "date"}.get(dtype, dtype)
            st.markdown(
                f'<div class="col-row" style="padding:3px 0;">'
                f'<div class="col-row-head">'
                f'<span class="col-name" style="font-size:12px;">{col}</span>'
                f'<span class="type-chip {chip_cls}">{chip_label}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )


def _render_schema_profile(schema: Dict[str, str]) -> None:
    st.markdown(
        '<div class="schema-section-head">'
        '<span>Schema</span>'
        '</div>',
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

    type_chip_class = {
        "numeric": "type-num",
        "text":    "type-text",
        "date":    "type-date",
    }

    for col, dtype in schema.items():
        p = profiles.get(col, {})
        pct = p.get("pct_complete", 100)
        chip_cls = type_chip_class.get(dtype, "type-text")
        chip_label = {"numeric": "num", "text": "text", "date": "date"}.get(dtype, dtype)
        stats_html = _col_stats_html(dtype, p)
        bar_color = {"numeric": "#1D4ED8", "date": "#2F3F70"}.get(dtype, "")
        bar_style = f'style="width:{pct}%;{f" background:{bar_color}" if bar_color else ""}"'

        selected = bool(model and col in (model.selected_columns or []))

        cb_col, label_col = st.columns([0.12, 0.88], gap="small")
        with cb_col:
            st.checkbox(
                label=col,
                value=selected,
                key=f"sel_{col}",
                label_visibility="collapsed",
                on_change=_toggle_column,
                args=(col,),
            )
        with label_col:
            sel_bg = "rgba(194,65,12,0.04)" if selected else "transparent"
            st.markdown(
                f'<div class="col-row{" selected" if selected else ""}" style="background:{sel_bg};">'
                f'<div class="col-row-head">'
                f'<span class="col-name">{col}</span>'
                f'<span class="type-chip {chip_cls}">{chip_label}</span>'
                f'</div>'
                f'<div class="col-stats">{stats_html}'
                f'<span class="micro-bar"><i {bar_style}></i></span>'
                f'</div>'
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


def _fmt_num(v) -> str:
    """Human-readable number for sidebar stats — no scientific notation."""
    if v is None:
        return "—"
    f = float(v)
    if f == int(f):
        return f"{int(f):,}"
    if abs(f) >= 10_000:
        return f"{f:,.0f}"
    if abs(f) >= 100:
        return f"{f:,.1f}"
    return f"{f:.2f}"


def _col_stats_html(dtype: str, p: dict) -> str:
    """Return exactly 2 stat spans before the micro-bar (auto auto 1fr grid)."""
    if dtype == "numeric":
        mn, mx = p.get("min"), p.get("max")
        if mn is not None:
            return (
                f'<span class="stat"><strong>{_fmt_num(mn)}</strong> min</span>'
                f'<span class="stat"><strong>{_fmt_num(mx)}</strong> max</span>'
            )
    elif dtype == "text":
        u = p.get("unique_count")
        pct = p.get("pct_complete", 100)
        if u is not None:
            return (
                f'<span class="stat"><strong>{u}</strong> unique</span>'
                f'<span class="stat"><strong>{pct:.0f}%</strong> full</span>'
            )
    elif dtype == "date":
        mn = p.get("min_date", "")
        mx = p.get("max_date", "")
        if mn:
            return (
                f'<span class="stat"><strong>{mn}</strong></span>'
                f'<span class="stat"><strong>{mx}</strong></span>'
            )
    pct = p.get("pct_complete", 100)
    return f'<span class="stat"><strong>{pct:.0f}%</strong> full</span><span class="stat"></span>'


def _render_recent_runs() -> None:
    st.markdown(
        '<div class="schema-section-head"><span>Recent runs</span></div>',
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

    items_html = ""
    for entry in entries:
        sql_short = entry.get("sql", "")[:60].replace("\n", " ")
        if len(entry.get("sql", "")) > 60:
            sql_short += "…"
        rows = entry.get("rows", "?")
        ts = entry.get("ts", "")[:16].replace("T", " ")
        items_html += (
            f'<div style="padding:6px 0;border-bottom:1px solid #E4DFD2;">'
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
            f'color:#57514A;margin-bottom:3px;word-break:break-word;">{sql_short}</div>'
            f'<div style="font-size:11px;color:#8E867B;">{ts} &nbsp;·&nbsp; {rows} rows</div>'
            f'</div>'
        )
    st.markdown(items_html, unsafe_allow_html=True)
