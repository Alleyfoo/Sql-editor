from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

from src.history import DEFAULT_HISTORY_PATH
from src.streamlit_app.components.ask import render_quick_queries


@st.dialog("Query Composer", width="large")
def _composer_dialog() -> None:
    """Full query composer rendered inside a modal dialog.

    Flushes any pending widget sync (queued by ask/_sync_composer_widgets)
    before composer.render() instantiates the keyed widgets.
    """
    if pending := st.session_state.pop("_pending_composer_sync", None):
        for _k, _v in pending.items():
            st.session_state[_k] = _v
    from src.streamlit_app.components import composer
    composer.render()


def _render_composer_in_sidebar() -> None:
    """Render the full composer directly in the sidebar tab."""
    if pending := st.session_state.pop("_pending_composer_sync", None):
        for _k, _v in pending.items():
            st.session_state[_k] = _v
    
    schema = st.session_state.get("schema", {})
    if not schema:
        st.markdown(
            '<div style="font-size:12px;color:#B8B0A2;padding:20px 0;">Load a dataset to compose queries.</div>',
            unsafe_allow_html=True,
        )
        return
    
    from src.streamlit_app.components import composer
    composer.render()


def _render_composer_summary() -> None:
    """Mini query-builder card in the sidebar.

    Shows selected columns + active filter count.  An 'Open Composer'
    button launches the full composer in a modal dialog so the narrow
    sidebar never has to host the 5-section form.
    """
    model = st.session_state.get("model")
    schema = st.session_state.get("schema", {})
    has_data = bool(schema)

    st.markdown(
        '<div class="schema-section-head"><span>Query Builder</span></div>',
        unsafe_allow_html=True,
    )

    if not has_data:
        st.markdown(
            '<div style="font-size:12px;color:#B8B0A2;">Load a dataset to compose queries.</div>',
            unsafe_allow_html=True,
        )
        return

    selected = list(model.selected_columns) if model else []
    n_filters = len(model.filters) if model else 0
    n_aggs = len(model.aggregations) if model else 0
    n_order = len(model.order_by) if model else 0

    # Build a compact one-line summary of the active query state
    parts = []
    if selected:
        col_summary = ", ".join(selected[:3])
        if len(selected) > 3:
            col_summary += f" +{len(selected) - 3}"
        parts.append(f"**SELECT** {col_summary}")
    if n_filters:
        parts.append(f"**{n_filters}** filter{'s' if n_filters != 1 else ''}")
    if n_aggs:
        parts.append(f"**{n_aggs}** agg{'s' if n_aggs != 1 else ''}")
    if n_order:
        parts.append(f"**{n_order}** sort")

    if parts:
        st.markdown(
            '<div style="font-size:11.5px;color:#57514A;margin-bottom:8px;line-height:1.6;">'
            + " &middot; ".join(parts)
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:11.5px;color:#B8B0A2;margin-bottom:8px;">No query composed yet.</div>',
            unsafe_allow_html=True,
        )

    if st.button("⚙️ Open Composer", key="sidebar_open_composer", width="stretch"):
        _composer_dialog()


def _render_quick_queries_section(schema: dict, has_data: bool) -> None:
    st.markdown(
        '<div class="schema-section-head"><span>Quick Queries</span></div>',
        unsafe_allow_html=True,
    )
    render_quick_queries(schema, has_data)


def render() -> None:
    # Flush any pending composer widget state queued by ask/_sync_composer_widgets.
    # Must happen before composer.render() instantiates the keyed widgets.
    if pending := st.session_state.pop("_pending_composer_sync", None):
        for _k, _v in pending.items():
            st.session_state[_k] = _v

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

        # Tab system for Schema/Compose/History
        schema_tab, compose_tab, history_tab = st.tabs(["📋 Schema", "🔧 Compose", "⏱ History"])

        with schema_tab:
            tables: Dict[str, Dict[str, str]] = st.session_state.get("tables", {})
            relationships: list = st.session_state.get("relationships", [])
            if tables:
                _render_multi_table_schema(tables, relationships)
                if relationships:
                    st.divider()
                    _render_relationships(relationships)
            elif schema:
                _render_schema_profile(schema)
            st.divider()
            _render_quick_queries_section(schema, has_data=bool(schema))

        with compose_tab:
            _render_composer_in_sidebar()

        with history_tab:
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


def _render_multi_table_schema(tables: Dict[str, Dict[str, str]], relationships: list) -> None:
    """Render schema sections for each table in a multi-table dataset."""
    from src.relationships import get_joinable_columns

    st.markdown(
        '<div class="schema-section-head"><span>Schema</span></div>',
        unsafe_allow_html=True,
    )

    # Build a map of joinable columns
    joinable = get_joinable_columns(relationships) if relationships else {}

    type_chip_class = {"numeric": "type-num", "text": "type-text", "date": "type-date"}
    for table_name, schema in tables.items():
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;letter-spacing:.06em;'
            f'text-transform:uppercase;color:#C2410C;margin:10px 0 4px 0;">'
            f'{table_name}</div>',
            unsafe_allow_html=True,
        )
        table_joinable = joinable.get(table_name, set())
        for col, dtype in schema.items():
            chip_cls = type_chip_class.get(dtype, "type-text")
            chip_label = {"numeric": "num", "text": "text", "date": "date"}.get(dtype, dtype)

            # Add joinable indicator if this column is part of a relationship
            joinable_indicator = ""
            if col in table_joinable:
                joinable_indicator = (
                    '<span class="joinable-chip" title="Joinable column">🔗</span>'
                )

            st.markdown(
                f'<div class="col-row" style="padding:3px 0;">'
                f'<div class="col-row-head">'
                f'<span class="col-name" style="font-size:12px;">{col}</span>'
                f'<span class="type-chip {chip_cls}">{chip_label}</span>'
                f'{joinable_indicator}'
                f'</div></div>',
                unsafe_allow_html=True,
            )


def _render_relationships(relationships: list) -> None:
    """Render detected relationships between tables."""
    from src.relationships import format_relationship_label

    st.markdown(
        '<div class="schema-section-head"><span>Relationships</span></div>',
        unsafe_allow_html=True,
    )

    if not relationships:
        st.markdown(
            '<div style="font-size:12px;color:#B8B0A2;">No relationships detected.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div style="font-size:11px;color:#8E867B;margin-bottom:8px;">'
        f'{len(relationships)} potential join{"s" if len(relationships) != 1 else ""} detected</div>',
        unsafe_allow_html=True,
    )

    for rel in relationships:
        label = format_relationship_label(rel)
        confidence = rel.get("confidence", "medium")

        # Color based on confidence
        if confidence == "high":
            bg_color = "#E1F0E2"
            border_color = "#C8DDC9"
            text_color = "#3F6B45"
        else:
            bg_color = "#FAE7D0"
            border_color = "#E9C79A"
            text_color = "#8A4A11"

        st.markdown(
            f'<div style="padding:8px 10px;margin:4px 0;background:{bg_color};'
            f'border:1px solid {border_color};border-radius:4px;'
            f'font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
            f'color:{text_color};">{label}</div>',
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
