"""Horizontal column-header strip shown above the assistant panel.

Each chip represents one column from the active dataset.  When the current
QueryModel touches a column the chip lights up and shows a small badge
indicating the operation (SELECT, WHERE, GROUP, SUM/COUNT/…, ORDER).
This gives users an at-a-glance view of what the query is doing and which
fields the LLM chose to access.
"""
from __future__ import annotations

import html
from typing import Dict, List

import streamlit as st


# ── Operation badge colours ────────────────────────────────────────────────
_OP_COLOUR: Dict[str, str] = {
    "SEL":   "#1D4ED8",   # blue    — in SELECT
    "WHERE": "#C2410C",   # orange  — filtered
    "GRP":   "#7C3AED",   # purple  — GROUP BY
    "SORT":  "#0891B2",   # teal    — ORDER BY
    "SUM":   "#059669",   # green   — aggregation
    "COUNT": "#059669",
    "AVG":   "#059669",
    "MIN":   "#059669",
    "MAX":   "#059669",
    "CNT":   "#059669",   # COUNT DISTINCT abbreviated
}

_OP_PRIORITY = ["WHERE", "SEL", "GRP", "SUM", "COUNT", "AVG", "MIN", "MAX", "CNT", "SORT"]


def _ops_for_model(model) -> Dict[str, List[str]]:
    """Return {column: [op_badge, ...]} for the current QueryModel."""
    if model is None:
        return {}
    ops: Dict[str, List[str]] = {}

    for col in (model.selected_columns or []):
        ops.setdefault(col, []).append("SEL")

    for f in (model.filters or []):
        col = f.column
        if "WHERE" not in ops.get(col, []):
            ops.setdefault(col, []).append("WHERE")

    for col in (model.group_by or []):
        if "GRP" not in ops.get(col, []):
            ops.setdefault(col, []).append("GRP")

    for agg in (model.aggregations or []):
        if agg.column == "*":
            continue
        fn = agg.function.upper()
        badge = fn[:3] if fn != "COUNT DISTINCT" else "CNT"
        if badge not in ops.get(agg.column, []):
            ops.setdefault(agg.column, []).append(badge)

    for col, _ in (model.order_by or []):
        if "SORT" not in ops.get(col, []):
            ops.setdefault(col, []).append("SORT")

    return ops


def _chip_html(col: str, dtype: str, col_ops: List[str]) -> str:
    """Build HTML for a single column chip."""
    col_esc = html.escape(col)

    # Determine highlight colour from the highest-priority operation
    primary_colour = None
    for op in _OP_PRIORITY:
        if op in col_ops:
            primary_colour = _OP_COLOUR[op]
            break

    type_short = {"numeric": "num", "text": "txt", "date": "date"}.get(dtype, dtype)

    if primary_colour:
        # Active chip
        badge_html = ""
        for op in col_ops:
            c = _OP_COLOUR.get(op, "#666")
            badge_html += (
                f'<span style="background:{c};color:#fff;font-size:9px;font-weight:700;'
                f'letter-spacing:.04em;padding:1px 4px;border-radius:3px;margin-left:3px;">'
                f'{op}</span>'
            )
        return (
            f'<span title="{col_esc} · {dtype}" style="'
            f'display:inline-flex;align-items:center;gap:2px;'
            f'padding:3px 8px 3px 7px;border-radius:5px;'
            f'background:{primary_colour}18;border:1px solid {primary_colour}55;'
            f'font-size:11.5px;font-weight:600;color:{primary_colour};'
            f'white-space:nowrap;cursor:default;">'
            f'{col_esc}{badge_html}'
            f'</span>'
        )
    else:
        # Inactive chip
        type_colour = {"numeric": "#1D4ED8", "text": "#57514A", "date": "#2F3F70"}.get(dtype, "#57514A")
        return (
            f'<span title="{col_esc} · {dtype}" style="'
            f'display:inline-flex;align-items:center;gap:4px;'
            f'padding:3px 8px;border-radius:5px;'
            f'background:#F0EDE7;border:1px solid #E0DBD3;'
            f'font-size:11.5px;color:#8E867B;'
            f'white-space:nowrap;cursor:default;">'
            f'{col_esc}'
            f'<span style="font-size:9px;color:{type_colour};opacity:0.7;">{type_short}</span>'
            f'</span>'
        )


def render() -> None:
    schema: Dict[str, str] = st.session_state.get("schema", {})
    tables: Dict[str, Dict[str, str]] = st.session_state.get("tables", {})
    model = st.session_state.get("model")

    if not schema:
        return

    active_ops = _ops_for_model(model)

    # Build chip rows — grouped by table for multi-table datasets
    sections: List[tuple[str, Dict[str, str]]] = (
        list(tables.items()) if tables else [("", schema)]
    )

    rows_html = ""
    for table_name, tbl_schema in sections:
        chips = ""
        for col, dtype in tbl_schema.items():
            chips += _chip_html(col, dtype, active_ops.get(col, [])) + " "

        if table_name:
            rows_html += (
                f'<span style="font-size:10px;font-weight:700;letter-spacing:.07em;'
                f'text-transform:uppercase;color:#C2410C;margin-right:6px;'
                f'white-space:nowrap;">{html.escape(table_name)}</span>'
                f'<span style="color:#E0DBD3;margin-right:6px;">·</span>'
            )
        rows_html += chips

        if table_name and table_name != sections[-1][0]:
            rows_html += '<span style="display:inline-block;width:12px;"></span>'

    # Legend for active operations
    present_ops = {op for ops in active_ops.values() for op in ops}
    legend_html = ""
    if present_ops:
        legend_html = '<span style="margin-left:16px;display:inline-flex;align-items:center;gap:6px;">'
        for op in _OP_PRIORITY:
            if op in present_ops:
                c = _OP_COLOUR[op]
                legend_html += (
                    f'<span style="font-size:9px;background:{c};color:#fff;font-weight:700;'
                    f'padding:1px 5px;border-radius:3px;">{op}</span>'
                )
        legend_html += '</span>'

    st.markdown(
        f'<div style="overflow-x:auto;padding:6px 0 8px;'
        f'border-bottom:1px solid #E4DFD2;margin-bottom:10px;'
        f'display:flex;align-items:center;flex-wrap:nowrap;gap:4px;">'
        f'{rows_html}'
        f'{legend_html}'
        f'</div>',
        unsafe_allow_html=True,
    )
