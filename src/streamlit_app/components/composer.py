from __future__ import annotations

import urllib.parse
from typing import Dict, List

import streamlit as st

from src.query_model import (
    AGGREGATION_FUNCTIONS,
    ORDER_DIRECTIONS,
    OPERATORS_BY_TYPE,
    Aggregation,
    Filter,
    QueryModel,
)


def render() -> None:
    if not st.session_state.get("schema"):
        st.info("Open a CSV file to start composing a query.")
        return

    schema: Dict[str, str] = st.session_state.schema
    model: QueryModel = st.session_state.model
    cols = list(schema.keys())

    # Handle query param mutations (pill removal, group chip removal)
    _handle_qp(model, cols)

    # Panel header + Reset button
    head_col, btn_col = st.columns([1, 0.28])
    with head_col:
        st.markdown(
            '<div class="cp-panel-head">'
            '<span class="cp-title">Compose</span>'
            '<span class="cp-sub">visual builder · synced with SQL ↔</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with btn_col:
        rc, sc = st.columns(2)
        if rc.button("Reset", key="cp_reset", use_container_width=True):
            from src.streamlit_app import state
            state.reset_query()
            st.rerun()
        sc.button("Save ↗", key="cp_save", disabled=True, use_container_width=True)

    _select_section(schema, model, cols)
    _where_section(schema, model, cols)
    _group_agg_section(schema, model, cols)
    _having_section(schema, model, cols)
    _order_limit_section(schema, model, cols)

    _refresh_sql(model)


# ── helpers ──────────────────────────────────────────────────────────────────

def _handle_qp(model: QueryModel, cols: List[str]) -> None:
    changed = False

    rm_col = st.query_params.get("rm_col")
    if rm_col:
        sel = list(model.selected_columns)
        if rm_col in sel:
            sel.remove(rm_col)
            model.selected_columns = sel
            changed = True
        del st.query_params["rm_col"]

    rm_grp = st.query_params.get("rm_grp")
    if rm_grp:
        grp = list(model.group_by)
        if rm_grp in grp:
            grp.remove(rm_grp)
            model.group_by = grp
            changed = True
        del st.query_params["rm_grp"]

    if changed:
        _refresh_sql(model)
        st.rerun()


def _is_open(key: str, default: bool = True) -> bool:
    return st.session_state.get(f"cs_{key}_open", default)


def _section_header(num: int, title: str, key: str,
                    summary: str = "", count: int | None = None) -> bool:
    is_open = _is_open(key)
    active = "active" if is_open else ""
    count_html = f'<span class="cs-count">{count}</span>' if count is not None else ""
    summary_html = (f'<span class="cs-summary">{summary}</span>') if summary else ""
    chevron = "▲" if is_open else "▼"

    left, right = st.columns([14, 1])
    with left:
        st.markdown(
            f'<div class="cs-head">'
            f'<span class="cs-num {active}">{num}</span>'
            f'<span class="cs-title">{title}</span>'
            f'{count_html}'
            f'{summary_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with right:
        if st.button(chevron, key=f"cs_tog_{key}", help="Toggle section"):
            st.session_state[f"cs_{key}_open"] = not is_open
            st.rerun()
    return is_open


# ── SELECT ────────────────────────────────────────────────────────────────────

def _select_section(schema, model: QueryModel, cols: List[str]) -> None:
    sel = list(model.selected_columns)
    if sel:
        summary = ", ".join(sel[:3]) + ("…" if len(sel) > 3 else "")
    else:
        summary = "all columns (*)"

    is_open = _section_header(1, "SELECT", "select", summary)
    if not is_open:
        return

    with st.container():
        # Render removable pills
        pills_html = ""
        for col in sel:
            enc = urllib.parse.quote(col)
            pills_html += (
                f'<a href="?rm_col={enc}" class="select-pill" title="Remove {col}">'
                f'<span class="grip">⋮⋮</span>{col}'
                f'<span class="px">×</span></a>'
            )

        avail = [c for c in cols if c not in sel]
        if avail:
            pills_html += '<span class="select-pill add">＋ Add column</span>'

        st.markdown(
            f'<div class="select-pills">{pills_html}</div>',
            unsafe_allow_html=True,
        )

        # Add column selectbox (shown only when columns available)
        if avail:
            add_col = st.selectbox(
                "Add column",
                ["— pick to add —"] + avail,
                key="select_add_col",
                label_visibility="collapsed",
            )
            if add_col and add_col != "— pick to add —":
                model.selected_columns = sel + [add_col]
                _refresh_sql(model)
                st.rerun()


# ── WHERE ─────────────────────────────────────────────────────────────────────

def _where_section(schema, model: QueryModel, cols: List[str]) -> None:
    rows: List[dict] = st.session_state.get("where_rows", [])
    count = len(rows) if rows else None
    summary = _filter_summary(rows) if rows else "no filters"

    is_open = _section_header(2, "WHERE", "where", summary, count)
    if not is_open:
        _sync_filters(rows, model.filters)
        return

    with st.container():
        _filter_rows_ui(schema, cols, "where_rows", model.filters)


def _filter_summary(rows: List[dict]) -> str:
    parts = []
    for r in rows[:2]:
        conj = r.get("logical", "AND")
        col = r.get("column", "")
        op = r.get("operator", "=")
        val = r.get("value", "")
        parts.append(f"{conj + ' ' if parts else ''}{col} {op} {val}")
    if len(rows) > 2:
        parts.append(f"+{len(rows)-2} more")
    return " ".join(parts)


def _filter_rows_ui(schema, cols, row_key: str, target_list: list) -> None:
    rows: List[dict] = st.session_state.get(row_key, [])
    to_remove = None

    for i, row in enumerate(rows):
        c1, c2, c3, c4, c5 = st.columns([0.13, 0.28, 0.22, 0.32, 0.05])

        with c1:
            if i == 0:
                st.markdown('<span class="conj-label">WHERE</span>', unsafe_allow_html=True)
            else:
                logical = st.radio(
                    "logic",
                    ["AND", "OR"],
                    index=0 if row.get("logical", "AND") == "AND" else 1,
                    key=f"{row_key}_logic_{i}",
                    horizontal=True,
                    label_visibility="collapsed",
                )
                row["logical"] = logical

        with c2:
            col = st.selectbox(
                "column",
                cols,
                index=cols.index(row["column"]) if row.get("column") in cols else 0,
                key=f"{row_key}_col_{i}",
                label_visibility="collapsed",
            )
            row["column"] = col

        with c3:
            dtype = schema.get(col, "text")
            ops = OPERATORS_BY_TYPE.get(dtype, OPERATORS_BY_TYPE["text"])
            op = st.selectbox(
                "op",
                list(ops),
                index=list(ops).index(row["operator"]) if row.get("operator") in ops else 0,
                key=f"{row_key}_op_{i}",
                label_visibility="collapsed",
            )
            row["operator"] = op

        with c4:
            if op in ("IS NULL", "IS NOT NULL"):
                row["value"] = None
                st.empty()
            elif op == "BETWEEN":
                lo_col, _, hi_col = st.columns([1, 0.05, 1])
                lo = lo_col.text_input("lo", value=str(row.get("value_lo", "")),
                                       key=f"{row_key}_lo_{i}",
                                       label_visibility="collapsed",
                                       placeholder="low")
                hi_col.text_input("and", value="and", disabled=True,
                                  key=f"{row_key}_andsep_{i}",
                                  label_visibility="collapsed")
                hi = st.text_input("hi", value=str(row.get("value_hi", "")),
                                   key=f"{row_key}_hi_{i}",
                                   label_visibility="collapsed",
                                   placeholder="high")
                row["value_lo"] = lo
                row["value_hi"] = hi
                row["value"] = (lo, hi)
            else:
                if dtype == "numeric":
                    val = st.text_input(
                        "value",
                        value=str(row.get("value", "")),
                        key=f"{row_key}_val_{i}",
                        label_visibility="collapsed",
                    )
                    try:
                        row["value"] = float(val) if "." in val else int(val)
                    except (ValueError, TypeError):
                        row["value"] = val
                else:
                    row["value"] = st.text_input(
                        "value",
                        value=str(row.get("value", "")),
                        key=f"{row_key}_val_{i}",
                        label_visibility="collapsed",
                    )

        with c5:
            if st.button("×", key=f"{row_key}_rm_{i}"):
                to_remove = i

    if to_remove is not None:
        rows.pop(to_remove)
        st.session_state[row_key] = rows
        st.rerun()

    if st.button("＋ Add condition", key=f"{row_key}_add"):
        rows.append({"column": cols[0], "operator": "=", "value": "", "logical": "AND"})
        st.session_state[row_key] = rows
        st.rerun()

    _sync_filters(rows, target_list)


def _sync_filters(rows: List[dict], target_list: list) -> None:
    target_list.clear()
    for row in rows:
        try:
            target_list.append(Filter(
                column=row["column"],
                operator=row["operator"],
                value=row.get("value"),
                logical=row.get("logical", "AND"),
            ))
        except Exception:
            pass


# ── GROUP BY / AGGREGATE ──────────────────────────────────────────────────────

def _group_agg_section(schema, model: QueryModel, cols: List[str]) -> None:
    agg_rows: List[dict] = st.session_state.get("agg_rows", [])
    n_group = len(model.group_by)
    n_agg = len(agg_rows)
    count = n_group + n_agg if (n_group or n_agg) else None
    summary = (
        f"{n_group} group{'s' if n_group != 1 else ''}, "
        f"{n_agg} agg{'s' if n_agg != 1 else ''}"
        if (n_group or n_agg) else "no grouping"
    )

    is_open = _section_header(3, "GROUP BY · Aggregate", "group", summary, count)
    if not is_open:
        return

    with st.container():
        # Group by as chips
        st.markdown(
            '<div style="font-size:10.5px;color:var(--ink-3);text-transform:uppercase;'
            'letter-spacing:.08em;margin-bottom:5px;">Group by</div>',
            unsafe_allow_html=True,
        )
        chips_html = ""
        for g in model.group_by:
            enc = urllib.parse.quote(g)
            chips_html += (
                f'<a href="?rm_grp={enc}" class="group-chip">'
                f'{g} <span class="gx">×</span></a>'
            )
        avail_grp = [c for c in cols if c not in model.group_by]
        if avail_grp:
            chips_html += '<span class="select-pill add" style="height:24px;font-size:11.5px;">＋ Add group</span>'
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px;">'
            f'{chips_html}</div>',
            unsafe_allow_html=True,
        )
        if avail_grp:
            add_grp = st.selectbox(
                "Add group",
                ["— pick to group —"] + avail_grp,
                key="group_add_col",
                label_visibility="collapsed",
            )
            if add_grp and add_grp != "— pick to group —":
                model.group_by = list(model.group_by) + [add_grp]
                _refresh_sql(model)
                st.rerun()

        # Aggregations
        st.markdown(
            '<div style="font-size:10.5px;color:var(--ink-3);text-transform:uppercase;'
            'letter-spacing:.08em;margin-top:10px;margin-bottom:6px;">Aggregations</div>',
            unsafe_allow_html=True,
        )
        to_remove = None
        for i, row in enumerate(agg_rows):
            c1, c2, c3, c4, c5 = st.columns([0.18, 0.35, 0.05, 0.35, 0.07])
            with c1:
                fn = st.selectbox(
                    "fn",
                    list(AGGREGATION_FUNCTIONS),
                    index=list(AGGREGATION_FUNCTIONS).index(row.get("fn", "COUNT"))
                    if row.get("fn") in AGGREGATION_FUNCTIONS else 0,
                    key=f"agg_fn_{i}",
                    label_visibility="collapsed",
                )
                row["fn"] = fn
            with c2:
                agg_cols = ["*"] + cols if fn == "COUNT" else cols
                col_val = st.selectbox(
                    "col",
                    agg_cols,
                    index=agg_cols.index(row["col"]) if row.get("col") in agg_cols else 0,
                    key=f"agg_col_{i}",
                    label_visibility="collapsed",
                )
                row["col"] = col_val
            with c3:
                st.markdown('<div style="text-align:center;padding-top:6px;color:var(--ink-3);">→</div>',
                            unsafe_allow_html=True)
            with c4:
                alias = st.text_input(
                    "alias",
                    value=row.get("alias", ""),
                    key=f"agg_alias_{i}",
                    label_visibility="collapsed",
                    placeholder="alias",
                )
                row["alias"] = alias or None
            with c5:
                if st.button("×", key=f"agg_rm_{i}"):
                    to_remove = i

        if to_remove is not None:
            agg_rows.pop(to_remove)
            st.session_state.agg_rows = agg_rows
            st.rerun()

        if st.button("＋ Add aggregation", key="agg_add"):
            agg_rows.append({"fn": "COUNT", "col": "*", "alias": None})
            st.session_state.agg_rows = agg_rows
            st.rerun()

        model.aggregations = []
        for row in agg_rows:
            try:
                model.aggregations.append(
                    Aggregation(column=row["col"], function=row["fn"], alias=row.get("alias"))
                )
            except Exception:
                pass


# ── HAVING ────────────────────────────────────────────────────────────────────

def _having_section(schema, model: QueryModel, cols: List[str]) -> None:
    having_rows: List[dict] = st.session_state.get("having_rows", [])
    has_group = bool(model.group_by)
    count = len(having_rows) if having_rows else None
    summary = _filter_summary(having_rows) if having_rows else ("add GROUP BY first" if not has_group else "no filters")

    is_open = _section_header(4, "HAVING", "having", summary, count)
    if not is_open:
        _sync_filters(having_rows, model.having)
        return

    with st.container():
        if not has_group:
            st.caption("Add a GROUP BY column above to enable HAVING.")
        else:
            _filter_rows_ui(schema, cols, "having_rows", model.having)


# ── ORDER BY / LIMIT ──────────────────────────────────────────────────────────

def _order_limit_section(schema, model: QueryModel, cols: List[str]) -> None:
    order_rows: List[dict] = st.session_state.get("order_rows", [])
    limit_val = model.limit if model.limit is not None else 1000
    count = len(order_rows) if order_rows else None
    summary = (
        ", ".join(f"{r['col']} {r.get('dir','ASC')}" for r in order_rows[:2])
        + (f" · LIMIT {limit_val}" if order_rows else f"LIMIT {limit_val}")
    )

    is_open = _section_header(5, "ORDER BY · LIMIT", "order", summary, count)
    if not is_open:
        _sync_order(order_rows, model)
        return

    with st.container():
        to_remove = None
        for i, row in enumerate(order_rows):
            c1, c2, c3 = st.columns([0.55, 0.35, 0.10])
            with c1:
                col = st.selectbox(
                    "col",
                    cols,
                    index=cols.index(row["col"]) if row.get("col") in cols else 0,
                    key=f"order_col_{i}",
                    label_visibility="collapsed",
                )
                row["col"] = col
            with c2:
                direction = st.radio(
                    "dir",
                    ["DESC", "ASC"],
                    index=0 if row.get("dir", "DESC") == "DESC" else 1,
                    key=f"order_dir_{i}",
                    horizontal=True,
                    label_visibility="collapsed",
                )
                row["dir"] = direction
            with c3:
                if st.button("×", key=f"order_rm_{i}"):
                    to_remove = i

        if to_remove is not None:
            order_rows.pop(to_remove)
            st.session_state.order_rows = order_rows
            st.rerun()

        if st.button("＋ Add sort", key="order_add"):
            order_rows.append({"col": cols[0], "dir": "DESC"})
            st.session_state.order_rows = order_rows
            st.rerun()

        st.markdown('<div style="margin-top:10px;padding-top:10px;border-top:1px dashed var(--line);"></div>',
                    unsafe_allow_html=True)
        limit_col, hint_col = st.columns([0.3, 0.7])
        with limit_col:
            limit = st.number_input(
                "LIMIT",
                min_value=0,
                max_value=1_000_000,
                step=100,
                value=limit_val,
                key="limit_input",
            )
        with hint_col:
            st.markdown(
                '<div style="padding-top:28px;font-size:11.5px;color:var(--ink-3);">0 = no limit</div>',
                unsafe_allow_html=True,
            )
        model.limit = int(limit) if limit > 0 else None

        _sync_order(order_rows, model)


def _sync_order(order_rows: List[dict], model: QueryModel) -> None:
    model.order_by = []
    for row in order_rows:
        try:
            model.order_by.append((row["col"], row.get("dir", "DESC")))
        except Exception:
            pass


# ── SQL refresh ───────────────────────────────────────────────────────────────

def _refresh_sql(model: QueryModel) -> None:
    try:
        st.session_state.last_sql = model.to_sql()
    except ValueError as exc:
        st.session_state.last_sql = f"-- {exc}"
