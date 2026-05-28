from __future__ import annotations

from typing import Dict, List

import streamlit as st

from src.query_model import (
    AGGREGATION_FUNCTIONS,
    JOIN_TYPES,
    OPERATORS_BY_TYPE,
    Aggregation,
    Filter,
    Join,
    QueryModel,
)

def render() -> None:
    if not st.session_state.get("schema"):
        st.info("Open a CSV file to start composing a query.")
        return

    schema: Dict[str, str] = st.session_state.schema
    model: QueryModel = st.session_state.model
    tables: Dict[str, Dict[str, str]] = st.session_state.get("tables", {})
    relationships: list = st.session_state.get("relationships", [])

    # Build column list: use qualified names (table.column) for multi-table
    if tables:
        cols = [f"{table}.{col}" for table, schema_dict in tables.items() for col in schema_dict.keys()]
    else:
        cols = list(schema.keys())

    with st.container(key="composer_panel"):
        # Header row — inside the panel so it shares the border
        head_col, btn_col = st.columns([1, 0.32], gap="small")
        with head_col:
            st.markdown(
                '<div class="cp-head-text">'
                '<span class="cp-title">Compose</span>'
                '<span class="cp-sub">visual builder &middot; synced &harr; SQL</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        with btn_col:
            rc, sc = st.columns(2, gap="small")
            if rc.button("Reset", key="cp_reset", width='stretch'):
                from src.streamlit_app import state
                state.reset_query()
                st.rerun()
            sc.button("Save ↗", key="cp_save", disabled=True, width='stretch')

        _select_section(schema, model, cols)
        if tables:
            _join_section(tables, model, relationships)
        _where_section(schema, model, cols)
        _group_agg_section(schema, model, cols)
        _having_section(schema, model, cols)
        _order_limit_section(schema, model, cols)

    _refresh_sql(model)


# ── helpers ───────────────────────────────────────────────────────────────────

def _section(num: int, title: str, summary: str = "",
             count: int | None = None, *, expanded: bool = True):
    count_part = f"  ·  `{count}`" if count is not None else ""
    safe = summary.replace("*", "·") if summary else ""
    if safe and len(safe) > 60:
        safe = safe[:57] + "…"
    sep = "" if count is not None else "  ·  "
    summary_part = f"{sep}*{safe}*" if safe else ""
    label = f"**{num:02d}**  {title}{count_part}{summary_part}"
    return st.expander(label, expanded=expanded)


def _refresh_sql(model: QueryModel, *, from_user: bool = False) -> None:
    """Regenerate last_sql from the model.

    When from_user=True the caller is a direct user interaction (widget
    change inside the composer) — the raw-SQL lock is cleared and the SQL
    is regenerated.  When from_user=False (passive end-of-render call) we
    skip regeneration if an externally-set raw SQL is locked in, so quick
    queries and NL results are not overwritten on every rerun.
    """
    if from_user:
        st.session_state.pop("_raw_sql_lock", None)
    elif st.session_state.get("_raw_sql_lock", False):
        return
    try:
        st.session_state.last_sql = model.to_sql()
    except ValueError as exc:
        st.session_state.last_sql = f"-- {exc}"


# ── SELECT ────────────────────────────────────────────────────────────────────

def _select_section(schema, model: QueryModel, cols: List[str]) -> None:
    sel = list(model.selected_columns)
    summary = ", ".join(sel[:3]) + ("…" if len(sel) > 3 else "") if sel else "all columns"

    with _section(1, "SELECT", summary, expanded=True):
        new_sel = st.multiselect(
            "Columns",
            options=cols,
            default=sel,
            key="select_multiselect",
            label_visibility="collapsed",
            placeholder="＋ Add column",
        )
        if new_sel != sel:
            model.selected_columns = new_sel
            _refresh_sql(model, from_user=True)
            st.rerun()


# ── JOIN ──────────────────────────────────────────────────────────────────────

def _join_section(
    tables: Dict[str, Dict[str, str]],
    model: QueryModel,
    relationships: list,
) -> None:
    """Visual JOIN composer for multi-table queries."""
    join_rows: List[dict] = st.session_state.get("join_rows", [])
    n_joins = len(join_rows)
    summary = f"{n_joins} join{'s' if n_joins != 1 else ''}" if n_joins else "no joins"

    with _section(2, "JOIN", summary, count=n_joins or None, expanded=True):
        # Show detected relationships as quick-add buttons
        if relationships:
            st.markdown(
                '<div class="cs-subhead">Detected relationships</div>',
                unsafe_allow_html=True,
            )
            for i, rel in enumerate(relationships):
                left = f"{rel['left_table']}.{rel['left_col']}"
                right = f"{rel['right_table']}.{rel['right_col']}"
                label = f"{left} = {right}"

                # Check if this relationship is already added
                already_added = any(
                    jr.get("left_table") == rel["left_table"]
                    and jr.get("left_col") == rel["left_col"]
                    and jr.get("right_table") == rel["right_table"]
                    and jr.get("right_col") == rel["right_col"]
                    for jr in join_rows
                )

                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(
                        f'<div style="font-size:12px;color:var(--ink-2);padding:4px 0;">{label}</div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    if already_added:
                        st.markdown(
                            '<div style="font-size:11px;color:var(--ink-4);padding:4px 0;">✓ Added</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        if st.button("Add", key=f"rel_add_{i}", width='stretch'):
                            join_rows.append({
                                "left_table": rel["left_table"],
                                "left_col": rel["left_col"],
                                "right_table": rel["right_table"],
                                "right_col": rel["right_col"],
                                "join_type": "INNER",
                            })
                            st.session_state.join_rows = join_rows
                            st.session_state.pop("_raw_sql_lock", None)
                            st.rerun()

        # Manual JOIN configuration
        st.markdown('<div class="cs-subhead">Manual joins</div>', unsafe_allow_html=True)
        to_remove = None
        table_names = list(tables.keys())

        for i, row in enumerate(join_rows):
            c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 1, 2, 2, 0.5], gap="small")

            with c1:
                left_table = st.selectbox(
                    "left_table",
                    table_names,
                    index=table_names.index(row["left_table"]) if row.get("left_table") in table_names else 0,
                    key=f"join_lt_{i}",
                    label_visibility="collapsed",
                )
                row["left_table"] = left_table

            with c2:
                left_cols = list(tables[left_table].keys())
                left_col = st.selectbox(
                    "left_col",
                    left_cols,
                    index=left_cols.index(row["left_col"]) if row.get("left_col") in left_cols else 0,
                    key=f"join_lc_{i}",
                    label_visibility="collapsed",
                )
                row["left_col"] = left_col

            with c3:
                st.markdown(
                    '<div style="text-align:center;padding:8px 0;font-size:14px;color:var(--ink-3);">=</div>',
                    unsafe_allow_html=True,
                )

            with c4:
                right_table = st.selectbox(
                    "right_table",
                    table_names,
                    index=table_names.index(row["right_table"]) if row.get("right_table") in table_names else 0,
                    key=f"join_rt_{i}",
                    label_visibility="collapsed",
                )
                row["right_table"] = right_table

            with c5:
                right_cols = list(tables[right_table].keys())
                right_col = st.selectbox(
                    "right_col",
                    right_cols,
                    index=right_cols.index(row["right_col"]) if row.get("right_col") in right_cols else 0,
                    key=f"join_rc_{i}",
                    label_visibility="collapsed",
                )
                row["right_col"] = right_col

            with c6:
                if st.button("×", key=f"join_rm_{i}"):
                    to_remove = i

        if to_remove is not None:
            join_rows.pop(to_remove)
            st.session_state.join_rows = join_rows
            st.session_state.pop("_raw_sql_lock", None)
            st.rerun()

        # Add manual JOIN button
        if st.button("＋ Add join", key="join_add"):
            # Default to first two tables if available
            default_left = table_names[0] if table_names else ""
            default_right = table_names[1] if len(table_names) > 1 else (table_names[0] if table_names else "")
            default_left_col = list(tables[default_left].keys())[0] if default_left and default_left in tables else ""
            default_right_col = list(tables[default_right].keys())[0] if default_right and default_right in tables else ""

            join_rows.append({
                "left_table": default_left,
                "left_col": default_left_col,
                "right_table": default_right,
                "right_col": default_right_col,
                "join_type": "INNER",
            })
            st.session_state.join_rows = join_rows
            st.session_state.pop("_raw_sql_lock", None)
            st.rerun()

        # Sync to model
        model.joins = []
        for row in join_rows:
            try:
                model.joins.append(Join(
                    left_table=row["left_table"],
                    left_col=row["left_col"],
                    right_table=row["right_table"],
                    right_col=row["right_col"],
                    join_type=row.get("join_type", "INNER"),
                ))
            except Exception:
                pass


# ── WHERE ─────────────────────────────────────────────────────────────────────

def _where_section(schema, model: QueryModel, cols: List[str]) -> None:
    rows: List[dict] = st.session_state.get("where_rows", [])
    summary = _filter_summary(rows) if rows else "no filters"

    with _section(3, "WHERE", summary, count=len(rows) or None, expanded=True):
        _filter_rows_ui(schema, cols, "where_rows", model.filters)


def _filter_summary(rows: List[dict]) -> str:
    parts = []
    for r in rows[:2]:
        conj = ("AND " if parts else "")
        col = r.get("column", "")
        op = r.get("operator", "=")
        val = r.get("value", "")
        parts.append(f"{conj}{col} {op} {val}")
    if len(rows) > 2:
        parts.append(f"+{len(rows)-2} more")
    return " ".join(parts)


def _filter_rows_ui(schema, cols, row_key: str, target_list: list) -> None:
    rows: List[dict] = st.session_state.get(row_key, [])
    to_remove = None

    for i, row in enumerate(rows):
        c_conj, c_col, c_op, c_val, c_rm = st.columns([1, 3, 2, 4, 0.6], gap="small")

        with c_conj:
            if i == 0:
                st.markdown(
                    '<span class="conj-label">WHERE</span>',
                    unsafe_allow_html=True,
                )
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

        with c_col:
            col = st.selectbox(
                "column",
                cols,
                index=cols.index(row["column"]) if row.get("column") in cols else 0,
                key=f"{row_key}_col_{i}",
                label_visibility="collapsed",
            )
            row["column"] = col

        with c_op:
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

        with c_val:
            if op in ("IS NULL", "IS NOT NULL"):
                row["value"] = None
                st.markdown(
                    f'<div class="null-pill">{op.lower()}</div>',
                    unsafe_allow_html=True,
                )
            elif op == "BETWEEN":
                lo, hi = st.columns(2)
                lo_val = lo.text_input("lo", value=str(row.get("value_lo", "")),
                                       key=f"{row_key}_lo_{i}",
                                       label_visibility="collapsed",
                                       placeholder="low")
                hi_val = hi.text_input("hi", value=str(row.get("value_hi", "")),
                                       key=f"{row_key}_hi_{i}",
                                       label_visibility="collapsed",
                                       placeholder="high")
                row["value_lo"] = lo_val
                row["value_hi"] = hi_val
                row["value"] = (lo_val, hi_val)
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

        with c_rm:
            if st.button("×", key=f"{row_key}_rm_{i}"):
                to_remove = i

    if to_remove is not None:
        rows.pop(to_remove)
        st.session_state[row_key] = rows
        st.session_state.pop("_raw_sql_lock", None)
        st.rerun()

    if st.button("＋ Add condition", key=f"{row_key}_add"):
        rows.append({"column": cols[0], "operator": "=", "value": "", "logical": "AND"})
        st.session_state[row_key] = rows
        st.session_state.pop("_raw_sql_lock", None)
        st.rerun()

    # Sync to model
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
    n_total = n_group + n_agg or None
    summary = (
        f"{n_group} group{'s' if n_group != 1 else ''}, "
        f"{n_agg} agg{'s' if n_agg != 1 else ''}"
        if (n_group or n_agg) else "no grouping"
    )

    with _section(4, "GROUP BY · Aggregate", summary, count=n_total, expanded=False):
        st.markdown('<div class="cs-subhead">Group by</div>', unsafe_allow_html=True)
        new_grp = st.multiselect(
            "GROUP BY",
            options=cols,
            default=[c for c in model.group_by if c in cols],
            key="group_by_multiselect",
            label_visibility="collapsed",
            placeholder="＋ Add group column",
        )
        if new_grp != list(model.group_by):
            model.group_by = new_grp
            _refresh_sql(model, from_user=True)
            st.rerun()

        st.markdown('<div class="cs-subhead">Aggregations</div>', unsafe_allow_html=True)
        to_remove = None
        for i, row in enumerate(agg_rows):
            c1, c2, c3, c4, c5 = st.columns([2, 4, 0.5, 4, 0.7], gap="small")
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
                st.markdown('<div class="agg-arrow">→</div>', unsafe_allow_html=True)
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
    summary = _filter_summary(having_rows) if having_rows else ("add GROUP BY first" if not has_group else "no filters")

    with _section(5, "HAVING", summary, count=len(having_rows) or None, expanded=False):
        if not has_group:
            st.markdown(
                '<div class="having-empty">Add a GROUP BY column above to enable HAVING.</div>',
                unsafe_allow_html=True,
            )
        else:
            _filter_rows_ui(schema, cols, "having_rows", model.having)


# ── ORDER BY / LIMIT ──────────────────────────────────────────────────────────

def _order_limit_section(schema, model: QueryModel, cols: List[str]) -> None:
    order_rows: List[dict] = st.session_state.get("order_rows", [])
    limit_val = model.limit if model.limit is not None else 1000
    order_summary = ", ".join(
        f"{r['col']} {r.get('dir','DESC')}" for r in order_rows[:2]
    )
    summary = f"{order_summary} · LIMIT {limit_val}" if order_rows else f"LIMIT {limit_val}"

    with _section(6, "ORDER BY · LIMIT", summary, count=len(order_rows) or None, expanded=True):
        to_remove = None
        for i, row in enumerate(order_rows):
            c1, c2, c3 = st.columns([5, 3, 0.6], gap="small")
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

        st.markdown('<hr class="dashed-sep" />', unsafe_allow_html=True)
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin:6px 0 2px;">'
            '<span style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;'
            'color:var(--ink-3);font-weight:600;">LIMIT</span>'
            '<span style="font-size:11px;color:var(--ink-3);">— 0 means no cap</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        limit = st.number_input(
            "LIMIT", min_value=0, max_value=1_000_000, step=100,
            value=limit_val, key="limit_input",
            label_visibility="collapsed",
        )
        model.limit = int(limit) if limit > 0 else None

        model.order_by = []
        for row in order_rows:
            try:
                model.order_by.append((row["col"], row.get("dir", "DESC")))
            except Exception:
                pass
