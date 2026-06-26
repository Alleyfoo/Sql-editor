from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import streamlit as st


def render() -> None:
    # Use Streamlit container with key for CSS targeting
    with st.container(key="results_panel"):
        df: Optional[pd.DataFrame] = st.session_state.get("results_df")
        elapsed = st.session_state.get("last_exec_ms")

        if df is None:
            tables: dict = st.session_state.get("tables", {})
            dataset_df: Optional[pd.DataFrame] = st.session_state.get("dataset_df")
            conn = st.session_state.get("conn")

            if tables and conn:
                # Multi-table dataset — show one tab per table
                _render_multitable_preview(tables, conn)
            elif dataset_df is not None:
                _render_data_preview(dataset_df)
            else:
                st.markdown(
                    '<div style="padding:48px;text-align:center;color:#8E867B;font-size:13px;">'
                    "Load a CSV to get started."
                    "</div>",
                    unsafe_allow_html=True,
                )
            return

        # Compact stats bar
        n_rows, n_cols = df.shape
        elapsed_str = f"{elapsed:.0f} ms" if elapsed is not None else "—"
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        size_kb = round(len(csv_bytes) / 1024, 1)

        stats_col, export_col = st.columns([1, 0.18])
        with stats_col:
            st.markdown(
                f"""
                <div class="results-stats-bar">
                  <span class="rs"><strong>{n_rows:,}</strong> rows</span>
                  <span class="rs-sep">·</span>
                  <span class="rs"><strong>{n_cols}</strong> cols</span>
                  <span class="rs-sep">·</span>
                  <span class="rs"><strong>{elapsed_str}</strong> exec</span>
                  <span class="rs-sep">·</span>
                  <span class="rs"><strong>{size_kb} KB</strong> scanned</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with export_col:
            st.download_button(
                "Export CSV ↗",
                data=csv_bytes,
                file_name="query_result.csv",
                mime="text/csv",
                width='stretch',
            )

        table_tab, chart_tab, summary_tab, json_tab = st.tabs(
            ["Table", "Chart", "Summary", "JSON"]
        )

        with table_tab:
            _render_table(df)

        with chart_tab:
            _render_chart(df)

        with summary_tab:
            _render_summary(df)

        with json_tab:
            st.caption("Showing first 200 rows.")
            st.json(df.head(200).to_dict(orient="records"))

    # Footer
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        f'<div class="results-footer">'
        f'Showing all <strong>{n_rows:,}</strong> rows.'
        f'<span style="margin-left:8px;">Run finished <span class="mono">{ts}</span></span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_multitable_preview(tables: dict, conn) -> None:
    """Show one tab per table for multi-table datasets."""
    table_names = list(tables.keys())
    tabs = st.tabs(table_names)
    for tab, table_name in zip(tabs, table_names):
        with tab:
            try:
                tdf = pd.read_sql_query(
                    f'SELECT * FROM "{table_name}" LIMIT 500', conn
                )
                schema = tables[table_name]
                n_rows_q = pd.read_sql_query(
                    f'SELECT COUNT(*) AS n FROM "{table_name}"', conn
                )
                n_total = int(n_rows_q.iloc[0, 0])
                st.markdown(
                    f'<div class="results-stats-bar">'
                    f'<span class="rs"><strong>{n_total:,}</strong> rows</span>'
                    f'<span class="rs-sep">·</span>'
                    f'<span class="rs"><strong>{len(tdf.columns)}</strong> columns</span>'
                    f'<span class="rs-sep">·</span>'
                    f'<span class="rs" style="color:#8E867B;">click a column to ask about it →</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Column chips
                cols_list = list(tdf.columns)
                per_row = 5
                for i in range(0, len(cols_list), per_row):
                    batch = cols_list[i : i + per_row]
                    # Always use per_row columns to maintain uniform width
                    widgets = st.columns(per_row)
                    for idx, col_name in enumerate(batch):
                        col_type = schema.get(col_name, "text")
                        if col_type == "numeric":
                            prefill = f"What is the min, max and average of {col_name}?"
                        elif col_type == "date":
                            prefill = f"What is the date range of {col_name}?"
                        else:
                            prefill = f"What are the top values of {col_name}?"
                        with widgets[idx]:
                            if st.button(
                                col_name,
                                key=f"col_chip_{table_name}_{col_name}",
                                help=f"Ask: {prefill}",
                            ):
                                st.session_state["nl_prefill"] = prefill
                                st.rerun()
                # Table
                st.markdown('<div style="margin-top:6px;"></div>', unsafe_allow_html=True)
                display_tdf = _decorate_for_display(tdf)
                st.dataframe(
                    display_tdf,
                    column_config=_build_column_config(display_tdf),
                    use_container_width=True,
                    hide_index=True,
                )
                if n_total > 500:
                    st.caption(f"Showing first 500 of {n_total:,} rows.")
            except Exception as exc:
                st.warning(f"Could not preview {table_name}: {exc}")


def _render_data_preview(df: pd.DataFrame) -> None:
    """Show the raw loaded dataset with clickable column chips above it."""
    n_rows, n_cols = df.shape
    schema: dict = st.session_state.get("schema", {})

    # Stats bar
    st.markdown(
        f'<div class="results-stats-bar">'
        f'<span class="rs"><strong>{n_rows:,}</strong> rows</span>'
        f'<span class="rs-sep">·</span>'
        f'<span class="rs"><strong>{n_cols}</strong> columns</span>'
        f'<span class="rs-sep">·</span>'
        f'<span class="rs" style="color:#8E867B;">click a column to ask about it →</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Column chips — click to prefill the ask bar with a type-aware question
    cols_list = list(df.columns)
    per_row = 5
    for i in range(0, len(cols_list), per_row):
        batch = cols_list[i : i + per_row]
        # Always use per_row columns to maintain uniform width
        widgets = st.columns(per_row)
        for idx, col_name in enumerate(batch):
            col_type = schema.get(col_name, "text")
            if col_type == "numeric":
                prefill = f"What is the min, max and average of {col_name}?"
            elif col_type == "date":
                prefill = f"What is the date range of {col_name}?"
            else:
                prefill = f"What are the top values of {col_name}?"
            with widgets[idx]:
                if st.button(
                    col_name,
                    key=f"col_chip_{col_name}",
                    help=f"Ask: {prefill}",
                ):
                    st.session_state["nl_prefill"] = prefill
                    st.rerun()

    # Preview table
    st.markdown('<div style="margin-top:6px;"></div>', unsafe_allow_html=True)
    preview = df.head(500)
    display_df = _decorate_for_display(preview)
    col_cfg = _build_column_config(display_df)
    st.dataframe(
        display_df,
        column_config=col_cfg,
        use_container_width=True,
        hide_index=True,
    )
    if n_rows > 500:
        st.caption(f"Showing first 500 of {n_rows:,} rows. Run a query to filter and aggregate.")


def _decorate_for_display(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            name = col.lower()
            if name.endswith("_date") or name.endswith("_at"):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
    return df


def _build_column_config(df: pd.DataFrame) -> dict:
    cfg = {}
    magnitude_suffixes = ("revenue", "total", "count", "amount", "sum", "value", "qty")
    for col in df.columns:
        dtype = df[col].dtype
        col_lower = col.lower()
        if pd.api.types.is_numeric_dtype(dtype):
            non_null = df[col].dropna()
            if (
                len(non_null) > 1
                and non_null.min() >= 0
                and any(col_lower.endswith(s) for s in magnitude_suffixes)
            ):
                cfg[col] = st.column_config.ProgressColumn(
                    col,
                    min_value=0,
                    max_value=float(non_null.max()),
                    format="%.2f",
                )
            else:
                cfg[col] = st.column_config.NumberColumn(col, format="%.2f")
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            cfg[col] = st.column_config.DatetimeColumn(col, format="YYYY-MM-DD")
    return cfg


def _render_table(df: pd.DataFrame) -> None:
    display_df = _decorate_for_display(df)
    col_cfg = _build_column_config(display_df)
    st.dataframe(
        display_df,
        column_config=col_cfg,
        width='stretch',
        hide_index=True,
    )


def _render_chart_from_spec(spec, df: pd.DataFrame) -> None:
    """Render a single ChartSpec (from AnalysisCoordinator) as an Altair chart."""
    try:
        import altair as alt
    except ImportError:
        return

    if spec.x_field not in df.columns or spec.y_field not in df.columns:
        st.caption(f"Chart fields ({spec.x_field}, {spec.y_field}) not found in result.")
        return

    mark = alt.Chart(df.head(500))
    if spec.chart_type == "line":
        chart = mark.mark_line(color="#C2410C", strokeWidth=2, point=True)
    elif spec.chart_type == "bar":
        chart = mark.mark_bar(
            color="#C2410C", cornerRadiusTopLeft=3, cornerRadiusTopRight=3
        )
    else:
        chart = mark.mark_point(color="#C2410C", opacity=0.7)

    tooltip = [spec.x_field, spec.y_field]
    if spec.series_field and spec.series_field in df.columns:
        tooltip.append(spec.series_field)

    chart = (
        chart.encode(
            x=alt.X(spec.x_field),
            y=alt.Y(spec.y_field),
            tooltip=tooltip,
            **({"color": spec.series_field} if spec.series_field and spec.series_field in df.columns else {}),
        )
        .properties(title=spec.title, height=320)
    )
    st.altair_chart(chart)


def _render_chart(df: pd.DataFrame) -> None:
    # If AnalysisCoordinator produced chart specs, use those
    chart_specs = st.session_state.get("last_chart_specs")
    if chart_specs:
        for spec in chart_specs:
            _render_chart_from_spec(spec, df)
        return

    try:
        import altair as alt
    except ImportError:
        st.info("Install altair to enable charting: `pip install altair`")
        return

    text_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    # Auto-pick chart type
    if len(text_cols) == 1 and len(num_cols) == 1:
        chart = (
            alt.Chart(df.head(100))
            .mark_bar(color="#C2410C", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X(text_cols[0], sort="-y"),
                y=alt.Y(num_cols[0]),
                tooltip=[text_cols[0], num_cols[0]],
            )
            .properties(height=320)
        )
        st.altair_chart(chart)
    elif date_cols and num_cols:
        chart = (
            alt.Chart(df.head(500))
            .mark_line(color="#C2410C", strokeWidth=2)
            .encode(
                x=alt.X(date_cols[0]),
                y=alt.Y(num_cols[0]),
                tooltip=[date_cols[0], num_cols[0]],
            )
            .properties(height=320)
        )
        st.altair_chart(chart)
    elif len(num_cols) >= 2:
        x_col = st.selectbox("X axis", num_cols, key="chart_x")
        y_col = st.selectbox("Y axis", [c for c in num_cols if c != x_col], key="chart_y")
        chart = (
            alt.Chart(df.head(500))
            .mark_point(color="#C2410C", opacity=0.7)
            .encode(
                x=alt.X(x_col),
                y=alt.Y(y_col),
                tooltip=list(df.columns[:4]),
            )
            .properties(height=320)
        )
        st.altair_chart(chart)
    else:
        sel_x = st.selectbox("X axis", list(df.columns), key="chart_x_free")
        sel_y = st.selectbox("Y axis", list(df.columns), key="chart_y_free")
        if sel_x and sel_y and sel_x != sel_y:
            chart = (
                alt.Chart(df.head(100))
                .mark_bar(color="#C2410C")
                .encode(x=sel_x, y=sel_y, tooltip=[sel_x, sel_y])
                .properties(height=320)
            )
            st.altair_chart(chart)


def _render_summary(df: pd.DataFrame) -> None:
    for col in df.columns:
        dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "text"
        count = int(df[col].count())
        nulls = int(df[col].isna().sum())
        total = len(df)

        if dtype == "numeric":
            mn = float(df[col].min()) if count else None
            mx = float(df[col].max()) if count else None
            mean = float(df[col].mean()) if count else None
            std = float(df[col].std()) if count > 1 else None
            detail = (
                f"min {mn:,.2f} &nbsp;·&nbsp; mean {mean:,.2f} "
                f"&nbsp;·&nbsp; max {mx:,.2f}"
                + (f" &nbsp;·&nbsp; std {std:,.2f}" if std else "")
            ) if mn is not None else "no data"
        else:
            unique = int(df[col].nunique())
            detail = f"{unique} unique values"

        pct = round(count / total * 100, 1) if total else 0
        st.markdown(
            f"""
            <div class="insight-card" style="margin-bottom:6px;">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="font-weight:500;font-size:13px;">{col}</span>
                <span class="type-chip {dtype}">{dtype}</span>
              </div>
              <div style="font-size:12px;color:#57514A;">{detail}</div>
              <div style="font-size:11px;color:#8E867B;margin-top:2px;">
                {count:,} / {total:,} non-null &nbsp;({pct}% complete)
                {f'&nbsp;·&nbsp; {nulls} nulls' if nulls else ''}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
