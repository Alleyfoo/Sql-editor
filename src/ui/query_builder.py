"""Main Tkinter window for the Visual Query Builder."""

from __future__ import annotations

import sqlite3
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import pandas as pd

from .. import history
from ..config import load_config
from ..executor import ExecutionError, execute
from ..ingestion import TABLE_NAME, load_csv
from ..query_model import QueryModel
from .aggregation import AggregationPanel
from .filter_rows import FilterRows
from .order_by import OrderByRows
from .results_table import ResultsTable
from .sql_preview import SqlPreview


class QueryBuilderApp(tk.Tk):
    """Three-panel layout: columns | filters+etc | SQL preview.

    Bottom strip: Run / row count / Export / results table.
    """

    def __init__(self) -> None:
        super().__init__()
        self._config = load_config()
        app_cfg = self._config.get("app", {}) or {}
        self.title(app_cfg.get("window_title", "Visual Query Builder"))
        self.geometry("1100x720")

        self._conn: Optional[sqlite3.Connection] = None
        self._schema: Dict[str, str] = {}
        self._model = QueryModel(table=TABLE_NAME)
        self._default_limit = int(app_cfg.get("default_limit", 1000))

        self._build_menu()
        self._build_layout()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open CSV…", command=self._open_csv)
        file_menu.add_command(label="Export Results…", command=self._export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    def _build_layout(self) -> None:
        # Top paned window: left (columns) | center (sections) | right (SQL)
        top = ttk.PanedWindow(self, orient="horizontal")
        top.pack(fill="both", expand=True, padx=6, pady=6)

        # --- Left panel: columns ---
        left = ttk.LabelFrame(top, text="Columns")
        self._columns_tree = ttk.Treeview(
            left, columns=("type",), show="tree headings", selectmode="none"
        )
        self._columns_tree.heading("#0", text="Column")
        self._columns_tree.heading("type", text="Type")
        self._columns_tree.column("#0", width=150)
        self._columns_tree.column("type", width=70, anchor="w")
        self._columns_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self._columns_tree.bind("<Button-1>", self._on_column_click)
        top.add(left, weight=1)

        # --- Center panel: sections ---
        center = ttk.Frame(top)
        top.add(center, weight=2)

        self._filters_frame = ttk.LabelFrame(center, text="Filters (WHERE)")
        self._filters_frame.pack(fill="x", padx=4, pady=4)
        self._filter_rows = FilterRows(
            self._filters_frame, schema={}, on_change=self._refresh_sql
        )
        self._filter_rows.pack(fill="x", padx=4, pady=4)

        self._agg_frame = ttk.LabelFrame(center, text="Grouping & Aggregations")
        self._agg_frame.pack(fill="x", padx=4, pady=4)
        self._aggregation = AggregationPanel(
            self._agg_frame, schema={}, on_change=self._refresh_sql
        )
        self._aggregation.pack(fill="x", padx=4, pady=4)

        self._having_frame = ttk.LabelFrame(center, text="HAVING (on grouped results)")
        self._having_frame.pack(fill="x", padx=4, pady=4)
        self._having_rows = FilterRows(
            self._having_frame, schema={}, on_change=self._refresh_sql
        )
        self._having_rows.pack(fill="x", padx=4, pady=4)
        self._having_hint = ttk.Label(
            self._having_frame,
            text="(Add a GROUP BY column above to enable HAVING.)",
            foreground="#888888",
        )
        self._having_hint.pack(padx=4, anchor="w")

        self._order_frame = ttk.LabelFrame(center, text="Order & Limit")
        self._order_frame.pack(fill="x", padx=4, pady=4)
        self._order_rows = OrderByRows(
            self._order_frame, columns=[], on_change=self._refresh_sql
        )
        self._order_rows.pack(fill="x", padx=4, pady=(4, 0))

        limit_row = ttk.Frame(self._order_frame)
        limit_row.pack(fill="x", padx=4, pady=(6, 4))
        ttk.Label(limit_row, text="LIMIT:").pack(side="left")
        self._limit_var = tk.StringVar(value=str(self._default_limit))
        self._limit_var.trace_add("write", lambda *_: self._refresh_sql())
        self._limit_spin = ttk.Spinbox(
            limit_row,
            from_=0,
            to=1_000_000,
            textvariable=self._limit_var,
            width=10,
        )
        self._limit_spin.pack(side="left", padx=4)
        ttk.Label(
            limit_row,
            text="(0 = no limit)",
            foreground="#888888",
        ).pack(side="left", padx=6)

        # --- Right panel: SQL preview ---
        right = ttk.LabelFrame(top, text="Generated SQL")
        self._sql_preview = SqlPreview(right)
        self._sql_preview.pack(fill="both", expand=True, padx=4, pady=4)
        top.add(right, weight=2)

        # --- Bottom strip: Run / row count / Export / Results ---
        bottom = ttk.Frame(self)
        bottom.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        control_bar = ttk.Frame(bottom)
        control_bar.pack(fill="x")
        self._run_btn = ttk.Button(control_bar, text="Run", command=self._run_query)
        self._run_btn.pack(side="left")
        self._row_count_var = tk.StringVar(value="0 rows")
        ttk.Label(control_bar, textvariable=self._row_count_var).pack(side="left", padx=10)
        self._export_btn = ttk.Button(
            control_bar, text="Export CSV…", command=self._export_results, state="disabled"
        )
        self._export_btn.pack(side="right")

        self._results = ResultsTable(
            bottom, max_rows=int((self._config.get("app") or {}).get("max_preview_rows", 1000))
        )
        self._results.pack(fill="both", expand=True, pady=(4, 0))

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def _open_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Open CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            conn, schema = load_csv(path)
        except Exception as exc:
            messagebox.showerror("Failed to load CSV", str(exc))
            return

        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = conn
        self._schema = schema
        self._model = QueryModel(table=TABLE_NAME)

        self._populate_columns(schema)
        self._filter_rows.clear()
        self._filter_rows.set_schema(schema)
        self._aggregation.clear()
        self._aggregation.set_schema(schema)
        self._having_rows.clear()
        self._having_rows.set_schema({})
        self._order_rows.clear()
        self._order_rows.set_columns(list(schema.keys()))
        self._results.clear()
        self._row_count_var.set("0 rows")
        self._export_btn.configure(state="disabled")
        self._refresh_sql()
        self.title(f"{self.title()} — {Path(path).name}")

    def _populate_columns(self, schema: Dict[str, str]) -> None:
        for item in self._columns_tree.get_children():
            self._columns_tree.delete(item)
        for col, col_type in schema.items():
            self._columns_tree.insert(
                "", "end", iid=col, text=f"☐ {col}", values=(col_type,)
            )

    def _on_column_click(self, event: tk.Event) -> None:
        item = self._columns_tree.identify_row(event.y)
        if not item:
            return
        if item in self._model.selected_columns:
            self._model.selected_columns.remove(item)
            self._columns_tree.item(item, text=f"☐ {item}")
        else:
            self._model.selected_columns.append(item)
            self._columns_tree.item(item, text=f"☑ {item}")
        self._refresh_sql()

    # ------------------------------------------------------------------
    # SQL refresh
    # ------------------------------------------------------------------

    def _refresh_sql(self) -> None:
        if self._conn is None:
            self._sql_preview.set_error("Open a CSV to begin.")
            self._run_btn.configure(state="disabled")
            return

        group_by = self._aggregation.get_group_by()
        aggregations = self._aggregation.get_aggregations()

        # HAVING is only meaningful with GROUP BY — keep its schema in sync
        # with the current group columns + aggregation aliases. This does
        # not fire on_change (set_schema is silent by design).
        self._sync_having_schema(group_by, aggregations)
        # ORDER BY can reference any of the source columns plus any
        # aggregation aliases.
        self._sync_order_columns(aggregations)

        self._model.filters = self._filter_rows.get_filters()
        self._model.group_by = group_by
        self._model.aggregations = aggregations
        self._model.having = self._having_rows.get_filters() if group_by else []
        self._model.order_by = self._order_rows.get_order_by()
        self._model.limit = self._parse_limit()

        try:
            sql = self._model.to_sql()
        except ValueError as exc:
            self._sql_preview.set_error(str(exc))
            self._run_btn.configure(state="disabled")
            return

        self._sql_preview.set_sql(sql)
        self._run_btn.configure(state="normal")

    def _sync_having_schema(self, group_by, aggregations) -> None:
        """HAVING can reference group columns and aggregation aliases."""
        having_schema: Dict[str, str] = {
            col: self._schema.get(col, "text") for col in group_by
        }
        for agg in aggregations:
            # Aggregation outputs are treated as numeric for operator
            # purposes — min/max/avg/sum/count all support numeric
            # comparisons, and MIN/MAX on text still admits ordering.
            having_schema[agg.display_name] = "numeric"
        self._having_rows.set_schema(having_schema)

        # Show or hide the "needs GROUP BY" hint. Rows added without
        # GROUP BY are silently dropped by ``_refresh_sql``.
        if group_by:
            self._having_hint.pack_forget()
        else:
            self._having_hint.pack(padx=4, anchor="w")

    def _sync_order_columns(self, aggregations) -> None:
        cols = list(self._schema.keys())
        for agg in aggregations:
            name = agg.display_name
            if name not in cols:
                cols.append(name)
        self._order_rows.set_columns(cols)

    def _parse_limit(self) -> Optional[int]:
        raw = (self._limit_var.get() or "").strip()
        if not raw:
            return None
        try:
            n = int(raw)
        except ValueError:
            return None
        return None if n <= 0 else n

    # ------------------------------------------------------------------
    # Run / export
    # ------------------------------------------------------------------

    def _run_query(self) -> None:
        if self._conn is None:
            return
        try:
            sql = self._model.to_sql()
        except ValueError as exc:
            messagebox.showerror("Invalid query", str(exc))
            return
        try:
            df = execute(self._conn, sql)
        except ExecutionError as exc:
            messagebox.showerror("Execution error", str(exc))
            return
        self._results.set_dataframe(df)
        self._row_count_var.set(f"{len(df)} rows")
        self._export_btn.configure(state="normal" if len(df) else "disabled")
        try:
            history.log_query(sql, len(df))
        except OSError as exc:
            # History is best-effort; surface a warning but don't block.
            print(f"warning: failed to log query history: {exc}")

    def _export_results(self) -> None:
        df = self._results.current_df
        if df is None or df.empty:
            messagebox.showinfo("Export CSV", "No results to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        try:
            df.to_csv(path, index=False)
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        messagebox.showinfo("Export CSV", f"Wrote {len(df)} rows to {path}.")

    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self.destroy()


__all__ = ["QueryBuilderApp"]
