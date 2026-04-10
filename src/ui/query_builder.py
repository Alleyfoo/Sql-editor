"""Main Tkinter window for the Visual Query Builder."""

from __future__ import annotations

import sqlite3
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import pandas as pd

from .. import history
from ..config import load_config
from ..executor import ExecutionError, execute
from ..ingestion import TABLE_NAME, load_csv
from ..llm.natural_language import (
    LLMError,
    RouteToPythonError,
    load_llm_config,
    nl_to_query_model,
)
from ..llm.result_analysis import (
    AnalysisError,
    ResultAnalysis,
    analyze_result_with_llm,
    fallback_result_analysis,
)
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
        self._llm_config = load_llm_config(self._config)
        # Bounded conversation history: list of (question, reply) tuples.
        self._nl_history: list = []

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
        # --- Top strip: natural-language input (Phase 3) ---
        nl_bar = ttk.Frame(self)
        nl_bar.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(nl_bar, text="Ask in natural language:").pack(side="left")
        self._nl_var = tk.StringVar()
        self._nl_entry = ttk.Entry(nl_bar, textvariable=self._nl_var)
        self._nl_entry.pack(side="left", fill="x", expand=True, padx=6)
        self._nl_entry.bind("<Return>", lambda _e: self._ask_nl())
        self._nl_entry.configure(state="disabled")
        self._ask_btn = ttk.Button(
            nl_bar, text="Ask", command=self._ask_nl, state="disabled"
        )
        self._ask_btn.pack(side="left")
        self._ask_analyze_btn = ttk.Button(
            nl_bar, text="Ask + Analyze", command=self._ask_nl_analyze, state="disabled"
        )
        self._ask_analyze_btn.pack(side="left", padx=(4, 0))
        self._nl_status_var = tk.StringVar(value="")
        ttk.Label(
            nl_bar,
            textvariable=self._nl_status_var,
            foreground="#888888",
        ).pack(side="left", padx=8)

        # --- Chat history (reply + SQL) ---
        chat_frame = ttk.LabelFrame(self, text="Assistant")
        chat_frame.pack(fill="x", padx=6, pady=(4, 0))
        self._chat_text = tk.Text(
            chat_frame,
            height=5,
            state="disabled",
            wrap="word",
            font=("TkDefaultFont", 9),
            background="#f7f7f7",
            relief="flat",
        )
        chat_scroll = ttk.Scrollbar(chat_frame, command=self._chat_text.yview)
        self._chat_text.configure(yscrollcommand=chat_scroll.set)
        chat_scroll.pack(side="right", fill="y")
        self._chat_text.pack(fill="x", padx=4, pady=4)
        # Text tags for styling
        self._chat_text.tag_configure(
            "user", foreground="#0055aa", font=("TkDefaultFont", 9, "bold")
        )
        self._chat_text.tag_configure("reply", foreground="#333333")
        self._chat_text.tag_configure(
            "sql", foreground="#006600", font=("TkFixedFont", 9)
        )
        self._chat_text.tag_configure("analysis", foreground="#4c3a00")
        self._chat_text.tag_configure("sep", foreground="#cccccc")

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
        ttk.Label(control_bar, textvariable=self._row_count_var).pack(
            side="left", padx=10
        )
        self._export_btn = ttk.Button(
            control_bar,
            text="Export CSV…",
            command=self._export_results,
            state="disabled",
        )
        self._export_btn.pack(side="right")

        self._results = ResultsTable(
            bottom,
            max_rows=int((self._config.get("app") or {}).get("max_preview_rows", 1000)),
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
        self._nl_history = []  # new dataset → fresh context

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
        self._nl_entry.configure(state="normal")
        self._ask_btn.configure(state="normal")
        self._ask_analyze_btn.configure(state="normal")
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
    # Natural-language (Phase 3)
    # ------------------------------------------------------------------

    def _set_nl_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._ask_btn.configure(state=state)
        self._ask_analyze_btn.configure(state=state)
        self._nl_entry.configure(state=state)

    def _append_nl_history(self, question: str, reply: str) -> None:
        depth = self._llm_config.history_depth
        if depth > 0 and question:
            self._nl_history.append((question, reply or "Done."))
            if len(self._nl_history) > depth:
                self._nl_history = self._nl_history[-depth:]

    def _ask_nl(self) -> None:
        if self._conn is None:
            return
        text = (self._nl_var.get() or "").strip()
        if not text:
            return
        self._set_nl_controls_enabled(False)
        self._nl_status_var.set("Thinking…")
        self._last_nl_text = text
        selected = list(self._model.selected_columns)
        history_snapshot = list(self._nl_history)
        thread = threading.Thread(
            target=self._nl_worker, args=(text, selected, history_snapshot), daemon=True
        )
        thread.start()

    def _ask_nl_analyze(self) -> None:
        if self._conn is None:
            return
        text = (self._nl_var.get() or "").strip()
        if not text:
            return
        self._set_nl_controls_enabled(False)
        self._nl_status_var.set("Planning, running, and analyzing…")
        self._last_nl_text = text
        selected = list(self._model.selected_columns)
        history_snapshot = list(self._nl_history)
        thread = threading.Thread(
            target=self._nl_agent_worker,
            args=(text, selected, history_snapshot),
            daemon=True,
        )
        thread.start()

    def _nl_worker(self, text: str, selected_columns: list, history: list) -> None:
        """Runs on a background thread — must never touch Tk widgets."""
        try:
            model = nl_to_query_model(
                text,
                self._schema,
                config=self._llm_config,
                selected_columns=selected_columns or None,
                history=history or None,
            )
        except LLMError as exc:
            self.after(0, self._nl_done, exc)
            return
        except Exception as exc:  # pragma: no cover - unexpected transport bugs
            self.after(0, self._nl_done, exc)
            return
        self.after(0, self._nl_done, model)

    def _nl_agent_worker(self, text: str, selected_columns: list, history: list) -> None:
        """NL agent flow: plan -> SQL -> execute -> analysis."""
        try:
            model = nl_to_query_model(
                text,
                self._schema,
                config=self._llm_config,
                selected_columns=selected_columns or None,
                history=history or None,
            )
            sql = model.to_sql()
            if self._conn is None:
                raise RuntimeError("Database connection is not available.")
            df = execute(self._conn, sql)
            try:
                analysis = analyze_result_with_llm(
                    text,
                    sql,
                    df,
                    self._schema,
                    config=self._llm_config,
                )
            except AnalysisError as exc:
                analysis = fallback_result_analysis(text, sql, df, warning=str(exc))
            self.after(
                0,
                self._nl_agent_done,
                {"model": model, "sql": sql, "df": df, "analysis": analysis},
            )
        except Exception as exc:  # includes LLMError/RouteToPythonError/ExecutionError
            self.after(0, self._nl_agent_done, exc)

    def _nl_done(self, payload: object) -> None:
        """Runs on the Tk thread. ``payload`` is a ``QueryModel`` or an
        ``Exception``."""
        self._set_nl_controls_enabled(True)
        self._nl_status_var.set("")
        question = getattr(self, "_last_nl_text", "")
        if isinstance(payload, Exception):
            if isinstance(payload, RouteToPythonError):
                self._chat_append(question, reply=str(payload))
                messagebox.showinfo("Routed to Python analytics", str(payload))
            else:
                self._chat_append(question, error=str(payload))
                messagebox.showerror("LLM error", str(payload))
            return
        if isinstance(payload, QueryModel):
            self._apply_model(payload)
            sql = ""
            try:
                sql = payload.to_sql()
            except Exception:
                pass
            self._chat_append(question, reply=payload.reply, sql=sql)
            self._append_nl_history(question, payload.reply or "Done.")

    def _nl_agent_done(self, payload: object) -> None:
        """Runs on Tk thread with either an error or agent output bundle."""
        self._set_nl_controls_enabled(True)
        self._nl_status_var.set("")
        question = getattr(self, "_last_nl_text", "")
        if isinstance(payload, Exception):
            if isinstance(payload, RouteToPythonError):
                self._chat_append(question, reply=str(payload))
                messagebox.showinfo("Routed to Python analytics", str(payload))
            elif isinstance(payload, ExecutionError):
                self._chat_append(question, error=str(payload))
                messagebox.showerror("Execution error", str(payload))
            else:
                self._chat_append(question, error=str(payload))
                messagebox.showerror("LLM error", str(payload))
            return
        if not isinstance(payload, dict):
            self._chat_append(question, error="Unexpected agent payload.")
            messagebox.showerror("Agent error", "Unexpected agent payload.")
            return
        model = payload.get("model")
        sql = payload.get("sql", "")
        df = payload.get("df")
        analysis = payload.get("analysis")
        if not isinstance(model, QueryModel) or not isinstance(df, pd.DataFrame):
            self._chat_append(question, error="Invalid agent response shape.")
            messagebox.showerror("Agent error", "Invalid agent response shape.")
            return
        self._apply_model(model)
        self._results.set_dataframe(df)
        self._row_count_var.set(f"{len(df)} rows")
        self._export_btn.configure(state="normal" if len(df) else "disabled")
        if isinstance(sql, str) and sql:
            try:
                history.log_query(sql, len(df))
            except OSError as exc:
                print(f"warning: failed to log query history: {exc}")
        analysis_text = ""
        history_reply = model.reply or "Done."
        if isinstance(analysis, ResultAnalysis):
            history_reply = analysis.summary or history_reply
            insights = "\n".join(f"- {line}" for line in analysis.insights[:4])
            next_q = "\n".join(f"- {line}" for line in analysis.next_questions[:3])
            warns = "\n".join(f"- {line}" for line in analysis.warnings[:2])
            parts = [analysis.summary]
            if insights:
                parts.append("Insights:\n" + insights)
            if next_q:
                parts.append("Follow-ups:\n" + next_q)
            if warns:
                parts.append("Warnings:\n" + warns)
            analysis_text = "\n\n".join(parts).strip()
        self._chat_append(question, reply=model.reply, sql=sql if isinstance(sql, str) else "", analysis=analysis_text)
        self._append_nl_history(question, history_reply)

    def _chat_append(
        self,
        question: str,
        *,
        reply: str = "",
        sql: str = "",
        analysis: str = "",
        error: str = "",
    ) -> None:
        """Append one exchange to the chat history widget."""
        self._chat_text.configure(state="normal")
        # Separator between exchanges
        if self._chat_text.get("1.0", "end").strip():
            self._chat_text.insert("end", "─" * 60 + "\n", "sep")
        if question:
            self._chat_text.insert("end", f"You: {question}\n", "user")
        if error:
            self._chat_text.insert("end", f"Error: {error}\n", "reply")
        else:
            if reply:
                self._chat_text.insert("end", f"{reply}\n", "reply")
            if sql:
                self._chat_text.insert("end", f"{sql}\n", "sql")
            if analysis:
                self._chat_text.insert("end", f"{analysis}\n", "analysis")
        self._chat_text.configure(state="disabled")
        self._chat_text.see("end")

    def _apply_model(self, model: QueryModel) -> None:
        """Push an LLM-produced ``QueryModel`` into the UI widgets.

        Does NOT execute the query — the user must click Run. This is a
        deliberate safety boundary: the user always sees the generated
        SQL before any rows move.
        """
        # Reconcile column selection with the columns tree.
        wanted = set(model.selected_columns)
        for item in self._columns_tree.get_children():
            label = "☑ " if item in wanted else "☐ "
            self._columns_tree.item(item, text=f"{label}{item}")
        self._model.selected_columns = [
            c for c in model.selected_columns if c in self._schema
        ]

        self._filter_rows.set_filters(model.filters)
        self._aggregation.set_state(model.group_by, model.aggregations)
        # HAVING schema will be re-synced by _refresh_sql below; populate
        # the rows first so they're ready for the new schema.
        self._having_rows.set_filters(model.having)
        self._order_rows.set_order_by(model.order_by)
        self._limit_var.set("" if model.limit is None else str(model.limit))
        self._refresh_sql()

    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self.destroy()


__all__ = ["QueryBuilderApp"]
