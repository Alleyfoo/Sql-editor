"""GROUP BY + aggregation rows widget.

Layout:

    Group by:  [col1, col2, ...]  (multi-select, driven by checkboxes)

    Aggregations
    ------------
    [function] [column] [alias] [X]
    [function] [column] [alias] [X]
    [ + Add aggregation ]

``get_group_by()`` returns the list of checked GROUP BY column names.
``get_aggregations()`` returns a list of ``query_model.Aggregation`` rows
(rows with no column or no function are dropped).

On any change, the parent's ``on_change`` callback fires so the SQL
preview refreshes live.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List

from ..query_model import AGGREGATION_FUNCTIONS, Aggregation


class _AggregationRow:
    def __init__(
        self, parent: "AggregationPanel", container: ttk.Frame
    ) -> None:
        self._parent = parent
        self._frame = ttk.Frame(container)

        self.function_var = tk.StringVar(value="SUM")
        self.column_var = tk.StringVar()
        self.alias_var = tk.StringVar()

        self._function_box = ttk.Combobox(
            self._frame,
            textvariable=self.function_var,
            values=list(AGGREGATION_FUNCTIONS),
            width=14,
            state="readonly",
        )
        self._column_box = ttk.Combobox(
            self._frame,
            textvariable=self.column_var,
            values=self._column_choices(),
            width=16,
            state="readonly",
        )
        self._alias_entry = ttk.Entry(
            self._frame, textvariable=self.alias_var, width=14
        )
        self._remove_btn = ttk.Button(
            self._frame, text="X", width=2, command=self._remove
        )

        ttk.Label(self._frame, text="fn").grid(row=0, column=0, padx=(0, 2))
        self._function_box.grid(row=0, column=1, padx=2)
        ttk.Label(self._frame, text="col").grid(row=0, column=2, padx=(6, 2))
        self._column_box.grid(row=0, column=3, padx=2)
        ttk.Label(self._frame, text="as").grid(row=0, column=4, padx=(6, 2))
        self._alias_entry.grid(row=0, column=5, padx=2)
        self._remove_btn.grid(row=0, column=6, padx=(6, 2))

        self.function_var.trace_add("write", lambda *_: parent._fire_change())
        self.column_var.trace_add("write", lambda *_: parent._fire_change())
        self.alias_var.trace_add("write", lambda *_: parent._fire_change())

        self._frame.grid(sticky="ew", pady=1)

    # ------------------------------------------------------------------

    def _column_choices(self) -> List[str]:
        # COUNT accepts "*" as a shortcut; include it alongside real columns.
        return ["*"] + list(self._parent.schema.keys())

    def refresh_columns(self) -> None:
        self._column_box.configure(values=self._column_choices())

    def destroy(self) -> None:
        self._frame.destroy()

    def _remove(self) -> None:
        self._parent._remove_row(self)

    def to_aggregation(self) -> Aggregation | None:
        col = self.column_var.get().strip()
        func = (self.function_var.get() or "").strip()
        if not col or not func:
            return None
        alias = self.alias_var.get().strip() or None
        return Aggregation(column=col, function=func, alias=alias)


class AggregationPanel(ttk.Frame):
    """Combined GROUP BY selector + aggregation rows."""

    def __init__(
        self,
        master: tk.Misc,
        schema: Dict[str, str],
        on_change: Callable[[], None],
    ) -> None:
        super().__init__(master)
        self.schema = schema
        self._on_change = on_change
        self._rows: List[_AggregationRow] = []
        self._group_by_vars: Dict[str, tk.BooleanVar] = {}

        # --- GROUP BY row ---
        gb_frame = ttk.Frame(self)
        gb_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(gb_frame, text="GROUP BY:").pack(side="left", padx=(0, 6))
        self._gb_container = ttk.Frame(gb_frame)
        self._gb_container.pack(side="left", fill="x", expand=True)

        # --- Aggregations header + rows ---
        ttk.Separator(self, orient="horizontal").grid(
            row=1, column=0, sticky="ew", pady=2
        )
        ttk.Label(self, text="Aggregations").grid(row=2, column=0, sticky="w")
        self._rows_container = ttk.Frame(self)
        self._rows_container.grid(row=3, column=0, sticky="ew")
        self._add_btn = ttk.Button(
            self, text="+ Add aggregation", command=self.add_row
        )
        self._add_btn.grid(row=4, column=0, sticky="w", pady=(4, 0))

        self.columnconfigure(0, weight=1)
        self._rebuild_group_by_checks()

    # ------------------------------------------------------------------

    def set_schema(self, schema: Dict[str, str]) -> None:
        """Replace the schema. Does NOT fire on_change.

        The caller is responsible for triggering a refresh afterwards if
        needed (``_open_csv`` calls ``_refresh_sql`` explicitly).
        """
        self.schema = schema
        self._rebuild_group_by_checks()
        for row in self._rows:
            row.refresh_columns()

    def clear(self) -> None:
        for row in self._rows:
            row.destroy()
        self._rows = []
        for col, var in self._group_by_vars.items():
            var.set(False)
        self._fire_change()

    def get_group_by(self) -> List[str]:
        return [col for col, var in self._group_by_vars.items() if var.get()]

    def get_aggregations(self) -> List[Aggregation]:
        out: List[Aggregation] = []
        for row in self._rows:
            agg = row.to_aggregation()
            if agg is not None:
                out.append(agg)
        return out

    def add_row(self) -> None:
        self._rows.append(_AggregationRow(self, self._rows_container))
        self._fire_change()

    def set_state(
        self,
        group_by: List[str],
        aggregations: List[Aggregation],
    ) -> None:
        """Replace the GROUP BY selection and aggregation rows in one go.

        Used by the Phase 3 NL flow. Fires ``on_change`` exactly once at
        the end so the SQL preview refreshes a single time.
        """
        wanted = set(group_by)
        for col, var in self._group_by_vars.items():
            var.set(col in wanted)

        for row in self._rows:
            row.destroy()
        self._rows = []

        for agg in aggregations:
            row = _AggregationRow(self, self._rows_container)
            row.function_var.set(agg.function)
            row.column_var.set(agg.column)
            row.alias_var.set(agg.alias or "")
            self._rows.append(row)

        self._fire_change()

    # ------------------------------------------------------------------

    def _rebuild_group_by_checks(self) -> None:
        for child in self._gb_container.winfo_children():
            child.destroy()
        self._group_by_vars = {}
        for col in self.schema.keys():
            var = tk.BooleanVar(value=False)
            self._group_by_vars[col] = var
            cb = ttk.Checkbutton(
                self._gb_container,
                text=col,
                variable=var,
                command=self._fire_change,
            )
            cb.pack(side="left", padx=2)

    def _remove_row(self, row: _AggregationRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
            row.destroy()
            self._fire_change()

    def _fire_change(self) -> None:
        try:
            self._on_change()
        except Exception:  # pragma: no cover - UI callback safety
            pass
