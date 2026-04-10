"""Dynamic filter row composer.

Each row represents one WHERE condition:

    [AND/OR] [column] [operator] [value] [X]

The first row has its AND/OR toggle disabled. Operators available depend
on the selected column's type (text / numeric / date). ``BETWEEN``
swaps the single value entry for two. ``IS NULL`` / ``IS NOT NULL``
hide the value entry entirely.

``get_filters()`` returns a list of ``query_model.Filter`` instances
matching the current UI state. Any change fires the ``on_change``
callback passed at construction time.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional

from ..query_model import (
    Filter,
    OPERATORS_BY_TYPE,
    TEXT_OPERATORS,
)


class _FilterRow:
    """One row of the filter composer."""

    def __init__(
        self,
        parent: "FilterRows",
        container: ttk.Frame,
        index: int,
    ) -> None:
        self._parent = parent
        self._frame = ttk.Frame(container)
        self._index = index

        self.logical_var = tk.StringVar(value="AND")
        self.column_var = tk.StringVar()
        self.operator_var = tk.StringVar()
        self.value_var = tk.StringVar()
        self.value2_var = tk.StringVar()

        self._logical_box = ttk.Combobox(
            self._frame,
            textvariable=self.logical_var,
            values=("AND", "OR"),
            width=5,
            state="readonly",
        )
        self._column_box = ttk.Combobox(
            self._frame,
            textvariable=self.column_var,
            values=list(parent.schema.keys()),
            width=18,
            state="readonly",
        )
        self._operator_box = ttk.Combobox(
            self._frame,
            textvariable=self.operator_var,
            values=list(TEXT_OPERATORS),
            width=12,
            state="readonly",
        )
        self._value_entry = ttk.Entry(
            self._frame, textvariable=self.value_var, width=16
        )
        self._value2_entry = ttk.Entry(
            self._frame, textvariable=self.value2_var, width=10
        )
        self._remove_btn = ttk.Button(
            self._frame, text="X", width=2, command=self._remove
        )

        self._logical_box.grid(row=0, column=0, padx=2)
        self._column_box.grid(row=0, column=1, padx=2)
        self._operator_box.grid(row=0, column=2, padx=2)
        self._value_entry.grid(row=0, column=3, padx=2)
        self._value2_entry.grid(row=0, column=4, padx=2)
        self._remove_btn.grid(row=0, column=5, padx=2)

        self._column_box.bind("<<ComboboxSelected>>", self._on_column_changed)
        self._operator_box.bind("<<ComboboxSelected>>", self._on_operator_changed)
        self.value_var.trace_add("write", lambda *_: self._parent._fire_change())
        self.value2_var.trace_add("write", lambda *_: self._parent._fire_change())
        self.logical_var.trace_add("write", lambda *_: self._parent._fire_change())

        self._frame.grid(sticky="ew", pady=1)

    # ------------------------------------------------------------------

    def set_is_first(self, is_first: bool) -> None:
        state = "disabled" if is_first else "readonly"
        self._logical_box.configure(state=state)

    def destroy(self) -> None:
        self._frame.destroy()

    # ------------------------------------------------------------------

    def _on_column_changed(self, _event: object = None) -> None:
        col = self.column_var.get()
        col_type = self._parent.schema.get(col, "text")
        ops = OPERATORS_BY_TYPE.get(col_type, TEXT_OPERATORS)
        self._operator_box.configure(values=list(ops))
        # Reset operator if it is no longer valid.
        if self.operator_var.get() not in ops:
            self.operator_var.set(ops[0])
        self._update_value_visibility()
        self._parent._fire_change()

    def _on_operator_changed(self, _event: object = None) -> None:
        self._update_value_visibility()
        self._parent._fire_change()

    def _update_value_visibility(self) -> None:
        op = (self.operator_var.get() or "").upper()
        if op in {"IS NULL", "IS NOT NULL"}:
            self._value_entry.grid_remove()
            self._value2_entry.grid_remove()
        elif op == "BETWEEN":
            self._value_entry.grid()
            self._value2_entry.grid()
        else:
            self._value_entry.grid()
            self._value2_entry.grid_remove()

    def _remove(self) -> None:
        self._parent._remove_row(self)

    # ------------------------------------------------------------------

    def to_filter(self) -> Optional[Filter]:
        col = self.column_var.get().strip()
        op = (self.operator_var.get() or "").strip().upper()
        if not col or not op:
            return None
        if op in {"IS NULL", "IS NOT NULL"}:
            value: object = None
        elif op == "BETWEEN":
            lo = self.value_var.get()
            hi = self.value2_var.get()
            if lo == "" or hi == "":
                return None
            value = (_coerce(lo, self._parent.schema.get(col, "text")),
                     _coerce(hi, self._parent.schema.get(col, "text")))
        else:
            raw = self.value_var.get()
            if raw == "":
                return None
            value = _coerce(raw, self._parent.schema.get(col, "text"))
        return Filter(
            column=col,
            operator=op,
            value=value,
            logical=self.logical_var.get() or "AND",
        )


def _coerce(raw: str, col_type: str) -> object:
    """Best-effort conversion of a string value to the column's type."""
    if col_type == "numeric":
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            return raw
    return raw


# ---------------------------------------------------------------------------


class FilterRows(ttk.Frame):
    """Container widget that manages the list of filter rows."""

    def __init__(
        self,
        master: tk.Misc,
        schema: Dict[str, str],
        on_change: Callable[[], None],
    ) -> None:
        super().__init__(master)
        self.schema = schema
        self._on_change = on_change
        self._rows: List[_FilterRow] = []

        self._rows_container = ttk.Frame(self)
        self._rows_container.grid(row=0, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)

        self._add_btn = ttk.Button(self, text="+ Add filter", command=self.add_row)
        self._add_btn.grid(row=1, column=0, sticky="w", pady=(4, 0))

    # ------------------------------------------------------------------

    def set_schema(self, schema: Dict[str, str]) -> None:
        """Replace the schema shown in column dropdowns.

        Intentionally does NOT fire on_change — the caller may be mid-refresh
        and a fresh event would cause re-entrancy.
        """
        self.schema = schema
        for row in self._rows:
            row._column_box.configure(values=list(schema.keys()))

    def add_row(self) -> None:
        row = _FilterRow(self, self._rows_container, len(self._rows))
        self._rows.append(row)
        self._relabel()
        self._fire_change()

    def clear(self) -> None:
        for row in self._rows:
            row.destroy()
        self._rows = []
        self._fire_change()

    def get_filters(self) -> List[Filter]:
        result: List[Filter] = []
        for row in self._rows:
            f = row.to_filter()
            if f is not None:
                result.append(f)
        return result

    # ------------------------------------------------------------------

    def _remove_row(self, row: _FilterRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
            row.destroy()
            self._relabel()
            self._fire_change()

    def _relabel(self) -> None:
        for idx, row in enumerate(self._rows):
            row.set_is_first(idx == 0)

    def _fire_change(self) -> None:
        try:
            self._on_change()
        except Exception:  # pragma: no cover - UI callback safety
            pass
