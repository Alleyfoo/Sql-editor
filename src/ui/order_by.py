"""Dynamic ORDER BY rows.

Each row: [column dropdown] [direction dropdown: ASC/DESC] [X remove]

``get_order_by()`` returns ``list[tuple[column, direction]]`` suitable for
``QueryModel.order_by``. Rows with no column selected are dropped.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Sequence, Tuple

from ..query_model import ORDER_DIRECTIONS


class _OrderRow:
    def __init__(self, parent: "OrderByRows", container: ttk.Frame) -> None:
        self._parent = parent
        self._frame = ttk.Frame(container)

        self.column_var = tk.StringVar()
        self.direction_var = tk.StringVar(value="ASC")

        self._column_box = ttk.Combobox(
            self._frame,
            textvariable=self.column_var,
            values=list(parent.columns),
            width=18,
            state="readonly",
        )
        self._direction_box = ttk.Combobox(
            self._frame,
            textvariable=self.direction_var,
            values=list(ORDER_DIRECTIONS),
            width=6,
            state="readonly",
        )
        self._remove_btn = ttk.Button(
            self._frame, text="X", width=2, command=self._remove
        )

        self._column_box.grid(row=0, column=0, padx=2)
        self._direction_box.grid(row=0, column=1, padx=2)
        self._remove_btn.grid(row=0, column=2, padx=(6, 2))

        self.column_var.trace_add("write", lambda *_: parent._fire_change())
        self.direction_var.trace_add("write", lambda *_: parent._fire_change())

        self._frame.grid(sticky="ew", pady=1)

    def refresh_columns(self) -> None:
        self._column_box.configure(values=list(self._parent.columns))

    def destroy(self) -> None:
        self._frame.destroy()

    def _remove(self) -> None:
        self._parent._remove_row(self)

    def to_entry(self) -> Tuple[str, str] | None:
        col = self.column_var.get().strip()
        if not col:
            return None
        direction = (self.direction_var.get() or "ASC").strip().upper()
        return col, direction


class OrderByRows(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        columns: Sequence[str],
        on_change: Callable[[], None],
    ) -> None:
        super().__init__(master)
        self.columns: List[str] = list(columns)
        self._on_change = on_change
        self._rows: List[_OrderRow] = []

        self._rows_container = ttk.Frame(self)
        self._rows_container.grid(row=0, column=0, sticky="ew")
        self._add_btn = ttk.Button(
            self, text="+ Add sort", command=self.add_row
        )
        self._add_btn.grid(row=1, column=0, sticky="w", pady=(4, 0))

    # ------------------------------------------------------------------

    def set_columns(self, columns: Sequence[str]) -> None:
        """Replace the column list shown in the row dropdowns.

        Intentionally does NOT fire on_change — the caller may be mid-refresh.
        """
        self.columns = list(columns)
        for row in self._rows:
            row.refresh_columns()

    def add_row(self) -> None:
        self._rows.append(_OrderRow(self, self._rows_container))
        self._fire_change()

    def clear(self) -> None:
        for row in self._rows:
            row.destroy()
        self._rows = []
        self._fire_change()

    def set_order_by(self, entries: Sequence[Tuple[str, str]]) -> None:
        """Replace all rows with the given (column, direction) pairs.

        Used by the Phase 3 NL flow. Fires ``on_change`` exactly once at
        the end so the SQL preview refreshes a single time.
        """
        for row in self._rows:
            row.destroy()
        self._rows = []
        for col, direction in entries:
            row = _OrderRow(self, self._rows_container)
            row.column_var.set(col)
            row.direction_var.set(
                direction if direction in ORDER_DIRECTIONS else "ASC"
            )
            self._rows.append(row)
        self._fire_change()

    def get_order_by(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for row in self._rows:
            entry = row.to_entry()
            if entry is not None:
                out.append(entry)
        return out

    def _remove_row(self, row: _OrderRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
            row.destroy()
            self._fire_change()

    def _fire_change(self) -> None:
        try:
            self._on_change()
        except Exception:  # pragma: no cover - UI callback safety
            pass
