"""Scrollable results table backed by ttk.Treeview."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

import pandas as pd


class ResultsTable(ttk.Frame):
    """Renders a pandas DataFrame in a Treeview (show='headings')."""

    def __init__(self, master: tk.Misc, max_rows: int = 1000) -> None:
        super().__init__(master)
        self.max_rows = max_rows

        self._tree = ttk.Treeview(self, show="headings")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._current_df: Optional[pd.DataFrame] = None

    @property
    def current_df(self) -> Optional[pd.DataFrame]:
        return self._current_df

    def clear(self) -> None:
        self._current_df = None
        for col in self._tree["columns"]:
            self._tree.heading(col, text="")
        self._tree["columns"] = ()
        for item in self._tree.get_children():
            self._tree.delete(item)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.clear()
        self._current_df = df

        columns = [str(c) for c in df.columns]
        self._tree["columns"] = columns
        for col in columns:
            self._tree.heading(col, text=col)
            self._tree.column(col, width=120, anchor="w")

        display = df.head(self.max_rows)
        for row in display.itertuples(index=False, name=None):
            values = tuple("" if pd.isna(v) else str(v) for v in row)
            self._tree.insert("", "end", values=values)
