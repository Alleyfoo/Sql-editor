"""Read-only SQL preview widget with basic keyword highlighting."""

from __future__ import annotations

import re
import tkinter as tk
from tkinter import ttk


_SQL_KEYWORDS = (
    "SELECT",
    "FROM",
    "WHERE",
    "AND",
    "OR",
    "NOT",
    "IN",
    "LIKE",
    "BETWEEN",
    "IS",
    "NULL",
    "GROUP",
    "BY",
    "ORDER",
    "HAVING",
    "LIMIT",
    "AS",
    "DISTINCT",
    "ASC",
    "DESC",
)
_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(_SQL_KEYWORDS) + r")\b", re.IGNORECASE
)


class SqlPreview(ttk.Frame):
    """A read-only Text widget showing the live SQL, plus a Copy button."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)

        self._text = tk.Text(
            self,
            wrap="word",
            height=10,
            width=40,
            font=("Courier", 10),
            state="disabled",
            background="#f7f7f7",
        )
        self._text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(self, command=self._text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self._text.configure(yscrollcommand=scroll.set)

        self._text.tag_configure("keyword", foreground="#0b5394", font=("Courier", 10, "bold"))
        self._text.tag_configure("error", foreground="#a00000")

        copy_btn = ttk.Button(self, text="Copy SQL", command=self._copy)
        copy_btn.grid(row=1, column=0, columnspan=2, sticky="e", pady=(4, 0))

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._current_sql = ""

    def set_sql(self, sql: str) -> None:
        self._current_sql = sql
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", sql)
        self._highlight()
        self._text.configure(state="disabled")

    def set_error(self, message: str) -> None:
        self._current_sql = ""
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", f"-- {message}")
        self._text.tag_add("error", "1.0", "end")
        self._text.configure(state="disabled")

    def _highlight(self) -> None:
        content = self._text.get("1.0", "end-1c")
        for match in _KEYWORD_RE.finditer(content):
            start = f"1.0+{match.start()}c"
            end = f"1.0+{match.end()}c"
            self._text.tag_add("keyword", start, end)

    def _copy(self) -> None:
        if not self._current_sql:
            return
        self.clipboard_clear()
        self.clipboard_append(self._current_sql)
