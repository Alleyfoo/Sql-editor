"""Read-only SELECT executor with a keyword blocklist.

This is the second defense layer after ``query_model._assert_select_only``.
Callers should never pass user-typed SQL here — Phase 1 has no raw-SQL
input field — but we still validate every string before it touches
SQLite.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Iterable

import pandas as pd

from .query_model import _assert_select_only

# Kept in sync with ``query_model._BLOCKED_KEYWORDS``. Any expansion must
# happen in both places.
BLOCKED_KEYWORDS = frozenset(
    {
        "DROP",
        "DELETE",
        "INSERT",
        "UPDATE",
        "ALTER",
        "CREATE",
        "ATTACH",
        "DETACH",
        "PRAGMA",
        "REPLACE",
        "TRUNCATE",
        "EXEC",
        "EXECUTE",
        "GRANT",
        "REVOKE",
    }
)

_WORD_RE = re.compile(r"[A-Za-z_]+")


class ExecutionError(RuntimeError):
    """Wraps SQLite errors and blocklist rejections for UI display."""


def _tokens_outside_strings(sql: str) -> Iterable[str]:
    """Yield word tokens from ``sql``, ignoring content inside '...' literals."""
    in_string = False
    buf: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if in_string:
            if ch == "'":
                if i + 1 < n and sql[i + 1] == "'":
                    i += 2
                    continue
                in_string = False
            i += 1
            continue
        if ch == "'":
            in_string = True
            if buf:
                yield "".join(buf)
                buf = []
            i += 1
            continue
        if ch.isalpha() or ch == "_":
            buf.append(ch)
        else:
            if buf:
                yield "".join(buf)
                buf = []
        i += 1
    if buf:
        yield "".join(buf)


def _enforce_blocklist(sql: str) -> None:
    for token in _tokens_outside_strings(sql):
        if token.upper() in BLOCKED_KEYWORDS:
            raise ExecutionError(
                f"rejected SQL: blocked keyword {token.upper()!r}"
            )


def execute(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Execute ``sql`` against ``conn`` and return the result as a DataFrame.

    Raises ``ExecutionError`` for blocked keywords, non-SELECT statements,
    or any SQLite error encountered during execution.
    """
    try:
        _assert_select_only(sql)
    except ValueError as exc:
        raise ExecutionError(str(exc)) from exc

    _enforce_blocklist(sql)

    try:
        return pd.read_sql_query(sql, conn)
    except (sqlite3.DatabaseError, pd.errors.DatabaseError) as exc:
        raise ExecutionError(f"SQLite error: {exc}") from exc


__all__ = ["execute", "ExecutionError", "BLOCKED_KEYWORDS"]
