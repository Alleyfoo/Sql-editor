"""Pure data model for the visual query builder.

No UI imports — this module is testable in isolation.

The model captures the user's current query-builder state and can emit a
SELECT statement via ``to_sql()``. It self-validates as SELECT-only before
returning; ``executor.py`` re-validates with a second-layer keyword
blocklist at run time.

Phase 1 uses ``selected_columns`` and ``filters`` only. The other fields
(``group_by``, ``aggregations``, ``having``, ``order_by``, ``limit``) are
inert seams for Phase 2 — ``to_sql()`` emits nothing for them when empty.

TODO(Phase 2): move filter values to parameterized execution once the
executor API grows to accept parameter binds. Inlining is safe here
because (a) the model is never fed user-typed SQL, (b) values are
single-quote-escaped, (c) the executor enforces a keyword blocklist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

TEXT_OPERATORS = (
    "=",
    "!=",
    "LIKE",
    "NOT LIKE",
    "IS NULL",
    "IS NOT NULL",
)

NUMERIC_OPERATORS = (
    "=",
    "!=",
    "<",
    ">",
    "<=",
    ">=",
    "BETWEEN",
    "IS NULL",
    "IS NOT NULL",
)

DATE_OPERATORS = NUMERIC_OPERATORS

OPERATORS_BY_TYPE = {
    "text": TEXT_OPERATORS,
    "numeric": NUMERIC_OPERATORS,
    "date": DATE_OPERATORS,
}

_NULLARY_OPERATORS = {"IS NULL", "IS NOT NULL"}
_BINARY_RANGE_OPERATORS = {"BETWEEN"}

# Keywords that must never appear in generated SQL. Mirrors
# ``executor.BLOCKED_KEYWORDS`` for defense in depth.
_BLOCKED_KEYWORDS = {
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

_WORD_RE = re.compile(r"[A-Za-z_]+")
_COMMENT_RE = re.compile(r"(--[^\n]*)|(/\*.*?\*/)", re.DOTALL)


# ---------------------------------------------------------------------------
# Identifier and value quoting
# ---------------------------------------------------------------------------


def quote_ident(name: str) -> str:
    """Double-quote a SQLite identifier, escaping embedded quotes."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"invalid identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def quote_value(value: object) -> str:
    """Render a Python value as a safe SQL literal.

    Text values are single-quote escaped. Numeric values pass through.
    None becomes NULL.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        # Reject NaN/inf — they are never valid SQL literals.
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise ValueError(f"cannot serialise non-finite float: {value!r}")
        return repr(value)
    # Everything else is rendered as text with single-quote escaping.
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


# ---------------------------------------------------------------------------
# Filter / QueryModel dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Filter:
    """A single WHERE clause row."""

    column: str
    operator: str
    value: Union[str, int, float, Tuple[object, object], None] = None
    logical: str = "AND"  # joining keyword applied *before* this row

    def to_sql(self) -> str:
        op = self.operator.upper()
        col = quote_ident(self.column)
        if op in _NULLARY_OPERATORS:
            return f"{col} {op}"
        if op in _BINARY_RANGE_OPERATORS:
            if not isinstance(self.value, tuple) or len(self.value) != 2:
                raise ValueError(
                    f"BETWEEN requires a (low, high) tuple, got {self.value!r}"
                )
            lo, hi = self.value
            return f"{col} BETWEEN {quote_value(lo)} AND {quote_value(hi)}"
        return f"{col} {op} {quote_value(self.value)}"


@dataclass
class Aggregation:
    """Phase 2 seam — unused in Phase 1."""

    column: str
    function: str  # SUM, COUNT, AVG, MIN, MAX, COUNT DISTINCT
    alias: Optional[str] = None


@dataclass
class QueryModel:
    """Current visual-query-builder state.

    Only ``selected_columns`` and ``filters`` are wired up in Phase 1.
    """

    table: str = "data"
    selected_columns: list = field(default_factory=list)
    filters: list = field(default_factory=list)

    # Phase 2 seams — ignored when empty.
    group_by: list = field(default_factory=list)
    aggregations: list = field(default_factory=list)
    having: list = field(default_factory=list)
    order_by: list = field(default_factory=list)
    limit: Optional[int] = None

    # ------------------------------------------------------------------
    # SQL generation
    # ------------------------------------------------------------------

    def to_sql(self) -> str:
        select_clause = self._select_clause()
        from_clause = f"FROM {quote_ident(self.table)}"
        where_clause = self._where_clause(self.filters)

        parts = [f"SELECT {select_clause}", from_clause]
        if where_clause:
            parts.append(f"WHERE {where_clause}")

        # Phase 2 seams
        if self.group_by:
            parts.append(
                "GROUP BY " + ", ".join(quote_ident(c) for c in self.group_by)
            )
        if self.having:
            parts.append("HAVING " + self._where_clause(self.having))
        if self.order_by:
            parts.append(
                "ORDER BY "
                + ", ".join(
                    f"{quote_ident(col)} {direction.upper()}"
                    for col, direction in self.order_by
                )
            )
        if self.limit is not None:
            if not isinstance(self.limit, int) or self.limit < 0:
                raise ValueError(f"invalid LIMIT: {self.limit!r}")
            parts.append(f"LIMIT {self.limit}")

        sql = " ".join(parts)
        _assert_select_only(sql)
        return sql

    # ------------------------------------------------------------------

    def _select_clause(self) -> str:
        if self.aggregations:
            # Phase 2 path — include aggregations alongside any group-by cols.
            pieces = []
            for col in self.selected_columns:
                pieces.append(quote_ident(col))
            for agg in self.aggregations:
                func = agg.function.upper()
                col = "*" if agg.column == "*" else quote_ident(agg.column)
                expr = f"{func}({col})"
                if agg.alias:
                    expr += f" AS {quote_ident(agg.alias)}"
                pieces.append(expr)
            return ", ".join(pieces) if pieces else "*"
        if not self.selected_columns:
            return "*"
        return ", ".join(quote_ident(c) for c in self.selected_columns)

    @staticmethod
    def _where_clause(filters: list) -> str:
        if not filters:
            return ""
        pieces: list = []
        for idx, f in enumerate(filters):
            frag = f.to_sql()
            if idx == 0:
                pieces.append(frag)
            else:
                joiner = (f.logical or "AND").upper()
                if joiner not in {"AND", "OR"}:
                    raise ValueError(f"invalid logical joiner: {f.logical!r}")
                pieces.append(f"{joiner} {frag}")
        return " ".join(pieces)


# ---------------------------------------------------------------------------
# SELECT-only validator
# ---------------------------------------------------------------------------


def _strip_string_literals(sql: str) -> str:
    """Remove single-quoted string literals so keyword scanning can't be
    fooled by values like ``'; DROP TABLE data;--``."""
    out = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":
            # Skip to the matching close quote, respecting '' escapes.
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append("''")
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _assert_select_only(sql: str) -> None:
    """Raise ValueError if ``sql`` is anything other than a single SELECT."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("empty SQL")

    stripped = _COMMENT_RE.sub(" ", sql).strip()
    if not stripped:
        raise ValueError("SQL contains only comments")

    # Statement must begin with SELECT.
    head = stripped.lstrip().split(None, 1)[0].upper()
    if head != "SELECT":
        raise ValueError(f"only SELECT statements are allowed, got: {head!r}")

    # No trailing statements.
    without_strings = _strip_string_literals(stripped).rstrip().rstrip(";")
    if ";" in without_strings:
        raise ValueError("multiple statements are not allowed")

    # No blocked keywords anywhere outside string literals.
    words = {w.upper() for w in _WORD_RE.findall(without_strings)}
    bad = words & _BLOCKED_KEYWORDS
    if bad:
        raise ValueError(f"blocked SQL keyword(s) in generated SQL: {sorted(bad)}")


__all__ = [
    "Filter",
    "Aggregation",
    "QueryModel",
    "OPERATORS_BY_TYPE",
    "TEXT_OPERATORS",
    "NUMERIC_OPERATORS",
    "DATE_OPERATORS",
    "quote_ident",
    "quote_value",
]
