"""Minimal SQL → HTML syntax highlighter for the dark code panel."""
from __future__ import annotations

import html
import re

_KEYWORDS = frozenset({
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "GROUP", "BY",
    "HAVING", "ORDER", "LIMIT", "AS", "ASC", "DESC", "IN", "LIKE",
    "NOT LIKE", "BETWEEN", "IS", "NULL", "DISTINCT", "JOIN", "ON",
    "LEFT", "RIGHT", "INNER", "OUTER", "UNION", "ALL", "CASE", "WHEN",
    "THEN", "ELSE", "END", "WITH",
})
_FUNCTIONS = frozenset({
    "SUM", "COUNT", "AVG", "MIN", "MAX", "ROUND", "COALESCE",
    "NULLIF", "CAST", "STRFTIME", "DATE", "UPPER", "LOWER", "LENGTH",
    "SUBSTR", "TRIM", "REPLACE", "ABS", "IFNULL",
})


def _tokenize(line: str) -> str:
    """Colour a single SQL line; input/output are plain-text / HTML."""
    escaped = html.escape(line)

    # Strings first (single-quoted)
    def sub_str(m: re.Match) -> str:
        return f'<span class="str">{m.group(0)}</span>'
    escaped = re.sub(r"'[^']*'", sub_str, escaped)

    # Inline comments
    def sub_cmt(m: re.Match) -> str:
        return f'<span class="cmt">{m.group(0)}</span>'
    escaped = re.sub(r"--[^\n]*", sub_cmt, escaped)

    # Numbers (standalone, not inside identifiers)
    def sub_num(m: re.Match) -> str:
        return f'<span class="num">{m.group(0)}</span>'
    escaped = re.sub(r"(?<![\"'\w])(\d+(?:\.\d+)?)(?![\"'\w])", sub_num, escaped)

    # Keywords and functions
    def sub_word(m: re.Match) -> str:
        word = m.group(0)
        u = word.upper()
        if u in _KEYWORDS:
            return f'<span class="kw">{word}</span>'
        if u in _FUNCTIONS:
            return f'<span class="cfn">{word}</span>'
        return word

    # Only substitute bare words (not inside existing <span> tags)
    # Simple approach: substitute outside of HTML tags
    result = re.sub(r"(?<!<[^>]{0,50})\b([A-Za-z_][A-Za-z_0-9]*)\b(?![^<]*>)", sub_word, escaped)
    return result


def render_sql_block(sql: str) -> str:
    """Return a <div class="sql-block"> with line numbers and syntax colouring."""
    lines = sql.splitlines() if sql.strip() else ["-- compose a query above --"]
    rows = []
    for i, line in enumerate(lines, 1):
        coloured = _tokenize(line) if line.strip() else "&nbsp;"
        rows.append(
            f'<div class="sql-line">'
            f'<span class="ln">{i}</span>'
            f'<span class="src">{coloured}</span>'
            f'</div>'
        )
    return '<div class="sql-block">' + "".join(rows) + "</div>"
