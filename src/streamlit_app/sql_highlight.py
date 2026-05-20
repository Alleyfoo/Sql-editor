"""Minimal SQL → HTML syntax highlighter for the dark code panel.

Single-pass tokenizer: scan the original SQL once with one compiled regex,
build the HTML output token by token. No lookbehind, no double-substitution.
"""
from __future__ import annotations

import html
import re

_KEYWORDS = frozenset({
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "GROUP", "BY",
    "HAVING", "ORDER", "LIMIT", "AS", "ASC", "DESC", "IN", "LIKE",
    "BETWEEN", "IS", "NULL", "DISTINCT", "JOIN", "ON",
    "LEFT", "RIGHT", "INNER", "OUTER", "UNION", "ALL",
    "CASE", "WHEN", "THEN", "ELSE", "END", "WITH",
})
_FUNCTIONS = frozenset({
    "SUM", "COUNT", "AVG", "MIN", "MAX", "ROUND", "COALESCE",
    "NULLIF", "CAST", "STRFTIME", "DATE", "UPPER", "LOWER",
    "LENGTH", "SUBSTR", "TRIM", "REPLACE", "ABS", "IFNULL",
})

# Single compiled regex: alternation in priority order.
# group 1 = string literal, group 2 = comment,
# group 3 = number,        group 4 = word
_TOKEN_RE = re.compile(
    r"('(?:[^']|'')*')"         # 1: single-quoted string
    r"|(--[^\n]*)"               # 2: line comment
    r"|\b(\d+(?:\.\d+)?)\b"     # 3: integer or float
    r"|([A-Za-z_]\w*)"          # 4: identifier / keyword
)


def _tokenize(line: str) -> str:
    parts: list[str] = []
    cursor = 0
    for m in _TOKEN_RE.finditer(line):
        # Escape and emit any gap before this token
        gap = line[cursor:m.start()]
        if gap:
            parts.append(html.escape(gap))

        if m.group(1):                          # string literal
            parts.append(f'<span class="str">{html.escape(m.group(1))}</span>')
        elif m.group(2):                        # comment
            parts.append(f'<span class="cmt">{html.escape(m.group(2))}</span>')
        elif m.group(3):                        # number
            parts.append(f'<span class="num">{html.escape(m.group(3))}</span>')
        else:                                   # word
            word = m.group(4)
            u = word.upper()
            esc = html.escape(word)
            if u in _KEYWORDS:
                parts.append(f'<span class="kw">{esc}</span>')
            elif u in _FUNCTIONS:
                parts.append(f'<span class="cfn">{esc}</span>')
            else:
                parts.append(esc)

        cursor = m.end()

    # Tail after last match
    if cursor < len(line):
        parts.append(html.escape(line[cursor:]))

    return "".join(parts)


def render_sql_block(sql: str) -> str:
    """Return a <div class="sql-block"> with line numbers and syntax colouring."""
    lines = sql.splitlines() if sql.strip() else ["-- compose a query above --"]
    rows = [
        f'<div class="sql-line">'
        f'<span class="ln">{i}</span>'
        f'<span class="src">{_tokenize(line) if line.strip() else "&nbsp;"}</span>'
        f'</div>'
        for i, line in enumerate(lines, 1)
    ]
    return '<div class="sql-block">' + "".join(rows) + "</div>"
