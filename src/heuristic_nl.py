"""Offline natural-language → ``QueryModel`` heuristic.

This is a deliberately small, deterministic parser used as a fallback
when the LLM is unreachable (and as an optional fast-path even when it
isn't). It produces real ``QueryModel`` instances so the existing
``to_sql()`` validator and executor blocklist still apply — there is no
new code path into SQL generation.

Scope
-----
Handles a useful subset of analytics phrasings:

* aggregations (``sum/total/avg/average/mean/count/min/max``) over a
  measurable column, optionally grouped ``by/per/for each X``;
* ranking (``top N`` / ``bottom N``);
* ``ORDER BY X (asc|desc)``;
* simple filters: ``> n``, ``< n``, ``>= n``, ``<= n``, ``= 'x'``,
  ``contains "x"``, ``is null``, ``is not null``,
  ``between A and B``;
* relative date windows: ``last N days`` / ``past N days``,
  ``since YYYY-MM-DD``, ``in YYYY``.

Anything outside this surface is reported as low confidence and the
caller should refuse rather than guess. Joins, window functions,
percentiles, rolling stats, and similar are intentionally not handled
here; those belong in the LLM/Python paths.

The parser is designed for unit-testability: every rule is a small
function over the lowercase token list, schema, and a result builder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from src.ingestion import TABLE_NAME
from src.query_model import (
    DATE_OPERATORS,
    NUMERIC_OPERATORS,
    TEXT_OPERATORS,
    Aggregation,
    Filter,
    QueryModel,
)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

_AGG_VERBS: Dict[str, str] = {
    "sum": "SUM",
    "total": "SUM",
    "totals": "SUM",
    "average": "AVG",
    "avg": "AVG",
    "mean": "AVG",
    "count": "COUNT",
    "max": "MAX",
    "maximum": "MAX",
    "highest": "MAX",
    "largest": "MAX",
    "biggest": "MAX",
    "min": "MIN",
    "minimum": "MIN",
    "lowest": "MIN",
    "smallest": "MIN",
}

_GROUP_PREPS = ("by", "per")  # also "for each" handled separately
_ORDER_DESC_VERBS = ("top", "highest", "largest", "biggest", "best")
_ORDER_ASC_VERBS = ("bottom", "lowest", "smallest", "worst")

_FILTER_OPS_NUMERIC = {
    "greater than or equal to": ">=",
    "less than or equal to": "<=",
    "at least": ">=",
    "at most": "<=",
    "greater than": ">",
    "more than": ">",
    "above": ">",
    "over": ">",
    "less than": "<",
    "fewer than": "<",
    "below": "<",
    "under": "<",
    "equal to": "=",
    "equals": "=",
    "is": "=",
    "=": "=",
    ">=": ">=",
    "<=": "<=",
    ">": ">",
    "<": "<",
}

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "show",
        "give",
        "list",
        "find",
        "me",
        "please",
        "all",
        "rows",
        "row",
        "data",
        "records",
        "record",
        "entries",
        "entry",
        "where",
        "and",
        "with",
        "in",
        "to",
        "that",
        "are",
        "is",
    }
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class HeuristicResult:
    """Outcome of a heuristic parse."""

    model: Optional[QueryModel]
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    unrecognized: List[str] = field(default_factory=list)

    @property
    def parsed(self) -> bool:
        return self.model is not None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_CONFIDENCE_FLOOR = 0.35

# Confidence at or above which the heuristic is trusted enough to bypass
# the LLM entirely, even when the LLM is reachable. Tuned so that
# unambiguous patterns ("top N X by Y", "sum X by Y", "count rows") qualify
# while ambiguous single-intent prompts still defer to the LLM.
HEURISTIC_FAST_PATH_THRESHOLD = 0.70


def parse_heuristic(nl: str, schema: Dict[str, str]) -> HeuristicResult:
    """Parse ``nl`` against ``schema`` and return a :class:`HeuristicResult`.

    Never raises; on failure ``model`` is ``None`` and ``confidence`` is
    low. Callers should refuse to act when ``model is None``.
    """
    if not isinstance(nl, str) or not nl.strip():
        return HeuristicResult(model=None, confidence=0.0, reasoning=["empty input"])
    if not isinstance(schema, dict) or not schema:
        return HeuristicResult(
            model=None, confidence=0.0, reasoning=["no dataset loaded"]
        )

    tokens = _tokenize(nl)
    if not tokens:
        return HeuristicResult(
            model=None, confidence=0.0, reasoning=["nothing to parse"]
        )

    builder = _Builder(schema=schema)

    # Order matters: detect filters & dates before aggregations, because
    # an aggregation may reference a column we mention only inside a
    # filter ("revenue > 500" -> revenue is a measure, not a group).
    _detect_limit_and_rank(tokens, builder)
    _detect_filters(nl, tokens, builder)
    _detect_date_windows(nl, builder)
    # The "top N X by Y" pattern must run BEFORE generic group-by so it
    # can claim the right columns (X is the group, Y is the measure).
    _detect_top_n_pattern(tokens, builder)
    _detect_how_many(tokens, builder)
    _detect_groups(tokens, builder)
    _detect_aggregations(tokens, builder)
    _detect_order_by(tokens, builder)
    _detect_simple_select(tokens, builder)

    model = builder.finalize()
    confidence = builder.confidence()
    reasoning = builder.reasoning
    unrecognized = builder.unrecognized()

    if model is None or confidence < _CONFIDENCE_FLOOR:
        return HeuristicResult(
            model=None,
            confidence=confidence,
            reasoning=reasoning
            + [
                "confidence below threshold "
                f"({confidence:.2f} < {_CONFIDENCE_FLOOR})"
            ],
            unrecognized=unrecognized,
        )

    return HeuristicResult(
        model=model,
        confidence=confidence,
        reasoning=reasoning,
        unrecognized=unrecognized,
    )


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[<>]=?|=|\"[^\"]*\"|'[^']*'"
)


def _tokenize(nl: str) -> List[str]:
    """Lower-case alpha tokens, keep numbers, comparison operators, and
    quoted strings (preserving their original case)."""
    out: List[str] = []
    for match in _TOKEN_RE.finditer(nl):
        tok = match.group(0)
        if tok.startswith("'") or tok.startswith('"'):
            out.append(tok)  # keep quoted literals as-is
        elif re.fullmatch(r"\d+(?:\.\d+)?", tok):
            out.append(tok)
        elif tok in {">", "<", ">=", "<=", "="}:
            out.append(tok)
        else:
            out.append(tok.lower())
    return out


# ---------------------------------------------------------------------------
# Schema-aware column matching
# ---------------------------------------------------------------------------


def _column_score(token: str, column: str) -> float:
    """Return a similarity score in [0, 1] for ``token`` vs ``column``.

    Exact match (case-insensitive) wins. Substring and snake-case
    components score lower so we still prefer ``revenue`` over
    ``revenue_pct`` when the user says "revenue".
    """
    t = token.lower()
    c = column.lower()
    if t == c:
        return 1.0
    parts = c.split("_")
    if t in parts:
        return 0.9
    if t in c:
        return 0.7
    if c in t:
        return 0.6
    return 0.0


def _match_column(token: str, schema: Dict[str, str]) -> Optional[str]:
    best: Tuple[float, Optional[str]] = (0.0, None)
    for col in schema:
        score = _column_score(token, col)
        if score > best[0]:
            best = (score, col)
    if best[0] >= 0.6:
        return best[1]
    return None


def _match_columns_in_phrase(
    tokens: List[str], schema: Dict[str, str], indices: range
) -> List[Tuple[int, str]]:
    """Walk ``tokens[indices]`` and return ``(index, column)`` for any
    token that maps to a schema column."""
    found: List[Tuple[int, str]] = []
    for i in indices:
        col = _match_column(tokens[i], schema)
        if col is not None:
            found.append((i, col))
    return found


# ---------------------------------------------------------------------------
# Builder helper
# ---------------------------------------------------------------------------


class _Builder:
    """Mutable scratchpad accumulated during parsing.

    ``finalize()`` collapses the scratchpad into a ``QueryModel`` — or
    ``None`` if no recognizable intent was found.
    """

    def __init__(self, schema: Dict[str, str]) -> None:
        self.schema = schema
        self.reasoning: List[str] = []
        self.consumed: set = set()  # token indices we recognized
        self.total_tokens: int = 0  # populated lazily by ``confidence()``

        # Scratchpad fields.
        self.aggregations: List[Aggregation] = []
        self.group_by: List[str] = []
        self.selected_columns: List[str] = []
        self.filters: List[Filter] = []
        self.order_by: List[Tuple[str, str]] = []
        self.limit: Optional[int] = None
        self.date_buckets: Dict[str, str] = {}
        self.scalar_count: bool = False  # "how many rows"
        self.simple_show: bool = False  # "show first N rows" / "show all"

    # ---- bookkeeping -----------------------------------------------------

    def consume(self, *idxs: int) -> None:
        for i in idxs:
            self.consumed.add(i)

    def add_reason(self, reason: str) -> None:
        if reason and reason not in self.reasoning:
            self.reasoning.append(reason)

    def unrecognized(self) -> List[str]:
        return []

    # ---- confidence ------------------------------------------------------

    def confidence(self) -> float:
        """Soft heuristic: did we recognize at least one strong intent?"""
        if (
            not self.aggregations
            and not self.filters
            and not self.order_by
            and not self.simple_show
            and not self.scalar_count
            and not self.selected_columns
            and not self.group_by
            and self.limit is None
        ):
            return 0.0

        score = 0.0
        if self.aggregations:
            # Strongest signal: an explicit aggregation verb mapped to a
            # real numeric column. Bumped in Phase 3 so that "sum X by Y"
            # clears the fast-path threshold on its own.
            score += 0.50
        if self.group_by:
            score += 0.25
        if self.filters:
            score += 0.20
        if self.order_by or self.limit is not None:
            score += 0.15
        if self.simple_show or self.scalar_count:
            score += 0.30
        if self.selected_columns and not self.aggregations:
            score += 0.15
        return min(1.0, score)

    # ---- finalisation ----------------------------------------------------

    def finalize(self) -> Optional[QueryModel]:
        # If aggregations are present, ensure SELECT is consistent: the
        # group-by columns get added to selected_columns.
        if self.aggregations:
            for col in self.group_by:
                if col not in self.selected_columns:
                    self.selected_columns.append(col)
            if self.scalar_count:
                # "how many rows" wins over any selection.
                self.selected_columns = []
                self.group_by = []
        elif self.scalar_count:
            self.aggregations = [
                Aggregation(column="*", function="COUNT", alias="row_count")
            ]
            self.selected_columns = []
            self.group_by = []

        if (
            not self.aggregations
            and not self.selected_columns
            and not self.simple_show
            and not self.filters
            and not self.order_by
            and self.limit is None
        ):
            return None

        reply = "; ".join(self.reasoning) if self.reasoning else "Heuristic match."
        return QueryModel(
            table=TABLE_NAME,
            selected_columns=list(self.selected_columns),
            filters=list(self.filters),
            group_by=list(self.group_by),
            aggregations=list(self.aggregations),
            order_by=list(self.order_by),
            limit=self.limit,
            date_buckets=dict(self.date_buckets),
            reply=reply,
        )


# ---------------------------------------------------------------------------
# Detection rules
# ---------------------------------------------------------------------------


def _detect_limit_and_rank(tokens: List[str], b: _Builder) -> None:
    """Handle ``top N`` / ``bottom N`` / ``first N`` / ``last N``.

    These imply both a LIMIT and an order direction. The actual ORDER BY
    column is filled in later by :func:`_detect_aggregations` (which
    knows the measure column) or :func:`_detect_order_by`.
    """
    n = len(tokens)
    pending_dir: Optional[str] = None  # set by aggregation step if needed
    for i, tok in enumerate(tokens):
        if (
            tok in _ORDER_DESC_VERBS
            or tok in _ORDER_ASC_VERBS
            or tok in {"first", "last"}
        ):
            j = i + 1
            if j < n and re.fullmatch(r"\d+", tokens[j]):
                limit_val = int(tokens[j])
                if 1 <= limit_val <= 1_000_000:
                    b.limit = limit_val
                    direction = (
                        "ASC" if tok in _ORDER_ASC_VERBS or tok == "last" else "DESC"
                    )
                    if tok == "first":
                        direction = "ASC"
                    if tok == "last":
                        direction = "DESC"
                    b.add_reason(f"limit {limit_val} ({direction.lower()})")
                    b.consume(i, j)
                    # Stash the desired direction on the builder so the
                    # aggregation/order step can apply it.
                    setattr(b, "_pending_dir", direction)


def _detect_top_n_pattern(tokens: List[str], b: _Builder) -> None:
    """Recognize ``top N X by Y`` (and ``bottom N X by Y``) as
    "GROUP BY X, SUM(Y), ORDER BY sum_Y, LIMIT N".

    Runs before :func:`_detect_groups` so it can claim ``X`` as the
    group column instead of ``Y``. ``Y`` must be numeric for SUM to make
    sense; otherwise we fall through to the generic detectors.
    """
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if tok not in _ORDER_DESC_VERBS and tok not in _ORDER_ASC_VERBS:
            continue
        if i + 1 >= n or not re.fullmatch(r"\d+", tokens[i + 1]):
            continue
        # X = first column-like token after the limit
        x_idx: Optional[int] = None
        x_col: Optional[str] = None
        for j in range(i + 2, min(i + 6, n)):
            if tokens[j] in _STOPWORDS or tokens[j] in {"by", "per"}:
                continue
            col = _match_column(tokens[j], b.schema)
            if col is not None:
                x_idx = j
                x_col = col
                break
        if x_col is None:
            continue
        # Find "by"/"per" after X, then Y as the next column.
        y_col: Optional[str] = None
        y_idx: Optional[int] = None
        for k in range(x_idx + 1, n):
            if tokens[k] in {"by", "per"}:
                for m in range(k + 1, min(k + 4, n)):
                    cand = _match_column(tokens[m], b.schema)
                    if cand is not None:
                        y_col = cand
                        y_idx = m
                        break
                break
        if y_col is None:
            continue
        if b.schema.get(y_col) != "numeric":
            continue
        if x_col == y_col:
            continue

        # Apply: X group, SUM(Y) agg, ORDER BY sum_Y DESC|ASC, LIMIT
        if x_col not in b.group_by:
            b.group_by.append(x_col)
        alias = f"sum_{y_col}"
        if not any(a.function == "SUM" and a.column == y_col for a in b.aggregations):
            b.aggregations.append(
                Aggregation(column=y_col, function="SUM", alias=alias)
            )
        direction = "ASC" if tok in _ORDER_ASC_VERBS else "DESC"
        if not b.order_by:
            b.order_by.append((alias, direction))
        b.consume(i, i + 1, x_idx, y_idx)
        b.add_reason(f"top {tokens[i + 1]} {x_col} by sum({y_col})")
        return


def _detect_how_many(tokens: List[str], b: _Builder) -> None:
    """Treat ``how many rows`` / ``how many records`` as scalar COUNT.

    Runs before :func:`_detect_aggregations` because the literal token
    ``count`` may not appear in the prompt.
    """
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if tok != "many" or i == 0 or tokens[i - 1] != "how":
            continue
        # Optional "rows/records/entries" right after.
        if i + 1 < n and tokens[i + 1] in {"rows", "records", "entries"}:
            b.consume(i + 1)
        b.scalar_count = True
        b.add_reason("count rows")
        b.consume(i - 1, i)
        return


def _detect_aggregations(tokens: List[str], b: _Builder) -> None:
    n = len(tokens)
    for i, tok in enumerate(tokens):
        func = _AGG_VERBS.get(tok)
        if func is None:
            continue
        # "count rows" / "how many" → scalar count
        if func == "COUNT" and (
            (i + 1 < n and tokens[i + 1] in {"rows", "records", "entries"})
            or (i >= 1 and tokens[i - 1] in {"how", "many"})
        ):
            b.scalar_count = True
            b.consume(i)
            if i + 1 < n and tokens[i + 1] in {"rows", "records", "entries"}:
                b.consume(i + 1)
            b.add_reason("count rows")
            continue

        # Look ahead for the measure column. Scan up to 4 tokens ahead.
        target_col: Optional[str] = None
        target_idx: Optional[int] = None
        for j in range(i + 1, min(i + 5, n)):
            if tokens[j] in _STOPWORDS:
                continue
            col = _match_column(tokens[j], b.schema)
            if col is not None:
                target_col = col
                target_idx = j
                break
        if target_col is None:
            continue
        if (
            target_col == "*"
            or b.schema.get(target_col) != "numeric"
            and func != "COUNT"
        ):
            # SUM/AVG/MIN/MAX require a numeric column; reject otherwise
            # (e.g. "average region" makes no sense).
            continue

        alias = f"{func.lower()}_{target_col}"
        b.aggregations.append(
            Aggregation(column=target_col, function=func, alias=alias)
        )
        b.consume(i, target_idx)
        b.add_reason(f"{func.lower()} of {target_col}")

        # If a "top N"/"bottom N" pending direction exists, set ORDER BY
        # on this aggregation.
        pending_dir = getattr(b, "_pending_dir", None)
        if pending_dir and not b.order_by:
            b.order_by.append((alias, pending_dir))
            b.add_reason(f"order by {alias} {pending_dir.lower()}")


def _detect_groups(tokens: List[str], b: _Builder) -> None:
    """Detect ``by X`` / ``per X`` / ``for each X`` group hints."""
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        if tok in _GROUP_PREPS or (
            tok == "for" and i + 1 < n and tokens[i + 1] == "each"
        ):
            start = i + 2 if tok == "for" else i + 1
            # Columns already claimed as aggregation measures (e.g. via the
            # "top N X by Y" pattern) must not be re-added as group keys.
            agg_columns = {a.column for a in b.aggregations}
            for j in range(start, min(start + 4, n)):
                col = _match_column(tokens[j], b.schema)
                if col is None:
                    continue
                if col in agg_columns:
                    # Treat this `by`/`per` as already handled by the
                    # top-N rule and consume the tokens so downstream
                    # detectors don't re-process them either.
                    b.consume(i, j)
                    if tok == "for":
                        b.consume(i + 1)
                    break
                if col not in b.group_by:
                    b.group_by.append(col)
                    b.add_reason(f"group by {col}")
                    b.consume(i, j)
                    if tok == "for":
                        b.consume(i + 1)
                break
        i += 1


def _detect_simple_select(tokens: List[str], b: _Builder) -> None:
    """Handle "show {col1}, {col2}" or "list {col}" with no aggregation."""
    if b.aggregations or b.scalar_count:
        return
    select_verb_seen = any(
        tok in {"show", "list", "display", "give", "find"} for tok in tokens
    )
    if not select_verb_seen and not b.filters and not b.order_by:
        return
    cols: List[str] = []
    for i, tok in enumerate(tokens):
        col = _match_column(tok, b.schema)
        if col is not None and col not in cols:
            cols.append(col)
            b.consume(i)
    if cols:
        b.selected_columns = cols
        b.add_reason(f"select {', '.join(cols)}")
        b.simple_show = True
    elif select_verb_seen:
        # Mark "show all rows" intent even without explicit columns.
        b.simple_show = True
        b.add_reason("select all columns")


def _detect_order_by(tokens: List[str], b: _Builder) -> None:
    """Detect ``sort/sorted/order by X (asc|desc)``."""
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if tok not in {"sort", "sorted", "ordered", "order"}:
            continue
        # Skip following "by"
        j = i + 1
        if j < n and tokens[j] == "by":
            j += 1
        if j >= n:
            continue
        col = _match_column(tokens[j], b.schema)
        if col is None:
            continue
        direction = "DESC"
        if j + 1 < n and tokens[j + 1] in {"asc", "ascending"}:
            direction = "ASC"
            b.consume(j + 1)
        elif j + 1 < n and tokens[j + 1] in {"desc", "descending"}:
            direction = "DESC"
            b.consume(j + 1)
        if (col, direction) not in b.order_by:
            b.order_by.append((col, direction))
            b.add_reason(f"order by {col} {direction.lower()}")
            b.consume(i, j)
            if j - 1 > i and tokens[j - 1] == "by":
                b.consume(j - 1)


# --- filters --------------------------------------------------------------


_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _detect_filters(nl: str, tokens: List[str], b: _Builder) -> None:
    """Detect simple comparison and equality filters.

    We deliberately scan the *raw* string for multi-word operators
    ("greater than or equal to") before falling back to per-token
    operator symbols (``>``, ``<``, ``=``, ``>=``, ``<=``).
    """
    low = nl.lower()

    # Multi-word operators first (longest phrase match wins).
    for phrase in sorted(_FILTER_OPS_NUMERIC, key=len, reverse=True):
        if phrase not in low:
            continue
        op = _FILTER_OPS_NUMERIC[phrase]
        # Find the column referenced before the phrase, and the value
        # immediately after.
        idx = low.find(phrase)
        before = low[:idx]
        after = low[idx + len(phrase) :]
        col = _find_last_column_in_text(before, b.schema)
        value = _find_first_value_in_text(after, b.schema.get(col) if col else None)
        if col is None or value is None:
            continue
        op_set = {
            "numeric": NUMERIC_OPERATORS,
            "date": DATE_OPERATORS,
            "text": TEXT_OPERATORS,
        }.get(b.schema.get(col), TEXT_OPERATORS)
        if op not in op_set:
            continue
        # Avoid duplicating filters when multiple synonyms hit ("more than" + ">").
        if any(
            f.column == col and f.operator == op and f.value == value for f in b.filters
        ):
            continue
        b.filters.append(Filter(column=col, operator=op, value=value, logical="AND"))
        b.add_reason(f"filter {col} {op} {value!r}")
        # Cannot easily mark token indices from raw scan; treat the
        # filter as recognized in confidence scoring via b.filters.

    # Equality on text via "= 'x'" or "is 'x'" handled above (op "=").
    # Add IS NULL / IS NOT NULL.
    if "is null" in low:
        col = _find_last_column_in_text(low.split("is null")[0], b.schema)
        if col is not None and not any(
            f.column == col and f.operator == "IS NULL" for f in b.filters
        ):
            b.filters.append(Filter(column=col, operator="IS NULL", value=None))
            b.add_reason(f"filter {col} IS NULL")
    if "is not null" in low:
        col = _find_last_column_in_text(low.split("is not null")[0], b.schema)
        if col is not None and not any(
            f.column == col and f.operator == "IS NOT NULL" for f in b.filters
        ):
            b.filters.append(Filter(column=col, operator="IS NOT NULL", value=None))
            b.add_reason(f"filter {col} IS NOT NULL")

    # "contains 'x'" -> LIKE %x%
    contains_match = re.search(
        r"([a-z_][a-z0-9_]*)\s+contains\s+['\"]([^'\"]+)['\"]", low
    )
    if contains_match:
        col_tok, val = contains_match.group(1), contains_match.group(2)
        col = _match_column(col_tok, b.schema)
        if col is not None and b.schema.get(col) == "text":
            b.filters.append(
                Filter(column=col, operator="LIKE", value=f"%{val}%", logical="AND")
            )
            b.add_reason(f"filter {col} LIKE %{val}%")

    # "between A and B" — numeric/date only
    between = re.search(r"([a-z_][a-z0-9_]*)\s+between\s+(\S+)\s+and\s+(\S+)", low)
    if between:
        col_tok, lo, hi = between.group(1), between.group(2), between.group(3)
        col = _match_column(col_tok, b.schema)
        if col is not None and b.schema.get(col) in {"numeric", "date"}:
            lo_v = _coerce_value(lo, b.schema[col])
            hi_v = _coerce_value(hi, b.schema[col])
            if lo_v is not None and hi_v is not None:
                b.filters.append(
                    Filter(column=col, operator="BETWEEN", value=(lo_v, hi_v))
                )
                b.add_reason(f"filter {col} BETWEEN {lo_v} AND {hi_v}")


def _find_last_column_in_text(text: str, schema: Dict[str, str]) -> Optional[str]:
    """Last column-like token appearing in ``text``."""
    best: Optional[str] = None
    for tok in re.findall(r"[a-z_][a-z0-9_]*", text):
        col = _match_column(tok, schema)
        if col is not None:
            best = col
    return best


def _find_first_value_in_text(text: str, dtype: Optional[str]) -> Optional[object]:
    """First numeric or quoted text value appearing in ``text``."""
    if dtype == "numeric":
        m = _NUMERIC_RE.search(text)
        if m:
            return _coerce_value(m.group(0), "numeric")
        return None
    if dtype == "date":
        m = re.search(r"\d{4}-\d{2}-\d{2}", text)
        if m:
            return m.group(0)
        return None
    # Text: prefer quoted literal, else first bareword
    m = re.search(r"['\"]([^'\"]+)['\"]", text)
    if m:
        return m.group(1)
    m2 = re.search(r"\b([A-Za-z][A-Za-z0-9_-]+)\b", text)
    if m2:
        word = m2.group(1)
        if word.lower() in _STOPWORDS:
            return None
        return word
    return None


def _coerce_value(raw: str, dtype: str) -> Optional[object]:
    if dtype == "numeric":
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            return None
    if dtype == "date":
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            return raw
        return None
    return raw.strip(" '\"")


# --- date windows ---------------------------------------------------------


def _detect_date_windows(nl: str, b: _Builder) -> None:
    """Translate phrases like ``last 30 days`` into a date-range filter."""
    low = nl.lower()
    date_col = next((c for c, t in b.schema.items() if t == "date"), None)

    last_n = re.search(r"(?:last|past)\s+(\d+)\s+days?", low)
    if last_n and date_col is not None:
        days = int(last_n.group(1))
        today = date.today()
        start = (today - timedelta(days=days - 1)).isoformat()
        end_excl = (today + timedelta(days=1)).isoformat()
        b.filters.append(
            Filter(column=date_col, operator=">=", value=start, logical="AND")
        )
        b.filters.append(
            Filter(column=date_col, operator="<", value=end_excl, logical="AND")
        )
        b.add_reason(f"last {days} days on {date_col}")

    in_year = re.search(r"\bin\s+(\d{4})\b", low)
    if in_year and date_col is not None:
        year = int(in_year.group(1))
        b.filters.append(Filter(column=date_col, operator=">=", value=f"{year}-01-01"))
        b.filters.append(
            Filter(column=date_col, operator="<", value=f"{year + 1}-01-01")
        )
        b.add_reason(f"in year {year} on {date_col}")

    since = re.search(r"\bsince\s+(\d{4}-\d{2}-\d{2})\b", low)
    if since and date_col is not None:
        b.filters.append(Filter(column=date_col, operator=">=", value=since.group(1)))
        b.add_reason(f"since {since.group(1)} on {date_col}")


__all__ = ["HeuristicResult", "parse_heuristic", "HEURISTIC_FAST_PATH_THRESHOLD"]
