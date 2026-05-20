"""Deterministic insight extraction for the Assistant panel.

Given a result :class:`pandas.DataFrame` and the :class:`QueryModel`
that produced it, this module computes the structured cards (label /
value / delta) and the bold headline finding the design calls for. It
is **pure** — no LLM dependency, no Streamlit imports — so it can be
unit-tested with golden fixtures and reused on the offline heuristic
fast-path where no analysis prose is available otherwise.

Patterns supported
------------------

* **Single-group aggregation** — one group-by column + one numeric
  aggregation (``SUM/AVG/COUNT/MIN/MAX``). Emits TOP / LOWEST /
  CONCENTRATION cards and a "X leads by ~kx" headline.
* **Daily trend** — exactly one date bucket + one numeric aggregation.
  Emits FIRST / LAST / TREND cards plus a headline describing the
  signed percentage change.
* **Scalar count** — ``SELECT COUNT(*)`` style: one card.
* **Filtered simple SELECT** — emits a MATCH RATE card when
  ``source_row_count`` is provided.
* **Plain SELECT** — emits a single ROWS card.
* **Empty result** — single warning card, no headline.

Anything more exotic returns an empty analysis so the UI can fall back
to the LLM enrichment layer (Phase 4c) or no cards at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from src.query_model import QueryModel


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Insight:
    """One insight card.

    Slots map 1:1 to the mockup's ``.insight`` template:

    * ``label`` — uppercase header (rendered tiny + tracked).
    * ``value`` — short italic-serif headline value.
    * ``delta`` — monospace context line under the value.
    * ``direction`` — one of ``"up"``, ``"down"``, ``"neutral"``; drives
      the delta line's accent color (green/red/grey).
    """

    label: str
    value: str
    delta: str
    direction: str = "neutral"  # "up" | "down" | "neutral"


@dataclass(frozen=True)
class Headline:
    """The bold one-sentence finding rendered above the insight cards."""

    text: str


@dataclass
class DeterministicAnalysis:
    """All the structured facts an Assistant turn needs.

    The LLM enrichment layer (Phase 4c) may set ``prose`` and extend
    ``next_questions`` but must not overwrite ``insights`` or ``headline``.
    """

    headline: Optional[Headline] = None
    insights: List[Insight] = field(default_factory=list)
    next_questions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    prose: Optional[str] = None  # LLM narrative written by Phase 4c
    pattern: str = "unknown"  # debugging aid: which branch fired

    @property
    def is_empty(self) -> bool:
        return (
            self.headline is None
            and not self.insights
            and not self.next_questions
            and not self.warnings
            and not self.prose
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_insights(
    df: pd.DataFrame,
    query_model: QueryModel,
    *,
    source_row_count: Optional[int] = None,
) -> DeterministicAnalysis:
    """Build a :class:`DeterministicAnalysis` from a result frame.

    ``df`` is the executed query result (already ordered by whatever
    ORDER BY clauses the model carries). ``query_model`` is the model
    that produced ``df``; we read ``group_by``, ``aggregations``,
    ``date_buckets``, and ``filters`` from it to decide which branch
    to take. ``source_row_count`` is the row count of the underlying
    dataset and enables the match-rate card on filtered queries.
    """
    if df is None or len(df) == 0:
        return DeterministicAnalysis(
            warnings=["Query returned no rows."],
            pattern="empty",
        )

    # Branch 1: daily trend (date bucket + aggregation).
    if (
        query_model.date_buckets
        and query_model.aggregations
        and len(query_model.group_by) == 1
    ):
        date_col = next(iter(query_model.date_buckets))
        if date_col in query_model.group_by:
            return _trend_analysis(df, query_model, date_col)

    # Branch 2: single-group aggregation.
    if (
        len(query_model.group_by) == 1
        and len(query_model.aggregations) == 1
        and not query_model.date_buckets
    ):
        return _single_group_aggregation(df, query_model)

    # Branch 3: scalar count (COUNT(*) with no group).
    if (
        not query_model.group_by
        and len(query_model.aggregations) == 1
        and query_model.aggregations[0].function == "COUNT"
        and query_model.aggregations[0].column == "*"
    ):
        return _scalar_count(df, query_model)

    # Branch 4: simple SELECT with optional filter.
    return _simple_select(df, query_model, source_row_count)


# ---------------------------------------------------------------------------
# Branch implementations
# ---------------------------------------------------------------------------


def _single_group_aggregation(
    df: pd.DataFrame, model: QueryModel
) -> DeterministicAnalysis:
    """TOP / LOWEST / CONCENTRATION cards + headline."""
    group_col = model.group_by[0]
    agg = model.aggregations[0]
    measure_col = _agg_output_column(agg, df)
    if measure_col is None or group_col not in df.columns:
        return DeterministicAnalysis(pattern="single_group_unparseable")

    series = pd.to_numeric(df[measure_col], errors="coerce").dropna()
    if series.empty:
        return DeterministicAnalysis(
            warnings=["No numeric values to summarise."],
            pattern="single_group_no_numeric",
        )

    # df is expected to be ordered DESC by the measure (the typical
    # SQL emitted by quick_queries / heuristic). Re-derive top/bottom
    # from the data rather than trusting position, since users may
    # have flipped the order.
    sorted_df = df.assign(_m=series.values).sort_values(
        "_m", ascending=False, kind="mergesort"
    )
    top_row = sorted_df.iloc[0]
    bottom_row = sorted_df.iloc[-1]
    top_val = float(top_row["_m"])
    bot_val = float(bottom_row["_m"])
    total = float(series.sum()) or 1e-12

    top_share = top_val / total
    ratio = (top_val / bot_val) if bot_val > 0 else float("inf")
    hhi = float(((series / total) ** 2).sum())

    insights = [
        Insight(
            label=f"TOP {group_col.upper()}",
            value=str(top_row[group_col]),
            delta=f"{_fmt_value(top_val, measure_col)} \u00b7 "
            f"{top_share * 100:.1f}% of total",
            direction="up",
        ),
        Insight(
            label="LOWEST",
            value=str(bottom_row[group_col]),
            delta=(
                f"{_fmt_value(bot_val, measure_col)} \u00b7 "
                f"-{ratio:.1f}\u00d7 vs top"
                if ratio != float("inf")
                else f"{_fmt_value(bot_val, measure_col)}"
            ),
            direction="down",
        ),
        Insight(
            label="CONCENTRATION",
            value=_hhi_label(hhi),
            delta=f"HHI {hhi:.2f} ({_hhi_qualifier(hhi)})",
            direction="neutral",
        ),
    ]

    headline_text: Optional[str]
    if ratio == float("inf") or ratio < 1.05:
        headline_text = None
    else:
        headline_text = (
            f"{top_row[group_col]} leads by \u2248{ratio:.1f}\u00d7."
        )

    return DeterministicAnalysis(
        headline=Headline(text=headline_text) if headline_text else None,
        insights=insights,
        next_questions=[
            f"Break {top_row[group_col]} down by another column",
            f"Show the bottom 5 {group_col}s",
        ],
        pattern="single_group_aggregation",
    )


def _trend_analysis(
    df: pd.DataFrame, model: QueryModel, date_col: str
) -> DeterministicAnalysis:
    """FIRST / LAST / TREND cards + signed-percent headline.

    Supports day, month, and year bucket grains.
    """
    agg = model.aggregations[0]
    measure_col = _agg_output_column(agg, df)
    grain = (model.date_buckets or {}).get(date_col, "day")
    bucket_alias = f"{date_col}_{grain}"
    date_label = bucket_alias if bucket_alias in df.columns else date_col
    if measure_col is None or date_label not in df.columns:
        return DeterministicAnalysis(pattern="trend_unparseable")

    series = pd.to_numeric(df[measure_col], errors="coerce")
    if series.dropna().empty:
        return DeterministicAnalysis(
            warnings=["No numeric values to summarise."],
            pattern="trend_no_numeric",
        )

    ordered = df.assign(_m=series.values).sort_values(
        date_label, kind="mergesort"
    )
    first_row = ordered.iloc[0]
    last_row = ordered.iloc[-1]
    first_val = float(first_row["_m"]) if pd.notna(first_row["_m"]) else None
    last_val = float(last_row["_m"]) if pd.notna(last_row["_m"]) else None

    _grain_labels = {"day": "DAY", "month": "MONTH", "year": "YEAR"}
    grain_label = _grain_labels.get(grain, grain.upper())
    _period_words = {"day": "days", "month": "months", "year": "years"}
    period_word = _period_words.get(grain, "periods")

    insights: List[Insight] = [
        Insight(
            label=f"FIRST {grain_label}",
            value=str(first_row[date_label]),
            delta=_fmt_value(first_val, measure_col)
            if first_val is not None
            else "n/a",
            direction="neutral",
        ),
        Insight(
            label=f"LAST {grain_label}",
            value=str(last_row[date_label]),
            delta=_fmt_value(last_val, measure_col)
            if last_val is not None
            else "n/a",
            direction="neutral",
        ),
    ]

    headline: Optional[Headline] = None
    if first_val is not None and last_val is not None and first_val != 0:
        pct = (last_val - first_val) / abs(first_val) * 100.0
        if abs(pct) < 1.0:
            trend_label = "Flat"
            verb = "held flat"
            direction = "neutral"
        elif pct > 0:
            trend_label = "Up"
            verb = "grew"
            direction = "up"
        else:
            trend_label = "Down"
            verb = "fell"
            direction = "down"
        n = len(ordered)
        insights.append(
            Insight(
                label="TREND",
                value=trend_label,
                delta=f"\u0394 {pct:+.1f}% over {n} {period_word}",
                direction=direction,
            )
        )
        headline = Headline(
            text=(
                f"{_pretty_measure(measure_col)} {verb} "
                f"{pct:+.1f}% from {first_row[date_label]} to "
                f"{last_row[date_label]}."
            )
        )

    # Suggest drilling deeper based on grain
    if grain == "month":
        follow_ups = [
            "Break down by year instead of month",
            "Which category drove growth each month?",
        ]
    elif grain == "year":
        follow_ups = [
            "Show quarterly breakdown instead",
            "Which region grew most year-over-year?",
        ]
    else:
        follow_ups = [
            "Bucket by month instead of day",
            "Compare this trend against the same period last year",
        ]

    return DeterministicAnalysis(
        headline=headline,
        insights=insights,
        next_questions=follow_ups,
        pattern=f"trend_{grain}",
    )


def _scalar_count(df: pd.DataFrame, model: QueryModel) -> DeterministicAnalysis:
    n = int(df.iloc[0, 0]) if len(df) else 0
    return DeterministicAnalysis(
        insights=[
            Insight(
                label="ROW COUNT",
                value=f"{n:,}",
                delta="records in current dataset",
                direction="neutral",
            )
        ],
        pattern="scalar_count",
    )


def _simple_select(
    df: pd.DataFrame,
    model: QueryModel,
    source_row_count: Optional[int],
) -> DeterministicAnalysis:
    insights: List[Insight] = [
        Insight(
            label="ROWS",
            value=f"{len(df):,}",
            delta=(
                f"{len(df.columns)} column{'s' if len(df.columns) != 1 else ''} \u00b7 "
                + ", ".join(str(c) for c in df.columns[:3])
                + ("\u2026" if len(df.columns) > 3 else "")
            ),
            direction="neutral",
        )
    ]
    if model.filters and source_row_count and source_row_count > 0:
        pct = len(df) / source_row_count * 100.0
        insights.append(
            Insight(
                label="MATCH RATE",
                value=f"{pct:.1f}%",
                delta=f"{len(df):,} of {source_row_count:,} rows",
                direction="up" if pct >= 50 else "down",
            )
        )
    return DeterministicAnalysis(
        insights=insights,
        pattern="simple_select",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agg_output_column(agg, df: pd.DataFrame) -> Optional[str]:
    """Resolve the column name an :class:`Aggregation` ends up writing.

    Falls back to the function-prefixed default alias used by
    ``Aggregation.display_name``, then a best-effort match by suffix.
    """
    candidates = []
    if agg.alias:
        candidates.append(agg.alias)
    candidates.append(agg.display_name)
    if agg.function.upper() == "COUNT" and agg.column == "*":
        candidates.append("COUNT(*)")
    for cand in candidates:
        if cand in df.columns:
            return cand
    # Last resort: any column starting with the function name (case-insensitive).
    fn_prefix = agg.function.lower().replace(" ", "_") + "_"
    for col in df.columns:
        if str(col).lower().startswith(fn_prefix):
            return str(col)
    return None


def _hhi_label(hhi: float) -> str:
    if hhi >= 0.25:
        return "High"
    if hhi >= 0.15:
        return "Moderate"
    return "Even"


def _hhi_qualifier(hhi: float) -> str:
    if hhi >= 0.25:
        return "skewed"
    if hhi >= 0.15:
        return "uneven"
    return "balanced"


_CURRENCY_HINTS = ("revenue", "sales", "amount", "total", "price", "cost", "value")
_PERCENT_HINTS = ("pct", "percent", "ratio", "share")


def _fmt_value(value: float, column_name: str) -> str:
    """Format a numeric measure for the delta line.

    The formatting is purely cosmetic — never used for computation.
    Heuristics: columns whose name hints at currency render with a
    ``$`` prefix; percentage-like columns get ``%`` suffix; everything
    else uses thousands-separated decimals.
    """
    if value is None:
        return "n/a"
    low = column_name.lower() if column_name else ""
    if any(hint in low for hint in _PERCENT_HINTS):
        return f"{value:.1f}%"
    if any(hint in low for hint in _CURRENCY_HINTS):
        if abs(value) >= 1000:
            return f"${value:,.0f}"
        return f"${value:,.2f}"
    if float(value).is_integer():
        return f"{int(value):,}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _pretty_measure(column_name: str) -> str:
    """Turn ``sum_revenue`` into ``Revenue`` for headline prose."""
    if not column_name:
        return "the measure"
    name = column_name
    for prefix in ("sum_", "avg_", "min_", "max_", "count_"):
        if name.lower().startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("_", " ").strip().capitalize() or "the measure"


__all__ = [
    "DeterministicAnalysis",
    "Headline",
    "Insight",
    "compute_insights",
]
