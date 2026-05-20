"""Schema-aware quick-query templates for the Streamlit app.

Each template inspects the active dataset's schema and returns a fully
formed ``QueryModel`` with a ``reply`` describing what it does. They run
entirely offline (no LLM) and feed the same ``to_sql()`` + executor
pipeline as any other query, so the read-only safety guarantees still
apply.

These power the "Quick queries · runs offline" strip in the Ask
component and the always-on fallback when Ollama is unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.ingestion import TABLE_NAME
from src.query_model import Aggregation, QueryModel


@dataclass(frozen=True)
class QuickQuery:
    """A pre-canned, schema-aware query template."""

    key: str  # stable id used for Streamlit widget keys
    label: str  # shown on the button
    description: str  # shown as tooltip / caption
    build: Callable[[Dict[str, str]], QueryModel]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _columns_by_type(schema: Dict[str, str], dtype: str) -> List[str]:
    return [col for col, t in schema.items() if t == dtype]


def _first(items: List[str]) -> Optional[str]:
    return items[0] if items else None


# Substrings that mark a column as the *headline* measure of a dataset
# (revenue, sales, totals). Preferred over secondary measures on ties.
_HEADLINE_MEASURE_HINTS = (
    "revenue",
    "sales",
    "amount",
    "total",
    "value",
    "balance",
)

# Other plausible measures (counts, prices, ratings, etc.). Still much
# better defaults than IDs but should yield to the headline group.
_SECONDARY_MEASURE_HINTS = (
    "price",
    "cost",
    "units",
    "qty",
    "quantity",
    "count",
    "score",
    "rating",
    "duration",
)

# Substrings that suggest a column is an identifier rather than a measure.
# These get pushed to the back of the list so we don't default to summing
# primary keys.
_ID_HINTS = ("_id", "id_", "uuid", "guid", "code", "number")


_CUSTOMER_HINTS = ("customer_id", "customer", "client_id", "client", "user_id", "account_id")
_MARGIN_HINTS = ("margin", "profit", "gross_margin", "net_margin")
_STATUS_HINTS = ("status", "state", "order_status", "stage")

# Text columns that are good GROUP BY candidates (prefer these over raw IDs)
_CATEGORY_HINTS = (
    "category", "region", "segment", "type", "country",
    "product", "department", "team", "group", "channel",
)
_ID_TEXT_HINTS = ("_id", "id_", "uuid", "guid", "code", "number")


def _find_customer_column(schema: Dict[str, str]) -> Optional[str]:
    """Return the most likely customer-identifier column, or None."""
    for hint in _CUSTOMER_HINTS:
        for col in schema:
            if hint in col.lower():
                return col
    return None


def _find_margin_column(schema: Dict[str, str]) -> Optional[str]:
    """Return the most likely margin/profit column, or None."""
    for hint in _MARGIN_HINTS:
        for col in schema:
            if hint in col.lower():
                return col
    return None


def _find_status_column(schema: Dict[str, str]) -> Optional[str]:
    """Return the most likely status/state column, or None."""
    for hint in _STATUS_HINTS:
        for col, dtype in schema.items():
            if hint in col.lower() and dtype == "text":
                return col
    return None


def _pick_text_category(schema: Dict[str, str]) -> Optional[str]:
    """Return the best GROUP BY text column — prefers category/region/segment
    over raw IDs like customer_id."""
    text_cols = _columns_by_type(schema, "text")
    if not text_cols:
        return None
    # Score: 0 = category hint, 1 = neutral, 2 = id-like
    def _rank(col: str) -> int:
        low = col.lower()
        if any(h in low for h in _CATEGORY_HINTS):
            return 0
        if low == "id" or any(h in low for h in _ID_TEXT_HINTS):
            return 2
        return 1
    return sorted(text_cols, key=_rank)[0]


def _rank_numeric(col: str) -> tuple:
    """Sort key: lower = better default measure.

    Buckets (lowest wins):
        0 = headline measure (revenue/sales/total/...)
        1 = secondary measure (price/units/count/...)
        2 = neutral
        3 = identifier-looking
    """
    low = col.lower()
    if any(hint in low for hint in _HEADLINE_MEASURE_HINTS):
        return (0,)
    if any(hint in low for hint in _SECONDARY_MEASURE_HINTS):
        return (1,)
    if low == "id" or any(hint in low for hint in _ID_HINTS):
        return (3,)
    return (2,)


def _pick_numeric(schema: Dict[str, str]) -> Optional[str]:
    """Return the most useful numeric column for aggregation, or None."""
    cols = _columns_by_type(schema, "numeric")
    if not cols:
        return None
    return sorted(cols, key=_rank_numeric)[0]


# ---------------------------------------------------------------------------
# Builders (one per template). Each returns a QueryModel. The dispatcher in
# ``build_quick_queries`` only includes a template if it makes sense for
# the active schema.
# ---------------------------------------------------------------------------


def _build_show_first_100(schema: Dict[str, str]) -> QueryModel:
    return QueryModel(
        table=TABLE_NAME,
        selected_columns=list(schema.keys()),
        limit=100,
        reply="Showing the first 100 rows.",
    )


def _build_count_rows(_schema: Dict[str, str]) -> QueryModel:
    return QueryModel(
        table=TABLE_NAME,
        aggregations=[Aggregation(column="*", function="COUNT", alias="row_count")],
        reply="Counting all rows.",
    )


def _build_top_n_text_by_numeric(schema: Dict[str, str], *, n: int = 10) -> QueryModel:
    text_col = _pick_text_category(schema)
    num_col = _pick_numeric(schema)
    if text_col is None or num_col is None:
        raise ValueError("requires at least one text and one numeric column")
    alias = f"sum_{num_col}"
    return QueryModel(
        table=TABLE_NAME,
        selected_columns=[text_col],
        group_by=[text_col],
        aggregations=[Aggregation(column=num_col, function="SUM", alias=alias)],
        order_by=[(alias, "DESC")],
        limit=n,
        reply=f"Top {n} {text_col} by total {num_col}.",
    )


def _build_sum_numeric_by_text(schema: Dict[str, str]) -> QueryModel:
    text_col = _pick_text_category(schema)
    num_col = _pick_numeric(schema)
    if text_col is None or num_col is None:
        raise ValueError("requires at least one text and one numeric column")
    alias = f"sum_{num_col}"
    return QueryModel(
        table=TABLE_NAME,
        selected_columns=[text_col],
        group_by=[text_col],
        aggregations=[Aggregation(column=num_col, function="SUM", alias=alias)],
        order_by=[(alias, "DESC")],
        reply=f"Total {num_col} grouped by {text_col}.",
    )


def _build_average_numeric_by_text(schema: Dict[str, str]) -> QueryModel:
    text_col = _pick_text_category(schema)
    num_col = _pick_numeric(schema)
    if text_col is None or num_col is None:
        raise ValueError("requires at least one text and one numeric column")
    alias = f"avg_{num_col}"
    return QueryModel(
        table=TABLE_NAME,
        selected_columns=[text_col],
        group_by=[text_col],
        aggregations=[Aggregation(column=num_col, function="AVG", alias=alias)],
        order_by=[(alias, "DESC")],
        reply=f"Average {num_col} grouped by {text_col}.",
    )


def _build_daily_trend(schema: Dict[str, str]) -> QueryModel:
    date_col = _first(_columns_by_type(schema, "date"))
    num_col = _pick_numeric(schema)
    if date_col is None or num_col is None:
        raise ValueError("requires at least one date and one numeric column")
    alias = f"sum_{num_col}"
    return QueryModel(
        table=TABLE_NAME,
        selected_columns=[date_col],
        group_by=[date_col],
        aggregations=[Aggregation(column=num_col, function="SUM", alias=alias)],
        order_by=[(date_col, "ASC")],
        date_buckets={date_col: "day"},
        reply=f"Daily trend of total {num_col}.",
    )


def _build_monthly_trend(schema: Dict[str, str]) -> QueryModel:
    date_col = _first(_columns_by_type(schema, "date"))
    num_col = _pick_numeric(schema)
    if date_col is None or num_col is None:
        raise ValueError("requires at least one date and one numeric column")
    alias = f"sum_{num_col}"
    return QueryModel(
        table=TABLE_NAME,
        selected_columns=[date_col],
        group_by=[date_col],
        aggregations=[Aggregation(column=num_col, function="SUM", alias=alias)],
        order_by=[(date_col, "ASC")],
        date_buckets={date_col: "month"},
        reply=f"Monthly trend of total {num_col}.",
    )


def _build_top_customers(schema: Dict[str, str], *, n: int = 10) -> QueryModel:
    """Top N customers by revenue — requires a customer_id-like column."""
    cust_col = _find_customer_column(schema)
    num_col = _pick_numeric(schema)
    if cust_col is None or num_col is None:
        raise ValueError("requires a customer column and a numeric column")
    alias = f"sum_{num_col}"
    return QueryModel(
        table=TABLE_NAME,
        selected_columns=[cust_col],
        group_by=[cust_col],
        aggregations=[Aggregation(column=num_col, function="SUM", alias=alias)],
        order_by=[(alias, "DESC")],
        limit=n,
        reply=f"Top {n} customers by total {num_col}.",
    )


def _build_margin_by_category(schema: Dict[str, str]) -> QueryModel:
    """SUM(margin) grouped by the best categorical text column."""
    margin_col = _find_margin_column(schema)
    text_col = _pick_text_category(schema)
    if margin_col is None or text_col is None:
        raise ValueError("requires a margin column and a text column")
    alias = f"sum_{margin_col}"
    return QueryModel(
        table=TABLE_NAME,
        selected_columns=[text_col],
        group_by=[text_col],
        aggregations=[Aggregation(column=margin_col, function="SUM", alias=alias)],
        order_by=[(alias, "DESC")],
        reply=f"Total {margin_col} grouped by {text_col}.",
    )


def _build_status_breakdown(schema: Dict[str, str]) -> QueryModel:
    """COUNT(*) grouped by status — great CASE WHEN / GROUP BY showcase."""
    status_col = _find_status_column(schema)
    if status_col is None:
        raise ValueError("requires a status-like column")
    return QueryModel(
        table=TABLE_NAME,
        selected_columns=[status_col],
        group_by=[status_col],
        aggregations=[Aggregation(column="*", function="COUNT", alias="order_count")],
        order_by=[("order_count", "DESC")],
        reply=f"Order count by {status_col}.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_quick_queries(schema: Dict[str, str]) -> List[QuickQuery]:
    """Return the list of quick-query templates valid for ``schema``.

    Templates that need column types not present in the schema are
    silently omitted so the UI doesn't show buttons that would fail.
    """
    if not isinstance(schema, dict) or not schema:
        return []

    text_cols = _columns_by_type(schema, "text")
    num_cols = _columns_by_type(schema, "numeric")
    date_cols = _columns_by_type(schema, "date")
    measure_col = _pick_numeric(schema)

    out: List[QuickQuery] = [
        QuickQuery(
            key="qq_show_first_100",
            label="Show first 100 rows",
            description="SELECT all columns, limit 100. Quick way to peek at the data.",
            build=_build_show_first_100,
        ),
        QuickQuery(
            key="qq_count_rows",
            label="Count rows",
            description="SELECT COUNT(*) — total rows in the dataset.",
            build=_build_count_rows,
        ),
    ]

    cat_col = _pick_text_category(schema)
    if cat_col is not None and measure_col is not None:
        num_col = measure_col
        out.extend(
            [
                QuickQuery(
                    key="qq_top10_text_by_numeric",
                    label=f"Top 10 {cat_col} by {num_col}",
                    description=(
                        f"GROUP BY {cat_col}, SUM({num_col}), order DESC, limit 10."
                    ),
                    build=lambda s: _build_top_n_text_by_numeric(s, n=10),
                ),
                QuickQuery(
                    key="qq_sum_numeric_by_text",
                    label=f"Sum {num_col} by {cat_col}",
                    description=f"Total {num_col} grouped by {cat_col}.",
                    build=_build_sum_numeric_by_text,
                ),
                QuickQuery(
                    key="qq_avg_numeric_by_text",
                    label=f"Average {num_col} by {cat_col}",
                    description=f"Mean {num_col} grouped by {cat_col}.",
                    build=_build_average_numeric_by_text,
                ),
            ]
        )

    if date_cols and measure_col is not None:
        date_col = date_cols[0]
        num_col = measure_col
        out.extend(
            [
                QuickQuery(
                    key="qq_monthly_trend",
                    label=f"Monthly {num_col} trend",
                    description=(
                        f"Bucket {date_col} by month, SUM({num_col}), ordered by date. "
                        "Great for spotting seasonality and year-over-year growth."
                    ),
                    build=_build_monthly_trend,
                ),
                QuickQuery(
                    key="qq_daily_trend",
                    label=f"Daily {num_col} trend",
                    description=(
                        f"Bucket {date_col} by day, SUM({num_col}), ordered by date."
                    ),
                    build=_build_daily_trend,
                ),
            ]
        )

    cust_col = _find_customer_column(schema)
    if cust_col is not None and measure_col is not None:
        out.append(
            QuickQuery(
                key="qq_top_customers",
                label=f"Top 10 customers by {measure_col}",
                description=(
                    f"GROUP BY {cust_col}, SUM({measure_col}), order DESC, limit 10. "
                    "Identifies your highest-value accounts."
                ),
                build=lambda s: _build_top_customers(s, n=10),
            )
        )

    margin_col = _find_margin_column(schema)
    cat_for_margin = _pick_text_category(schema)
    if margin_col is not None and cat_for_margin is not None:
        out.append(
            QuickQuery(
                key="qq_margin_by_category",
                label=f"Margin by {cat_for_margin}",
                description=(
                    f"SUM({margin_col}) grouped by {cat_for_margin}. "
                    "Shows which segments are most profitable."
                ),
                build=_build_margin_by_category,
            )
        )

    status_col = _find_status_column(schema)
    if status_col is not None:
        out.append(
            QuickQuery(
                key="qq_status_breakdown",
                label=f"Orders by {status_col}",
                description=(
                    f"COUNT(*) GROUP BY {status_col}. "
                    "Shows distribution across order states."
                ),
                build=_build_status_breakdown,
            )
        )

    return out


__all__ = ["QuickQuery", "build_quick_queries"]
