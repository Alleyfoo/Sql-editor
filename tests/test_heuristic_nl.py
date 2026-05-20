"""Coverage for the offline heuristic NL parser."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.executor import execute
from src.heuristic_nl import HEURISTIC_FAST_PATH_THRESHOLD, parse_heuristic
from src.ingestion import _make_readonly
from src.streamlit_app.demo_dataset import load_demo


# Schema mirroring the demo dataset; pulled once for all tests via fixture.
@pytest.fixture(scope="module")
def demo():
    conn, schema, df, meta = load_demo()
    yield {"conn": conn, "schema": schema, "df": df}
    try:
        conn.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------


def test_sum_revenue_by_region(demo):
    res = parse_heuristic("sum revenue by region", demo["schema"])
    assert res.parsed
    assert any(
        a.function == "SUM" and a.column == "revenue" for a in res.model.aggregations
    )
    assert "region" in res.model.group_by
    sql = res.model.to_sql()
    assert 'SUM("revenue")' in sql
    assert 'GROUP BY "region"' in sql
    out = execute(demo["conn"], sql)
    assert len(out) == 3  # EMEA / AMER / APAC


def test_top_5_products_by_revenue(demo):
    res = parse_heuristic("top 5 products by revenue", demo["schema"])
    assert res.parsed
    assert res.model.limit == 5
    # The group key should be the entity column only (product); the
    # measure column (revenue) belongs to the aggregation, not the
    # GROUP BY clause.
    assert res.model.group_by == ["product"], res.model.group_by
    assert "revenue" not in res.model.group_by
    assert "revenue" not in res.model.selected_columns
    assert any(
        a.function == "SUM" and a.column == "revenue" for a in res.model.aggregations
    )
    # ORDER BY should be DESC on the SUM alias.
    assert res.model.order_by and res.model.order_by[0][1] == "DESC"
    out = execute(demo["conn"], res.model.to_sql())
    assert 1 <= len(out) <= 5


def test_average_unit_price_per_category(demo):
    res = parse_heuristic("average unit_price per category", demo["schema"])
    assert res.parsed
    assert any(
        a.function == "AVG" and a.column == "unit_price" for a in res.model.aggregations
    )
    assert "category" in res.model.group_by


def test_count_rows_scalar(demo):
    res = parse_heuristic("count rows", demo["schema"])
    assert res.parsed
    assert any(
        a.function == "COUNT" and a.column == "*" for a in res.model.aggregations
    )
    assert not res.model.group_by
    out = execute(demo["conn"], res.model.to_sql())
    assert len(out) == 1
    assert int(out.iloc[0][out.columns[0]]) == 3000


def test_how_many_rows_scalar(demo):
    res = parse_heuristic("how many rows are there", demo["schema"])
    assert res.parsed
    assert any(
        a.function == "COUNT" and a.column == "*" for a in res.model.aggregations
    )


def test_revenue_greater_than_500(demo):
    res = parse_heuristic("orders where revenue greater than 500", demo["schema"])
    assert res.parsed
    assert any(
        f.column == "revenue" and f.operator == ">" and f.value == 500
        for f in res.model.filters
    )
    out = execute(demo["conn"], res.model.to_sql())
    assert (out["revenue"] > 500).all()


def test_revenue_above_500(demo):
    """Synonym for >."""
    res = parse_heuristic("revenue above 500", demo["schema"])
    assert res.parsed
    assert any(
        f.column == "revenue" and f.operator == ">" and f.value == 500
        for f in res.model.filters
    )


def test_units_at_least_10(demo):
    res = parse_heuristic("orders with units at least 10", demo["schema"])
    assert res.parsed
    assert any(
        f.column == "units" and f.operator == ">=" and f.value == 10
        for f in res.model.filters
    )


def test_revenue_between(demo):
    res = parse_heuristic("revenue between 100 and 500", demo["schema"])
    assert res.parsed
    assert any(
        f.column == "revenue" and f.operator == "BETWEEN" and f.value == (100, 500)
        for f in res.model.filters
    )


def test_last_30_days_window(demo):
    res = parse_heuristic("orders in the last 30 days", demo["schema"])
    assert res.parsed
    # Both ends of the window are added as separate filters.
    date_filters = [f for f in res.model.filters if f.column == "order_date"]
    assert len(date_filters) >= 2


def test_show_columns(demo):
    res = parse_heuristic("show region and country", demo["schema"])
    assert res.parsed
    assert "region" in res.model.selected_columns
    assert "country" in res.model.selected_columns


def test_sort_by_column_desc(demo):
    res = parse_heuristic("show revenue sorted by revenue desc", demo["schema"])
    assert res.parsed
    assert ("revenue", "DESC") in res.model.order_by


def test_returns_by_category_fuzzy(demo):
    """Demonstrates substring column match: 'returns' -> 'is_returned'."""
    res = parse_heuristic("sum is_returned by category", demo["schema"])
    assert res.parsed
    assert any(
        a.function == "SUM" and a.column == "is_returned"
        for a in res.model.aggregations
    )
    assert "category" in res.model.group_by


# ---------------------------------------------------------------------------
# Refusal: low-confidence / nonsense
# ---------------------------------------------------------------------------


def test_gibberish_returns_none(demo):
    res = parse_heuristic("asdf qwerty foo bar", demo["schema"])
    assert not res.parsed
    assert res.model is None
    assert res.confidence < 0.35


def test_empty_input(demo):
    res = parse_heuristic("", demo["schema"])
    assert not res.parsed
    assert res.confidence == 0.0


def test_empty_schema():
    res = parse_heuristic("sum revenue by region", {})
    assert not res.parsed


# ---------------------------------------------------------------------------
# Safety: every parsed model still passes the SELECT-only validator and
# the executor's read-only authorizer.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 3 fast-path threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "sum revenue by region",
        "top 5 products by revenue",
        "average unit_price per category",
        "sum revenue by region where revenue greater than 500",
    ],
)
def test_high_confidence_queries_clear_fast_path_threshold(demo, query):
    """These canonical patterns should bypass the LLM in the UI."""
    res = parse_heuristic(query, demo["schema"])
    assert res.parsed
    assert res.confidence >= HEURISTIC_FAST_PATH_THRESHOLD, (
        f"{query!r} scored {res.confidence:.2f}, expected >= "
        f"{HEURISTIC_FAST_PATH_THRESHOLD:.2f}"
    )


@pytest.mark.parametrize(
    "query",
    [
        # Single-intent ambiguous prompts: should defer to the LLM.
        "show product",
        "revenue",
    ],
)
def test_marginal_queries_defer_to_llm(demo, query):
    """Prompts with at most one intent must stay below the fast-path
    threshold so they route through the LLM when it is reachable."""
    res = parse_heuristic(query, demo["schema"])
    assert res.confidence < HEURISTIC_FAST_PATH_THRESHOLD, (
        f"{query!r} scored {res.confidence:.2f}, expected < "
        f"{HEURISTIC_FAST_PATH_THRESHOLD:.2f}"
    )


def test_parsed_models_remain_select_only(demo):
    queries = [
        "sum revenue by region",
        "top 5 products by revenue",
        "average unit_price per category",
        "count rows",
        "orders where revenue greater than 500",
        "revenue between 100 and 500",
    ]
    for q in queries:
        res = parse_heuristic(q, demo["schema"])
        assert res.parsed, q
        sql = res.model.to_sql()
        assert sql.lstrip().upper().startswith("SELECT"), q
        # Read-only connection accepts it.
        df = execute(demo["conn"], sql)
        assert isinstance(df, pd.DataFrame)
