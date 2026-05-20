"""Golden tests for the deterministic insight engine."""

from __future__ import annotations

import pandas as pd
import pytest

from src.query_model import Aggregation, Filter, QueryModel
from src.streamlit_app.insight_engine import (
    DeterministicAnalysis,
    Headline,
    Insight,
    compute_insights,
)


# ---------------------------------------------------------------------------
# Single-group aggregation
# ---------------------------------------------------------------------------


def _grouped_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["Electronics", "Apparel", "Stationery"],
            "sum_revenue": [24_488.0, 13_212.0, 7_251.0],
        }
    )


def _grouped_model() -> QueryModel:
    return QueryModel(
        table="data",
        selected_columns=["category"],
        group_by=["category"],
        aggregations=[Aggregation(column="revenue", function="SUM", alias="sum_revenue")],
        order_by=[("sum_revenue", "DESC")],
    )


def test_single_group_aggregation_top_lowest_concentration():
    res = compute_insights(_grouped_df(), _grouped_model())
    assert res.pattern == "single_group_aggregation"
    assert len(res.insights) == 3

    top, low, conc = res.insights
    assert top.label == "TOP CATEGORY"
    assert top.value == "Electronics"
    assert "$24,488" in top.delta
    assert "% of total" in top.delta
    assert top.direction == "up"

    assert low.label == "LOWEST"
    assert low.value == "Stationery"
    assert "vs top" in low.delta
    assert low.direction == "down"

    assert conc.label == "CONCENTRATION"
    assert conc.value in {"High", "Moderate", "Even"}
    assert "HHI" in conc.delta


def test_single_group_aggregation_headline_has_ratio():
    res = compute_insights(_grouped_df(), _grouped_model())
    assert res.headline is not None
    assert "Electronics leads" in res.headline.text
    # Ratio is roughly 24488 / 7251 ≈ 3.38, but the formatting rounds to one
    # decimal — anywhere between 3.3 and 3.4 is acceptable.
    assert any(token in res.headline.text for token in ("3.3", "3.4"))


def test_single_group_aggregation_no_headline_when_flat():
    """If the top and bottom values are within 5%, suppress the headline."""
    df = pd.DataFrame(
        {"category": ["A", "B", "C"], "sum_revenue": [100.0, 99.0, 98.0]}
    )
    res = compute_insights(df, _grouped_model())
    assert res.pattern == "single_group_aggregation"
    assert res.headline is None
    assert len(res.insights) == 3


def test_single_group_with_count_alias_resolves_column():
    df = pd.DataFrame({"region": ["AMER", "EMEA"], "count_orders": [120, 30]})
    model = QueryModel(
        table="data",
        selected_columns=["region"],
        group_by=["region"],
        aggregations=[
            Aggregation(column="order_id", function="COUNT", alias="count_orders")
        ],
        order_by=[("count_orders", "DESC")],
    )
    res = compute_insights(df, model)
    assert res.pattern == "single_group_aggregation"
    assert res.insights[0].value == "AMER"


# ---------------------------------------------------------------------------
# Trend (date bucket + measure)
# ---------------------------------------------------------------------------


def _trend_df_growth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_date_day": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
            ],
            "sum_revenue": [100.0, 110.0, 130.0, 140.0, 150.0],
        }
    )


def _trend_model() -> QueryModel:
    return QueryModel(
        table="data",
        selected_columns=["order_date"],
        group_by=["order_date"],
        aggregations=[Aggregation(column="revenue", function="SUM", alias="sum_revenue")],
        order_by=[("order_date", "ASC")],
        date_buckets={"order_date": "day"},
    )


def test_trend_growth_emits_up_direction_and_positive_headline():
    res = compute_insights(_trend_df_growth(), _trend_model())
    assert res.pattern == "trend_day"
    assert len(res.insights) == 3
    first, last, trend = res.insights
    assert first.label == "FIRST DAY"
    assert first.value == "2025-01-01"
    assert last.label == "LAST DAY"
    assert last.value == "2025-01-05"
    assert trend.label == "TREND"
    assert trend.value == "Up"
    assert trend.direction == "up"
    assert "+50" in trend.delta or "+50.0" in trend.delta
    assert res.headline is not None
    assert "grew" in res.headline.text


def test_trend_decline_emits_down_direction():
    df = _trend_df_growth().assign(sum_revenue=[150.0, 130.0, 110.0, 90.0, 70.0])
    res = compute_insights(df, _trend_model())
    trend = res.insights[2]
    assert trend.value == "Down"
    assert trend.direction == "down"
    assert "fell" in (res.headline.text if res.headline else "")


def test_trend_flat_emits_neutral():
    df = _trend_df_growth().assign(sum_revenue=[100.0, 100.5, 100.0, 99.5, 100.0])
    res = compute_insights(df, _trend_model())
    trend = res.insights[2]
    assert trend.value == "Flat"
    assert trend.direction == "neutral"


# ---------------------------------------------------------------------------
# Scalar count
# ---------------------------------------------------------------------------


def test_scalar_count():
    df = pd.DataFrame({"row_count": [150]})
    model = QueryModel(
        table="data",
        aggregations=[Aggregation(column="*", function="COUNT", alias="row_count")],
    )
    res = compute_insights(df, model)
    assert res.pattern == "scalar_count"
    assert len(res.insights) == 1
    card = res.insights[0]
    assert card.label == "ROW COUNT"
    assert card.value == "150"


# ---------------------------------------------------------------------------
# Simple SELECT + filter match rate
# ---------------------------------------------------------------------------


def test_simple_select_rows_card_only():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    model = QueryModel(table="data", selected_columns=["a", "b"])
    res = compute_insights(df, model)
    assert res.pattern == "simple_select"
    assert len(res.insights) == 1
    assert res.insights[0].label == "ROWS"
    assert res.insights[0].value == "3"


def test_simple_select_with_filter_adds_match_rate():
    df = pd.DataFrame({"a": [1, 2, 3]})
    model = QueryModel(
        table="data",
        selected_columns=["a"],
        filters=[Filter(column="a", operator=">", value=0)],
    )
    res = compute_insights(df, model, source_row_count=300)
    assert any(i.label == "MATCH RATE" for i in res.insights)
    match = next(i for i in res.insights if i.label == "MATCH RATE")
    assert "1.0%" in match.value or "1%" in match.value
    assert "of 300" in match.delta


# ---------------------------------------------------------------------------
# Empty result
# ---------------------------------------------------------------------------


def test_empty_dataframe_returns_warning_only():
    res = compute_insights(pd.DataFrame(), QueryModel(table="data"))
    assert res.pattern == "empty"
    assert res.warnings == ["Query returned no rows."]
    assert not res.insights
    assert res.headline is None


def test_none_dataframe_returns_warning():
    res = compute_insights(None, QueryModel(table="data"))  # type: ignore[arg-type]
    assert res.pattern == "empty"


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_single_group_with_all_zero_measure_skips_headline():
    df = pd.DataFrame({"category": ["A", "B"], "sum_revenue": [0.0, 0.0]})
    res = compute_insights(df, _grouped_model())
    # Ratio is infinite -> no headline, lowest delta drops the "vs top" suffix
    assert res.headline is None
