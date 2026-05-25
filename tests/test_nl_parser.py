"""Tests for the Phase 3 LLM JSON → QueryModel parser.

No live Ollama call — ``OllamaClient`` is replaced with a stub for the
end-to-end test. The happy-path parser tests just feed dicts in
directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import pytest

import src.llm.natural_language as nl_mod
from src.llm.natural_language import (
    LLMError,
    OllamaClient,
    RouteToPythonError,
    detect_python_route_reason,
    nl_to_query_model,
    parse_query_plan,
)
from src.query_model import Aggregation, Filter, QueryModel


SCHEMA: Dict[str, str] = {
    "id": "numeric",
    "name": "text",
    "amount": "numeric",
    "country": "text",
    "created_at": "date",
}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_parse_minimal() -> None:
    model = parse_query_plan({}, SCHEMA)
    assert isinstance(model, QueryModel)
    assert model.selected_columns == []
    assert model.filters == []
    assert model.group_by == []
    assert model.aggregations == []
    assert model.having == []
    assert model.order_by == []
    assert model.limit is None
    # And it round-trips to a valid SELECT *.
    sql = model.to_sql()
    assert sql.upper().startswith("SELECT *")


def test_parse_selected_columns() -> None:
    model = parse_query_plan(
        {"selected_columns": ["name", "amount"]}, SCHEMA
    )
    assert model.selected_columns == ["name", "amount"]


def test_parse_filter_happy_path() -> None:
    payload = {
        "filters": [
            {
                "column": "country",
                "operator": "=",
                "value": "FI",
                "logical": "AND",
            }
        ]
    }
    model = parse_query_plan(payload, SCHEMA)
    assert len(model.filters) == 1
    f = model.filters[0]
    assert isinstance(f, Filter)
    assert f.column == "country"
    assert f.operator == "="
    assert f.value == "FI"
    assert f.logical == "AND"


def test_parse_filter_between() -> None:
    model = parse_query_plan(
        {
            "filters": [
                {
                    "column": "amount",
                    "operator": "BETWEEN",
                    "value": [10, 100],
                }
            ]
        },
        SCHEMA,
    )
    f = model.filters[0]
    assert f.operator == "BETWEEN"
    assert f.value == (10, 100)


def test_parse_filter_is_null_no_value() -> None:
    model = parse_query_plan(
        {
            "filters": [
                {"column": "name", "operator": "IS NULL"}
            ]
        },
        SCHEMA,
    )
    assert model.filters[0].value is None


def test_parse_aggregation_count_star() -> None:
    model = parse_query_plan(
        {
            "aggregations": [
                {"function": "COUNT", "column": "*", "alias": "n"}
            ]
        },
        SCHEMA,
    )
    assert len(model.aggregations) == 1
    agg = model.aggregations[0]
    assert isinstance(agg, Aggregation)
    assert agg.function == "COUNT"
    assert agg.column == "*"
    assert agg.alias == "n"


def test_parse_full_grouped_query() -> None:
    payload = {
        "selected_columns": ["country"],
        "filters": [
            {"column": "amount", "operator": ">", "value": 0}
        ],
        "group_by": ["country"],
        "aggregations": [
            {"function": "SUM", "column": "amount", "alias": "total"},
            {"function": "COUNT", "column": "*"},
        ],
        "having": [
            {"column": "total", "operator": ">", "value": 100}
        ],
        "order_by": [["total", "DESC"]],
        "limit": 10,
    }
    model = parse_query_plan(payload, SCHEMA)
    sql = model.to_sql()
    assert "GROUP BY" in sql
    assert "HAVING" in sql
    assert "ORDER BY" in sql
    assert "LIMIT 10" in sql
    assert sql.upper().startswith("SELECT")


def test_parse_having_count_star_expression_maps_to_alias() -> None:
    model = parse_query_plan(
        {
            "group_by": ["country"],
            "aggregations": [{"function": "COUNT", "column": "*"}],
            "having": [{"column": "COUNT(*)", "operator": ">", "value": 10}],
        },
        SCHEMA,
    )
    assert model.having[0].column == "count_all"


def test_parse_having_count_star_expression_maps_to_alias_lowercase() -> None:
    model = parse_query_plan(
        {
            "group_by": ["country"],
            "aggregations": [{"function": "COUNT", "column": "*"}],
            "having": [{"column": "count(*)", "operator": ">", "value": 10}],
        },
        SCHEMA,
    )
    assert model.having[0].column == "count_all"


def test_parse_having_count_star_expression_with_spaces_maps_to_alias() -> None:
    model = parse_query_plan(
        {
            "group_by": ["country"],
            "aggregations": [{"function": "COUNT", "column": "*"}],
            "having": [{"column": " COUNT ( * ) ", "operator": ">", "value": 10}],
        },
        SCHEMA,
    )
    assert model.having[0].column == "count_all"


def test_parse_having_count_star_auto_adds_count_aggregation() -> None:
    model = parse_query_plan(
        {
            "group_by": ["country"],
            "having": [{"column": "COUNT(*)", "operator": ">", "value": 10}],
        },
        SCHEMA,
    )
    assert any(a.function == "COUNT" and a.column == "*" for a in model.aggregations)
    assert model.having[0].column == "count_all"


def test_parse_order_by_uses_agg_alias() -> None:
    model = parse_query_plan(
        {
            "aggregations": [
                {"function": "SUM", "column": "amount", "alias": "total"}
            ],
            "order_by": [["total", "DESC"]],
        },
        SCHEMA,
    )
    assert model.order_by == [("total", "DESC")]


def test_parse_limit_coercion() -> None:
    assert parse_query_plan({"limit": None}, SCHEMA).limit is None
    assert parse_query_plan({"limit": 0}, SCHEMA).limit is None
    assert parse_query_plan({"limit": 10}, SCHEMA).limit == 10


def test_parse_inline_aggregation_in_selected_columns() -> None:
    model = parse_query_plan(
        {"selected_columns": ["COUNT(id) AS event_count"]},
        SCHEMA,
    )
    assert model.selected_columns == []
    assert len(model.aggregations) == 1
    agg = model.aggregations[0]
    assert agg.function == "COUNT"
    assert agg.column == "id"
    assert agg.alias == "event_count"


def test_parse_date_buckets_day() -> None:
    model = parse_query_plan(
        {
            "selected_columns": ["created_at"],
            "group_by": ["created_at"],
            "aggregations": [{"function": "COUNT", "column": "id", "alias": "n"}],
            "date_buckets": {"created_at": "day"},
            "order_by": [["created_at", "ASC"]],
        },
        SCHEMA,
    )
    sql = model.to_sql()
    assert 'substr("created_at", 1, 10) AS "created_at_day"' in sql
    assert 'GROUP BY substr("created_at", 1, 10)' in sql


# ---------------------------------------------------------------------------
# Safety — the parser is the trust boundary
# ---------------------------------------------------------------------------


def test_parse_rejects_non_dict_payload() -> None:
    for bad in ([1, 2], "hello", 42, None):
        with pytest.raises(LLMError):
            parse_query_plan(bad, SCHEMA)  # type: ignore[arg-type]


def test_parse_rejects_unknown_selected_column() -> None:
    with pytest.raises(LLMError) as exc:
        parse_query_plan({"selected_columns": ["ghost"]}, SCHEMA)
    assert "ghost" in str(exc.value)


def test_parse_rejects_unknown_filter_column() -> None:
    with pytest.raises(LLMError):
        parse_query_plan(
            {"filters": [{"column": "ghost", "operator": "=", "value": "x"}]},
            SCHEMA,
        )


def test_parse_rejects_bad_operator_for_text_column() -> None:
    # ">" is not valid on text columns.
    with pytest.raises(LLMError) as exc:
        parse_query_plan(
            {"filters": [{"column": "name", "operator": ">", "value": "z"}]},
            SCHEMA,
        )
    assert "operator" in str(exc.value).lower()


def test_parse_rejects_bad_aggregation_function() -> None:
    with pytest.raises(LLMError):
        parse_query_plan(
            {"aggregations": [{"function": "SUMX", "column": "amount"}]},
            SCHEMA,
        )


def test_parse_rejects_star_outside_count() -> None:
    with pytest.raises(LLMError):
        parse_query_plan(
            {"aggregations": [{"function": "SUM", "column": "*"}]},
            SCHEMA,
        )


def test_parse_having_requires_group_by() -> None:
    with pytest.raises(LLMError):
        parse_query_plan(
            {"having": [{"column": "amount", "operator": ">", "value": 1}]},
            SCHEMA,
        )


def test_parse_rejects_unknown_order_by_column() -> None:
    with pytest.raises(LLMError):
        parse_query_plan({"order_by": [["ghost", "ASC"]]}, SCHEMA)


def test_parse_rejects_bad_order_direction() -> None:
    with pytest.raises(LLMError):
        parse_query_plan({"order_by": [["amount", "sideways"]]}, SCHEMA)


def test_parse_rejects_negative_limit() -> None:
    with pytest.raises(LLMError):
        parse_query_plan({"limit": -1}, SCHEMA)


def test_parse_rejects_non_scalar_filter_value() -> None:
    with pytest.raises(LLMError):
        parse_query_plan(
            {
                "filters": [
                    {"column": "amount", "operator": "=", "value": {"a": 1}}
                ]
            },
            SCHEMA,
        )


def test_parse_injection_attempt_in_value_round_trips_safely() -> None:
    """The parser accepts a hostile value — the DEFENSIVE step is that
    ``quote_value`` escapes it into a safe literal and
    ``_assert_select_only`` then passes."""
    model = parse_query_plan(
        {
            "filters": [
                {
                    "column": "name",
                    "operator": "=",
                    "value": "'; DROP TABLE data;--",
                }
            ]
        },
        SCHEMA,
    )
    sql = model.to_sql()
    # DROP must NOT appear as a bare token — it's inside a quoted literal.
    assert sql.upper().startswith("SELECT")
    # The escaped literal should contain doubled quotes.
    assert "''; DROP TABLE data;--'" in sql


# ---------------------------------------------------------------------------
# Injectable client — no network
# ---------------------------------------------------------------------------


class _StubClient:
    """Honours the ``OllamaClient.generate_json`` signature."""

    def __init__(self, response: Dict[str, Any]) -> None:
        self._response = response
        self.calls: list = []

    def generate_json(self, system: str, user: str) -> Dict[str, Any]:
        self.calls.append((system, user))
        return self._response


class _SequencedStubClient:
    def __init__(self, responses: list[Dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list = []

    def generate_json(self, system: str, user: str) -> Dict[str, Any]:
        self.calls.append((system, user))
        if not self._responses:
            return {}
        return self._responses.pop(0)


def test_nl_to_query_model_with_stub_client() -> None:
    stub = _StubClient(
        {
            "selected_columns": ["name"],
            "filters": [
                {"column": "country", "operator": "=", "value": "FI"}
            ],
            "limit": 5,
        }
    )
    model = nl_to_query_model(
        "show finnish names", SCHEMA, client=stub  # type: ignore[arg-type]
    )
    assert model.selected_columns == ["name"]
    assert model.filters[0].column == "country"
    assert model.limit == 5
    assert len(stub.calls) == 1
    # The stub received the schema listing and the user's request.
    _system, user_prompt = stub.calls[0]
    assert "country" in user_prompt
    assert "show finnish names" in user_prompt


def test_nl_to_query_model_rejects_empty_nl() -> None:
    with pytest.raises(LLMError):
        nl_to_query_model("   ", SCHEMA, client=_StubClient({}))  # type: ignore[arg-type]


def test_nl_to_query_model_rejects_empty_schema() -> None:
    with pytest.raises(LLMError):
        nl_to_query_model("any", {}, client=_StubClient({}))  # type: ignore[arg-type]


def test_nl_to_query_model_routes_percentile_to_python_path() -> None:
    stub = _StubClient({"selected_columns": ["name"]})
    with pytest.raises(RouteToPythonError) as exc:
        nl_to_query_model(
            "What is the 90th percentile of amount?",
            SCHEMA,
            client=stub,  # type: ignore[arg-type]
        )
    msg = str(exc.value).lower()
    assert "routed away from sql generation" in msg
    assert "why:" in msg
    assert "blocked in sql mode:" in msg
    assert "next best actions:" in msg
    assert "percentile" in msg
    assert len(stub.calls) == 0


def test_nl_to_query_model_routes_percentage_to_python_path() -> None:
    stub = _StubClient({"selected_columns": ["name"]})
    with pytest.raises(RouteToPythonError) as exc:
        nl_to_query_model(
            "For each station, what percentage of trips return to the same station?",
            SCHEMA,
            client=stub,  # type: ignore[arg-type]
        )
    msg = str(exc.value).lower()
    assert "blocked in sql mode:" in msg
    assert "next best actions:" in msg
    assert len(stub.calls) == 0


def test_nl_to_query_model_routes_same_station_compare_to_python_path() -> None:
    stub = _StubClient({"selected_columns": ["name"]})
    with pytest.raises(RouteToPythonError):
        nl_to_query_model(
            "How many rows have departure station id equals return station id?",
            SCHEMA,
            client=stub,  # type: ignore[arg-type]
        )
    assert len(stub.calls) == 0


def test_detect_python_route_reason_narrow_percentage_pattern_routes_when_ratio_of() -> None:
    reason = detect_python_route_reason("What percentage of rows are weather = rain?")
    assert reason == "percentage/rate"


def test_detect_python_route_reason_narrow_percentage_pattern_does_not_route_on_share_word() -> None:
    reason = detect_python_route_reason("Show share_count by country, highest first.")
    assert reason is None


def test_nl_to_query_model_routes_rolling_without_time_column_adds_preprocess_guidance() -> None:
    schema_no_time = {
        "id": "numeric",
        "station": "text",
        "amount": "numeric",
    }
    stub = _StubClient({"selected_columns": ["station"]})
    with pytest.raises(RouteToPythonError) as exc:
        nl_to_query_model(
            "Show 7-day rolling average of amount.",
            schema_no_time,
            client=stub,  # type: ignore[arg-type]
        )
    msg = str(exc.value).lower()
    assert "no queryable time column was detected" in msg
    assert "preprocessing" in msg
    assert len(stub.calls) == 0


def test_nl_to_query_model_normalizes_last_7_days_inclusive_window(monkeypatch) -> None:
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(nl_mod, "datetime", _FixedDateTime)
    stub = _StubClient(
        {
            "selected_columns": ["created_at"],
            "group_by": ["created_at"],
            "aggregations": [{"function": "COUNT", "column": "id", "alias": "n"}],
            "filters": [{"column": "created_at", "operator": ">=", "value": "date_7_days_ago"}],
            "date_buckets": {"created_at": "day"},
            "order_by": [["created_at", "ASC"]],
        }
    )
    model = nl_to_query_model(
        "How many rows per day for the last 7 days?",
        SCHEMA,
        client=stub,  # type: ignore[arg-type]
    )
    lower = next((f for f in model.filters if f.column == "created_at" and f.operator == ">="), None)
    upper = next((f for f in model.filters if f.column == "created_at" and f.operator == "<"), None)
    assert lower is not None
    assert upper is not None
    assert lower.value == "2026-04-04"
    assert upper.value == "2026-04-11"


def test_nl_to_query_model_drops_non_grouped_selected_columns_when_aggregating() -> None:
    stub = _StubClient(
        {
            "selected_columns": ["country", "amount"],
            "group_by": ["country"],
            "aggregations": [{"function": "AVG", "column": "amount", "alias": "avg_amount"}],
            "reply": "Average amount by country.",
        }
    )
    model = nl_to_query_model(
        "Average amount by country",
        SCHEMA,
        client=stub,  # type: ignore[arg-type]
    )
    assert model.selected_columns == ["country"]
    sql = model.to_sql()
    assert "GROUP BY" in sql


def test_nl_to_query_model_retries_schema_error_once() -> None:
    client = _SequencedStubClient(
        [
            {"selected_columns": ["ghost_col"]},
            {"selected_columns": ["name"], "limit": 3},
        ]
    )
    model = nl_to_query_model("show names", SCHEMA, client=client)  # type: ignore[arg-type]
    assert model.selected_columns == ["name"]
    assert model.limit == 3
    assert len(client.calls) == 2


def test_ollama_client_constructs_without_network() -> None:
    # Construction must not touch the network.
    client = OllamaClient(host="http://localhost:11434", model="gemma3", timeout=5.0)
    assert client.host == "http://localhost:11434"
    assert client.model == "gemma3"
    assert client.timeout == 5.0
