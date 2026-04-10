"""Tests for conversation history in the NL → QueryModel pipeline.

All tests use the injectable stub client — no live Ollama call.
The test CSV lives at artifacts/test_sales.csv and has these columns:
  product (text), category (text), price (numeric),
  quantity (numeric), sale_date (date)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from src.llm.natural_language import (
    LLMConfig,
    build_user_prompt,
    nl_to_query_model,
    parse_query_plan,
)
from src.llm.natural_language import LLMError
from src.query_model import QueryModel


SALES_CSV = Path(__file__).parent.parent / "artifacts" / "test_sales.csv"

SCHEMA: Dict[str, str] = {
    "product": "text",
    "category": "text",
    "price": "numeric",
    "quantity": "numeric",
    "sale_date": "date",
}


class _StubClient:
    """Captures every (system, user) call and returns a preset response."""

    def __init__(self, response: Dict[str, Any]) -> None:
        self._response = response
        self.calls: list = []

    def generate_json(self, system: str, user: str) -> Dict[str, Any]:
        self.calls.append((system, user))
        return self._response


# ---------------------------------------------------------------------------
# build_user_prompt — history embedding
# ---------------------------------------------------------------------------


def test_build_prompt_no_history_has_no_conversation_block() -> None:
    prompt = build_user_prompt("show all products", SCHEMA)
    assert "Conversation so far" not in prompt


def test_build_prompt_with_history_includes_qa_pairs() -> None:
    history = [
        ("show all electronics", "Selecting all rows where category is Electronics."),
        ("filter price above 50", "Adding a filter for price > 50."),
    ]
    prompt = build_user_prompt("now sort by price", SCHEMA, history=history)
    assert "Conversation so far" in prompt
    assert "show all electronics" in prompt
    assert "Selecting all rows where category is Electronics." in prompt
    assert "filter price above 50" in prompt
    assert "Adding a filter for price > 50." in prompt


def test_build_prompt_empty_history_list_omits_block() -> None:
    prompt = build_user_prompt("show all products", SCHEMA, history=[])
    assert "Conversation so far" not in prompt


def test_build_prompt_history_appears_before_json_shape() -> None:
    history = [("show products", "Selecting all products.")]
    prompt = build_user_prompt("filter by category", SCHEMA, history=history)
    history_pos = prompt.index("Conversation so far")
    shape_pos = prompt.index("Produce a JSON object")
    assert history_pos < shape_pos


# ---------------------------------------------------------------------------
# nl_to_query_model — history passed through to client
# ---------------------------------------------------------------------------


def test_nl_to_query_model_passes_history_in_prompt() -> None:
    stub = _StubClient(
        {
            "reply": "Filtering Electronics with price > 50.",
            "selected_columns": ["product", "price"],
            "filters": [
                {"column": "price", "operator": ">", "value": 50, "logical": "AND"}
            ],
        }
    )
    history = [("show all rows", "Selecting all rows from the dataset.")]
    nl_to_query_model(
        "filter to electronics with price above 50",
        SCHEMA,
        client=stub,  # type: ignore[arg-type]
        history=history,
    )
    assert len(stub.calls) == 1
    _system, user_prompt = stub.calls[0]
    assert "Conversation so far" in user_prompt
    assert "show all rows" in user_prompt


def test_nl_to_query_model_no_history_omits_block() -> None:
    stub = _StubClient({"selected_columns": ["product"]})
    nl_to_query_model("show products", SCHEMA, client=stub)  # type: ignore[arg-type]
    _system, user_prompt = stub.calls[0]
    assert "Conversation so far" not in user_prompt


# ---------------------------------------------------------------------------
# History depth capping (simulation at the prompt level)
# ---------------------------------------------------------------------------


def test_only_last_n_history_turns_appear_in_prompt() -> None:
    """When the caller trims to depth N before calling build_user_prompt,
    only those N turns appear. Here we verify the trim logic that the UI
    uses: len(history) > depth → keep last depth entries."""
    all_turns = [(f"question {i}", f"answer {i}") for i in range(10)]
    depth = 3
    trimmed = all_turns[-depth:]

    prompt = build_user_prompt("new question", SCHEMA, history=trimmed)
    # Only last 3 turns present
    for i in range(7):
        assert f"question {i}" not in prompt
    for i in range(7, 10):
        assert f"question {i}" in prompt


# ---------------------------------------------------------------------------
# Multi-turn scenario: follow-up references prior context
# ---------------------------------------------------------------------------


def test_multi_turn_follow_up_has_prior_context_in_prompt() -> None:
    """Simulate two sequential ASK calls.

    Turn 1: user asks for Electronics → stub returns category filter.
    Turn 2: user says 'now also filter price above 100' →
            prompt must contain the Q/A from turn 1 so the model
            knows what 'also' refers to.
    """
    turn1_reply = "Filtering to rows where category is Electronics."
    stub1 = _StubClient(
        {
            "reply": turn1_reply,
            "filters": [
                {
                    "column": "category",
                    "operator": "=",
                    "value": "Electronics",
                    "logical": "AND",
                }
            ],
        }
    )
    model1 = nl_to_query_model(
        "show only Electronics",
        SCHEMA,
        client=stub1,  # type: ignore[arg-type]
    )
    assert model1.reply == turn1_reply

    # Simulate what the UI does: append to history after success
    history: list = [("show only Electronics", model1.reply)]

    stub2 = _StubClient(
        {
            "reply": "Keeping the Electronics filter and adding price > 100.",
            "filters": [
                {
                    "column": "category",
                    "operator": "=",
                    "value": "Electronics",
                    "logical": "AND",
                },
                {
                    "column": "price",
                    "operator": ">",
                    "value": 100,
                    "logical": "AND",
                },
            ],
        }
    )
    model2 = nl_to_query_model(
        "now also filter price above 100",
        SCHEMA,
        client=stub2,  # type: ignore[arg-type]
        history=history,
    )
    _system, user_prompt2 = stub2.calls[0]
    assert "show only Electronics" in user_prompt2
    assert turn1_reply in user_prompt2
    assert len(model2.filters) == 2


# ---------------------------------------------------------------------------
# column_formats — numeric formatting via ROUND()
# ---------------------------------------------------------------------------


def test_column_formats_appear_in_system_prompt() -> None:
    """The system prompt must mention formatting so the model knows to use it."""
    from src.llm.natural_language import SYSTEM_PROMPT

    assert "column_formats" in SYSTEM_PROMPT or "formatting" in SYSTEM_PROMPT.lower()


def test_build_prompt_includes_column_formats_in_json_shape() -> None:
    prompt = build_user_prompt("show prices rounded to 2 decimals", SCHEMA)
    assert "column_formats" in prompt
    assert "round" in prompt


def test_nl_to_query_model_round_applied_in_select() -> None:
    """When the LLM returns column_formats, ROUND() should appear in the SQL."""
    stub = _StubClient(
        {
            "reply": "Showing price rounded to 2 decimal places.",
            "selected_columns": ["product", "price"],
            "column_formats": {"price": {"round": 2}},
        }
    )
    model = nl_to_query_model(
        "show product and price rounded to 2 decimals",
        SCHEMA,
        client=stub,  # type: ignore[arg-type]
    )
    sql = model.to_sql()
    assert 'ROUND("price", 2)' in sql
    assert '"product"' in sql


def test_parse_query_plan_rejects_non_numeric_column_format() -> None:
    payload = {
        "selected_columns": ["product"],
        "column_formats": {"product": {"round": 2}},
    }
    with pytest.raises(LLMError, match="formatting is only supported for numeric"):
        parse_query_plan(payload, SCHEMA)


def test_parse_query_plan_rejects_unknown_column_in_formats() -> None:
    payload = {
        "selected_columns": ["price"],
        "column_formats": {"total_cost": {"round": 2}},
    }
    with pytest.raises(LLMError, match="not in the dataset schema"):
        parse_query_plan(payload, SCHEMA)


def test_parse_query_plan_rejects_out_of_range_decimals() -> None:
    payload = {
        "selected_columns": ["price"],
        "column_formats": {"price": {"round": 99}},
    }
    with pytest.raises(LLMError, match="must be between 0 and"):
        parse_query_plan(payload, SCHEMA)


def test_parse_query_plan_no_column_formats_gives_plain_sql() -> None:
    payload = {"selected_columns": ["price"]}
    model = parse_query_plan(payload, SCHEMA)
    sql = model.to_sql()
    assert "ROUND" not in sql
    assert '"price"' in sql


# ---------------------------------------------------------------------------
# Test CSV integrity
# ---------------------------------------------------------------------------


def test_sales_csv_loads_correctly() -> None:
    """Verify the test fixture can be loaded by the ingestion pipeline."""
    pytest.importorskip("pandas")
    from src.ingestion import infer_schema, load_csv

    conn, schema = load_csv(SALES_CSV)
    conn.close()

    assert "product" in schema
    assert "category" in schema
    assert schema["price"] == "numeric"
    assert schema["quantity"] == "numeric"
    # sale_date should be detected as date
    assert schema["sale_date"] == "date"
