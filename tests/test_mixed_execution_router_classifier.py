from __future__ import annotations

from src.mixed_execution.router import build_routing_artifact


def test_classifier_mixed_header_table_for_very_low_header_confidence() -> None:
    schema = {
        "1.1": "text",
        "2.1": "numeric",
        "label": "text",
    }
    artifact = build_routing_artifact(
        question="Summarize this table.",
        schema=schema,
        header_confidence=0.5,
    )
    assert artifact["table_type"] == "mixed_header_table"
    assert artifact["gate_triggered"] is True
    assert "header_confidence_very_low" in artifact["reason_codes"]


def test_classifier_ambiguous_table_for_low_header_confidence() -> None:
    schema = {
        "metric": "numeric",
        "category": "text",
    }
    artifact = build_routing_artifact(
        question="Show top categories by metric.",
        schema=schema,
        header_confidence=0.85,
    )
    assert artifact["table_type"] == "ambiguous_table"
    assert artifact["gate_triggered"] is True
    assert "header_confidence_low" in artifact["reason_codes"]


def test_classifier_label_indexed_report_for_section_columns_without_time() -> None:
    schema = {
        "1.1": "text",
        "2.1": "numeric",
        "Indicator": "text",
    }
    artifact = build_routing_artifact(
        question="Compare indicator values.",
        schema=schema,
        header_confidence=1.0,
    )
    assert artifact["table_type"] == "label_indexed_report"
    assert artifact["has_time_column"] is False
    assert artifact["section_number_columns"] == ["1.1", "2.1"]
    assert "section_index_without_time" in artifact["reason_codes"]


def test_classifier_adds_trend_without_time_reason_code() -> None:
    schema = {
        "1.1": "text",
        "2.1": "numeric",
    }
    artifact = build_routing_artifact(
        question="What is the trend over time?",
        schema=schema,
        header_confidence=1.0,
    )
    assert artifact["table_type"] == "label_indexed_report"
    assert "trend_requested_without_time_column" in artifact["reason_codes"]
    assert artifact["redirect_reason"] == "no_queryable_time_column"
