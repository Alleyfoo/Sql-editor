from __future__ import annotations

import pandas as pd
import pytest

from src.llm.result_analysis import (
    AnalysisError,
    analyze_result_with_llm,
    fallback_result_analysis,
)


class _StubClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate_json(self, system: str, user: str):
        self.calls.append((system, user))
        return self.payload


def test_analyze_result_with_llm_parses_payload() -> None:
    df = pd.DataFrame(
        [
            {"station": "A", "trip_count": 10},
            {"station": "B", "trip_count": 7},
        ]
    )
    client = _StubClient(
        {
            "summary": "Station A has the most trips in this result slice.",
            "insights": ["A leads B by 3 trips."],
            "next_questions": ["Do you want this split by day?"],
            "warnings": [],
        }
    )
    analysis = analyze_result_with_llm(
        "Top stations",
        'SELECT "station", "trip_count" FROM "data"',
        df,
        {"station": "text", "trip_count": "numeric"},
        client=client,  # type: ignore[arg-type]
    )
    assert "most trips" in analysis.summary
    assert analysis.insights
    assert len(client.calls) == 1


def test_analyze_result_with_llm_rejects_invalid_payload() -> None:
    df = pd.DataFrame([{"x": 1}])
    client = _StubClient({"insights": ["missing summary"]})
    with pytest.raises(AnalysisError):
        analyze_result_with_llm(
            "q",
            'SELECT "x" FROM "data"',
            df,
            {"x": "numeric"},
            client=client,  # type: ignore[arg-type]
        )


def test_fallback_result_analysis_has_summary() -> None:
    df = pd.DataFrame([{"metric": 1}, {"metric": 3}])
    analysis = fallback_result_analysis(
        "q",
        'SELECT "metric" FROM "data"',
        df,
        warning="analysis model unavailable",
    )
    assert "returned 2 rows" in analysis.summary.lower()
    assert analysis.warnings == ["analysis model unavailable"]
