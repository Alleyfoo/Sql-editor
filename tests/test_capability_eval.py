from __future__ import annotations

import json
from pathlib import Path

from eval.capability_eval import (
    MockEvalProvider,
    evaluate_cases,
    load_cases,
    main,
)


def _write_cases(path: Path) -> None:
    payload = [
        {
            "id": "ok_case",
            "question": "Show product and sales where region is North.",
            "schema": {
                "region": "text",
                "product": "text",
                "sales": "numeric",
            },
            "mock_response": {
                "selected_columns": ["product", "sales"],
                "filters": [
                    {
                        "column": "region",
                        "operator": "=",
                        "value": "North",
                    }
                ],
            },
        },
        {
            "id": "hallucinated_col",
            "question": "Show margin by region.",
            "schema": {
                "region": "text",
                "product": "text",
                "sales": "numeric",
            },
            "mock_response": {
                "selected_columns": ["region", "margin"],
            },
        },
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_cases_mock_has_expected_rates(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    _write_cases(cases_path)
    cases = load_cases(cases_path)
    provider = MockEvalProvider(model="mock-test")

    report = evaluate_cases(provider=provider, cases=cases)

    assert report["cases_total"] == 2
    assert report["metrics"]["json_object_rate"] == 1.0
    assert report["metrics"]["valid_plan_rate"] == 0.5
    assert report["metrics"]["hallucination_rate"] == 0.5
    assert report["failures"]["hallucinated_column"] == 1


def test_main_writes_report_with_mock(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    report_path = tmp_path / "report.json"
    _write_cases(cases_path)

    rc = main(
        [
            "--provider",
            "mock",
            "--model",
            "mock-test",
            "--cases",
            str(cases_path),
            "--output",
            str(report_path),
        ]
    )
    assert rc == 0
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["provider"] == "mock"
    assert report["model"] == "mock-test"
    assert report["cases_total"] == 2

