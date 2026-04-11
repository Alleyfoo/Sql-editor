from __future__ import annotations

import json
from pathlib import Path

import eval.open_data_mixed_execution_eval as mixed_eval


def test_mixed_execution_eval_outputs_required_metrics(tmp_path: Path) -> None:
    cases = [
        {
            "id": "seattle_top10_wettest",
            "track": "sql_fit",
            "dataset": "data/open_data/seattle_weather.csv",
            "question": "Show the 10 wettest days by precipitation with date and precipitation.",
            "validator": "seattle_top10_wettest",
        },
        {
            "id": "usgs_p90_magnitude",
            "track": "python_fit",
            "dataset": "data/open_data/usgs_all_month.csv",
            "question": "What is the 90th percentile of magnitude as one number?",
            "validator": "usgs_p90_magnitude",
        },
    ]
    cases_path = tmp_path / "mixed_cases.json"
    cases_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    route_oracle = [
        {
            "id": "seattle_top10_wettest",
            "expected_route_family": "pushdown",
            "header_confidence": 1.0,
            "payload_budget": {
                "max_rows_scanned": 10000,
                "max_rows_materialized": 20,
                "max_bytes_fetched": 20000,
            },
        },
        {
            "id": "usgs_p90_magnitude",
            "expected_route_family": "hybrid_or_python",
            "header_confidence": 1.0,
            "payload_budget": {
                "max_rows_scanned": 50000,
                "max_rows_materialized": 12000,
                "max_bytes_fetched": 1200000,
            },
        },
    ]
    route_oracle_path = tmp_path / "mixed_route_oracle.json"
    route_oracle_path.write_text(json.dumps(route_oracle, indent=2), encoding="utf-8")

    rc = mixed_eval.main(
        [
            "--cases",
            str(cases_path),
            "--route-oracle",
            str(route_oracle_path),
            "--report-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0

    reports = sorted(tmp_path.glob("open_data_mixed_execution_*.json"))
    assert reports
    report = json.loads(reports[-1].read_text(encoding="utf-8"))

    summary = report["summary"]
    for key in [
        "plan_valid_rate",
        "route_correct_rate",
        "execution_correct_rate",
        "schema_correct_rate",
        "safety_pass_rate",
        "rows_scanned_avg",
        "rows_materialized_avg",
        "bytes_fetched_avg",
        "peak_memory_mb_avg",
        "fallback_used_rate",
        "payload_pass_rate",
        "overall_pass_rate",
    ]:
        assert key in summary
    assert "payload_summary" in summary
    assert "fallback_summary" in summary
    assert "safety_summary" in summary
    assert "backend_counts" in summary

    assert len(report["results"]) == 2
    required_row_fields = {
        "plan_valid",
        "route_correct",
        "execution_route",
        "execution_correct",
        "schema_correct",
        "safety_pass",
        "rows_scanned",
        "rows_materialized",
        "bytes_fetched",
        "peak_memory_mb",
        "fallback_used",
        "fallback_reason",
        "payload_budget",
        "payload_pass",
        "payload_violations",
        "overall_pass",
    }
    for row in report["results"]:
        assert required_row_fields.issubset(row.keys())


def test_mixed_execution_eval_payload_budget_violation_fails_case(tmp_path: Path) -> None:
    cases = [
        {
            "id": "seattle_top10_wettest",
            "track": "sql_fit",
            "dataset": "data/open_data/seattle_weather.csv",
            "question": "Show the 10 wettest days by precipitation with date and precipitation.",
            "validator": "seattle_top10_wettest",
        }
    ]
    cases_path = tmp_path / "mixed_cases.json"
    cases_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    route_oracle = [
        {
            "id": "seattle_top10_wettest",
            "expected_route_family": "pushdown",
            "header_confidence": 1.0,
            "payload_budget": {"max_rows_materialized": 1},
        }
    ]
    route_oracle_path = tmp_path / "mixed_route_oracle.json"
    route_oracle_path.write_text(json.dumps(route_oracle, indent=2), encoding="utf-8")

    rc = mixed_eval.main(
        [
            "--cases",
            str(cases_path),
            "--route-oracle",
            str(route_oracle_path),
            "--report-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0

    report = json.loads(sorted(tmp_path.glob("open_data_mixed_execution_*.json"))[-1].read_text(encoding="utf-8"))
    row = report["results"][0]
    assert row["payload_pass"] is False
    assert row["overall_pass"] is False
    assert row["payload_violations"]


def test_mixed_execution_eval_route_oracle_must_cover_all_cases(tmp_path: Path) -> None:
    cases = [
        {
            "id": "seattle_top10_wettest",
            "track": "sql_fit",
            "dataset": "data/open_data/seattle_weather.csv",
            "question": "Show the 10 wettest days by precipitation with date and precipitation.",
            "validator": "seattle_top10_wettest",
        }
    ]
    cases_path = tmp_path / "mixed_cases.json"
    cases_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    route_oracle_path = tmp_path / "mixed_route_oracle.json"
    route_oracle_path.write_text(json.dumps([], indent=2), encoding="utf-8")

    try:
        mixed_eval.main(
            [
                "--cases",
                str(cases_path),
                "--route-oracle",
                str(route_oracle_path),
                "--report-dir",
                str(tmp_path),
            ]
        )
        assert False, "expected SystemExit for missing oracle entry"
    except SystemExit as exc:
        assert "route oracle missing case ids" in str(exc)
