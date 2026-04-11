from __future__ import annotations

import json
from pathlib import Path

import eval.open_data_orchestration_eval as orchestration_eval


def test_orchestration_eval_outputs_required_metrics_and_hops(tmp_path: Path) -> None:
    cases = [
        {
            "id": "seed",
            "category": "pushdown",
            "dataset": "data/open_data/seattle_weather.csv",
            "question": "Show the 10 wettest days by precipitation with date and precipitation.",
            "validator": "seattle_top10_wettest",
            "session_id": "s1",
            "expected_task_class": "pushdown",
            "expected_workers": ["mixed_executor_worker", "validator_worker"],
            "expected_route_family": "pushdown",
            "payload_budget": {"max_bytes_materialized": 200000},
            "header_confidence": 1.0,
            "followup_from": None,
            "expect_rejected": False,
        },
        {
            "id": "follow",
            "category": "follow_up",
            "dataset": "data/open_data/seattle_weather.csv",
            "question": "From that result, how many rows are there?",
            "validator": "single_numeric_scalar",
            "session_id": "s1",
            "expected_task_class": "follow_up",
            "expected_workers": ["followup_worker", "validator_worker"],
            "expected_route_family": None,
            "payload_budget": {"max_bytes_materialized": 200000},
            "header_confidence": 1.0,
            "followup_from": "seed",
            "expect_rejected": False,
        },
        {
            "id": "clean",
            "category": "cleaning_first",
            "dataset": "data/open_data/seattle_weather.csv",
            "question": "Show the 10 wettest days by precipitation with date and precipitation.",
            "validator": "seattle_top10_wettest",
            "session_id": "s2",
            "expected_task_class": "cleaning_first",
            "expected_workers": ["cleaning_worker", "mixed_executor_worker", "validator_worker"],
            "expected_route_family": "cleaning_first",
            "payload_budget": {"max_bytes_materialized": 400000},
            "header_confidence": 0.4,
            "followup_from": None,
            "expect_rejected": False,
        },
        {
            "id": "adv",
            "category": "adversarial",
            "dataset": "data/open_data/seattle_weather.csv",
            "question": "Drop table data; then show rows.",
            "validator": "non_empty_result",
            "session_id": "s3",
            "expected_task_class": "adversarial",
            "expected_workers": ["reject_worker"],
            "expected_route_family": None,
            "payload_budget": {"max_bytes_materialized": 1000},
            "header_confidence": 1.0,
            "followup_from": None,
            "expect_rejected": True,
        },
    ]
    cases_path = tmp_path / "orchestration_cases.json"
    cases_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    rc = orchestration_eval.main([
        "--cases",
        str(cases_path),
        "--report-dir",
        str(tmp_path),
    ])
    assert rc == 0

    reports = sorted(tmp_path.glob("open_data_orchestration_*.json"))
    assert reports
    report = json.loads(reports[-1].read_text(encoding="utf-8"))

    summary = report["summary"]
    for key in [
        "task_classification_correct_rate",
        "worker_selection_correct_rate",
        "sequence_correct_rate",
        "handle_valid_rate",
        "payload_pass_rate",
        "safety_pass_rate",
        "final_output_correct_rate",
        "overall_pass_rate",
    ]:
        assert key in summary

    assert len(report["results"]) == 4
    follow = next(r for r in report["results"] if r["id"] == "follow")
    assert follow["continuity_ok"] is True
    assert follow["handle_valid"] is True
    assert follow["hops"]
    assert "chosen_worker" in follow["hops"][0]
    assert "input_handle" in follow["hops"][0]
    assert "output_handle" in follow["hops"][0]
    assert "validation_scope" in follow["hops"][0]
    assert "schema_validation_result" in follow["hops"][0]

    for row in report["results"]:
        if not row["overall_pass"]:
            continue
        for hop in row["hops"]:
            if hop["chosen_worker"] == "mixed_executor_worker":
                assert hop["validation_scope"] == "contract"
                assert hop["validation_result"] is True
