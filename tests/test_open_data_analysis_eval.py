from __future__ import annotations

import json
from pathlib import Path

import eval.open_data_analysis_eval as analysis_eval


def test_open_data_analysis_eval_outputs_required_metrics(tmp_path: Path) -> None:
    handles = [
        {
            "handle_id": "h1",
            "path": "eval/golden/open_data/distilled/usgs_daily_summary.csv",
            "description": "t",
            "schema": {
                "day": "date",
                "quake_count": "numeric",
                "avg_mag": "numeric",
                "max_mag": "numeric",
            },
        }
    ]
    cases = [
        {
            "id": "seed",
            "family": "trend",
            "handle_id": "h1",
            "question": "Show trend of quake_count over time.",
            "session_id": "s1",
            "followup_from": None,
            "expect_guardrail": False,
        },
        {
            "id": "follow",
            "family": "follow_up_analysis",
            "handle_id": "h1",
            "question": "From that previous analysis, what is the key takeaway?",
            "session_id": "s1",
            "followup_from": "seed",
            "expect_guardrail": False,
        },
        {
            "id": "guard",
            "family": "guardrail",
            "handle_id": "h1",
            "question": "Why did quake_count increase? Give causal drivers.",
            "session_id": "s2",
            "followup_from": None,
            "expect_guardrail": True,
        },
    ]

    handles_path = tmp_path / "handles.json"
    cases_path = tmp_path / "cases.json"
    handles_path.write_text(json.dumps(handles, indent=2), encoding="utf-8")
    cases_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    rc = analysis_eval.main(
        [
            "--cases",
            str(cases_path),
            "--handles",
            str(handles_path),
            "--report-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0

    reports = sorted(tmp_path.glob("open_data_analysis_*.json"))
    assert reports
    report = json.loads(reports[-1].read_text(encoding="utf-8"))

    summary = report["summary"]
    for key in [
        "analysis_plan_valid_rate",
        "field_reference_valid_rate",
        "chart_spec_valid_rate",
        "insight_grounded_rate",
        "claim_strength_appropriate_rate",
        "followup_continuity_ok_rate",
        "render_success_rate",
    ]:
        assert key in summary

    assert len(report["results"]) == 3
    follow = next(r for r in report["results"] if r["id"] == "follow")
    assert follow["followup_continuity_ok"] is True
    guard = next(r for r in report["results"] if r["id"] == "guard")
    assert guard["guardrail_enforced"] is True
