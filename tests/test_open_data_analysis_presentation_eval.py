from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import eval.open_data_analysis_presentation_eval as presentation_eval
from src.analysis_lane.engine import AnalysisCoordinator


def test_open_data_analysis_presentation_eval_outputs_required_metrics(tmp_path: Path) -> None:
    handles = [
        {
            "handle_id": "h1",
            "path": "eval/golden/open_data/distilled/usgs_daily_summary.csv",
            "description": "usgs daily distilled",
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
            "id": "trend_seed",
            "handle_id": "h1",
            "question": "Create a concise trend narrative of quake_count over time and recommend one chart.",
            "session_id": "s1",
            "followup_from": None,
            "expected_family": "trend",
            "expected_focus_fields": ["day", "quake_count"],
            "expected_chart_types": ["line"],
            "require_chart": True,
            "require_dashboard": False,
        },
        {
            "id": "dash",
            "handle_id": "h1",
            "question": "Design a dashboard for daily quake_count and avg_mag.",
            "session_id": "s2",
            "followup_from": None,
            "expected_family": "dashboard_design",
            "expected_focus_fields": ["day", "quake_count", "avg_mag"],
            "expected_chart_types": ["line", "bar"],
            "require_chart": True,
            "require_dashboard": True,
        },
        {
            "id": "follow",
            "handle_id": "h1",
            "question": "From that previous analysis, refine the summary for policy audience.",
            "session_id": "s1",
            "followup_from": "trend_seed",
            "expected_family": "follow_up_analysis",
            "expected_focus_fields": ["day", "quake_count"],
            "expected_chart_types": [],
            "require_chart": False,
            "require_dashboard": False,
        },
    ]

    handles_path = tmp_path / "handles.json"
    cases_path = tmp_path / "cases.json"
    handles_path.write_text(json.dumps(handles, indent=2), encoding="utf-8")
    cases_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    rc = presentation_eval.main(
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

    reports = sorted(tmp_path.glob("open_data_analysis_presentation_*.json"))
    assert reports
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    summary = report["summary"]

    for key in [
        "family_match_rate",
        "summary_useful_rate",
        "chart_choice_appropriate_rate",
        "dashboard_layout_coherent_rate",
        "insight_grounded_rate",
        "claim_strength_appropriate_rate",
        "followup_continuity_ok_rate",
        "render_success_rate",
        "overall_pass_rate",
    ]:
        assert key in summary

    by_id = {row["id"]: row for row in report["results"]}
    assert by_id["dash"]["dashboard_layout_coherent"] is True
    assert by_id["follow"]["followup_continuity_ok"] is True
    # guardrails_clean must be present in every result row
    for row in report["results"]:
        assert "guardrails_clean" in row
    # guardrails_clean_rate must be in summary
    assert "guardrails_clean_rate" in summary


# ---------------------------------------------------------------------------
# Guardrail integration tests via AnalysisCoordinator.run()
# ---------------------------------------------------------------------------

def _daily_quake_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
            "quake_count": [10, 15, 12, 18, 11],
            "avg_mag": [1.1, 1.3, 1.2, 1.4, 1.1],
        }
    )


def _daily_schema() -> dict:
    return {"day": "date", "quake_count": "numeric", "avg_mag": "numeric"}


def test_guardrails_clean_true_for_well_formed_trend() -> None:
    df = _daily_quake_df()
    coordinator = AnalysisCoordinator()
    run = coordinator.run(
        question="Show trend of quake_count over time.",
        distilled_df=df,
        schema=_daily_schema(),
        prior_analysis_handle=None,
        expected_followup_from=None,
    )
    assert run.guardrails_clean, f"Expected clean guardrails, got errors: {run.guardrail_errors}"
    assert run.guardrail_errors == []


def test_guardrails_clean_false_for_causal_language_in_descriptive() -> None:
    """Engine-level: a report with causal verbs in a descriptive claim is flagged."""
    from src.analysis_lane.models import AnalysisPlan, Insight, InsightReport
    from src.analysis_lane.validation import check_causal_language_in_descriptive_claims

    report = InsightReport(
        summary="quake_count increases drive avg_mag upward",
        insights=[
            Insight(
                claim="quake_count increases drive avg_mag upward",
                claim_strength="descriptive",
                confidence=0.8,
                evidence_fields=["quake_count"],
                evidence_values={},
                grounded=True,
            )
        ],
        blocked_claims=[],
    )
    errors = check_causal_language_in_descriptive_claims(report)
    assert errors, "causal-language guardrail should fire"


def test_guardrails_errors_surface_in_presentation_eval_report(tmp_path: Path) -> None:
    """End-to-end: guardrail_errors appear in the JSON report for every case."""
    handles = [
        {
            "handle_id": "h1",
            "path": "eval/golden/open_data/distilled/usgs_daily_summary.csv",
            "description": "usgs daily distilled",
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
            "id": "trend_guardrail_integration",
            "handle_id": "h1",
            "question": "Show the daily trend of quake_count over time.",
            "session_id": "s_gr",
            "followup_from": None,
            "expected_family": "trend",
            "expected_focus_fields": ["day", "quake_count"],
            "expected_chart_types": ["line"],
            "require_chart": True,
            "require_dashboard": False,
        },
    ]

    handles_path = tmp_path / "handles.json"
    cases_path = tmp_path / "cases.json"
    handles_path.write_text(json.dumps(handles), encoding="utf-8")
    cases_path.write_text(json.dumps(cases), encoding="utf-8")

    rc = presentation_eval.main(
        ["--cases", str(cases_path), "--handles", str(handles_path), "--report-dir", str(tmp_path)]
    )
    assert rc == 0

    reports = sorted(tmp_path.glob("open_data_analysis_presentation_*.json"))
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    row = report["results"][0]

    assert "guardrails_clean" in row
    assert "guardrail_errors" in row
    assert isinstance(row["guardrail_errors"], list)
    assert "guardrails_clean_rate" in report["summary"]


def test_weak_evidence_guardrail_fires_at_engine_level() -> None:
    """check_weak_evidence is wired: coordinator flags low-confidence insights."""
    from src.analysis_lane.models import Insight, InsightReport
    from src.analysis_lane.validation import check_weak_evidence

    report = InsightReport(
        summary="maybe a slight pattern",
        insights=[
            Insight(
                claim="slight pattern maybe present",
                claim_strength="descriptive",
                confidence=0.3,
                evidence_fields=["quake_count"],
                evidence_values={},
                grounded=True,
            )
        ],
        blocked_claims=[],
    )
    errors = check_weak_evidence(report)
    assert errors, "weak-evidence guardrail should fire for confidence=0.3"


def test_timeseries_phrasing_guardrail_fires_at_engine_level() -> None:
    """check_timeseries_phrasing is wired: coordinator flags vague segment phrasing."""
    from src.analysis_lane.models import AnalysisPlan, Insight, InsightReport
    from src.analysis_lane.validation import check_timeseries_phrasing

    plan = AnalysisPlan(
        family="trend",
        question="trend of quake_count",
        selected_dimensions=[],
        selected_metrics=["quake_count"],
        time_dimension="day",
        outputs=["insight_report", "chart_spec"],
        claim_policy={},
    )
    report = InsightReport(
        summary="The top segment had the most quakes.",
        insights=[],
        blocked_claims=[],
    )
    errors = check_timeseries_phrasing(report, plan)
    assert errors, "timeseries phrasing guardrail should fire for 'top segment'"


def test_bad_chart_guardrail_fires_at_engine_level() -> None:
    """check_chart_type_for_timeseries is wired: coordinator flags bar on time axis."""
    from src.analysis_lane.models import AnalysisPlan, ChartSpec
    from src.analysis_lane.validation import check_chart_type_for_timeseries

    plan = AnalysisPlan(
        family="trend",
        question="bar chart of quake_count",
        selected_dimensions=[],
        selected_metrics=["quake_count"],
        time_dimension="day",
        outputs=["insight_report", "chart_spec"],
        claim_policy={},
    )
    chart = ChartSpec(
        chart_type="bar",
        title="quake_count by day",
        x_field="day",
        y_field="quake_count",
        series_field=None,
        aggregation="none",
    )
    errors = check_chart_type_for_timeseries([chart], plan)
    assert errors, "bad-chart-type guardrail should fire for bar on time axis"

