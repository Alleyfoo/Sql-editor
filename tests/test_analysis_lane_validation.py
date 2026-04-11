from __future__ import annotations

import pandas as pd

from src.analysis_lane.engine import AnalysisCoordinator
from src.analysis_lane.models import (
    AnalysisPlan,
    ChartSpec,
    DashboardSpec,
    DashboardTile,
    Insight,
    InsightReport,
)
from src.analysis_lane.validation import (
    check_causal_language_in_descriptive_claims,
    check_chart_type_for_timeseries,
    check_claim_strength_policy,
    check_dashboard_metric_coverage,
    check_insight_grounding,
    check_timeseries_phrasing,
    check_weak_evidence,
    validate_analysis_plan,
)


def test_claim_strength_policy_blocks_causal_without_contract() -> None:
    report = InsightReport(
        summary="test",
        insights=[
            Insight(
                claim="X causes Y",
                claim_strength="causal",
                confidence=0.8,
                evidence_fields=["a"],
                evidence_values={"a": 1},
                grounded=True,
            )
        ],
        blocked_claims=[],
    )
    assert not check_claim_strength_policy(report, evidence_contract_supports_causal=False)
    assert check_claim_strength_policy(report, evidence_contract_supports_causal=True)


def test_analysis_worker_emits_typed_plan_only() -> None:
    df = pd.DataFrame(
        {
            "day": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "quake_count": [10, 12, 11],
            "avg_mag": [1.1, 1.2, 1.15],
        }
    )
    schema = {"day": "date", "quake_count": "numeric", "avg_mag": "numeric"}
    coordinator = AnalysisCoordinator()
    run = coordinator.run(
        question="Show trend of quake_count over time.",
        distilled_df=df,
        schema=schema,
        prior_analysis_handle=None,
        expected_followup_from=None,
    )

    assert not validate_analysis_plan(run.plan)
    assert "sql" not in run.plan.to_dict()
    assert "python" not in run.plan.to_dict()


def test_insight_grounding_detects_unknown_fields() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    report = InsightReport(
        summary="s",
        insights=[
            Insight(
                claim="bad",
                claim_strength="descriptive",
                confidence=0.6,
                evidence_fields=["ghost"],
                evidence_values={"ghost": 1},
                grounded=True,
            )
        ],
        blocked_claims=[],
    )
    assert not check_insight_grounding(report, df)


def test_family_inference_prefers_kpi_when_summary_and_monthly() -> None:
    family = AnalysisCoordinator._infer_family(
        "Provide an executive KPI summary for monthly precipitation and rain_days.",
        has_prior=False,
    )
    assert family == "kpi_summary"


def test_family_inference_keeps_trend_for_trend_summary_phrasing() -> None:
    family = AnalysisCoordinator._infer_family(
        "Provide a trend summary of avg_mag during the recent 14-day window.",
        has_prior=False,
    )
    assert family == "trend"


# ---------------------------------------------------------------------------
# Time-series phrasing guardrail tests
# ---------------------------------------------------------------------------

def _make_plan(*, time_dimension: str | None = "day") -> AnalysisPlan:
    return AnalysisPlan(
        family="trend",
        question="show trend",
        selected_dimensions=[],
        selected_metrics=["quake_count"],
        time_dimension=time_dimension,
        outputs=["insight_report", "chart_spec"],
        claim_policy={},
    )


def _make_report(summary: str, claims: list[str] | None = None) -> InsightReport:
    insights = [
        Insight(
            claim=c,
            claim_strength="descriptive",
            confidence=0.8,
            evidence_fields=["day"],
            evidence_values={},
            grounded=True,
        )
        for c in (claims or [])
    ]
    return InsightReport(summary=summary, insights=insights, blocked_claims=[])


def test_timeseries_phrasing_rejects_top_segment_in_summary() -> None:
    plan = _make_plan()
    report = _make_report("The top segment recorded the highest value.")
    errors = check_timeseries_phrasing(report, plan)
    assert errors, "should flag vague 'top segment' phrasing in time-series summary"


def test_timeseries_phrasing_rejects_top_segment_in_insight_claim() -> None:
    plan = _make_plan()
    report = _make_report("OK summary", ["The leading segment drove the trend."])
    errors = check_timeseries_phrasing(report, plan)
    assert errors, "should flag vague 'leading segment' in time-series insight claim"


def test_timeseries_phrasing_accepts_peak_day_wording() -> None:
    plan = _make_plan()
    report = _make_report(
        "quake_count peaked on 2026-01-15 with 42 events.",
        ["The peak day occurred on 2026-01-15."],
    )
    errors = check_timeseries_phrasing(report, plan)
    assert not errors, f"should accept time-aware phrasing, got: {errors}"


def test_timeseries_phrasing_skipped_without_time_dimension() -> None:
    plan = _make_plan(time_dimension=None)
    report = _make_report("The top segment was X.")
    errors = check_timeseries_phrasing(report, plan)
    assert not errors, "phrasing check should be skipped when no time_dimension is set"


# ---------------------------------------------------------------------------
# Bad chart type for time-series
# ---------------------------------------------------------------------------

def _make_chart(chart_type: str, x_field: str = "day") -> ChartSpec:
    return ChartSpec(
        chart_type=chart_type,
        title="test chart",
        x_field=x_field,
        y_field="quake_count",
        series_field=None,
        aggregation="none",
    )


def test_chart_type_bar_rejected_for_timeseries() -> None:
    plan = _make_plan()
    errors = check_chart_type_for_timeseries([_make_chart("bar")], plan)
    assert errors, "bar chart on time axis should be flagged"


def test_chart_type_scatter_rejected_for_timeseries() -> None:
    plan = _make_plan()
    errors = check_chart_type_for_timeseries([_make_chart("scatter")], plan)
    assert errors, "scatter chart on time axis should be flagged"


def test_chart_type_line_accepted_for_timeseries() -> None:
    plan = _make_plan()
    errors = check_chart_type_for_timeseries([_make_chart("line")], plan)
    assert not errors, f"line chart on time axis should be accepted, got: {errors}"


def test_chart_type_check_skipped_non_timeseries_x() -> None:
    plan = _make_plan()
    # bar chart on a non-time x-field — should not be flagged
    errors = check_chart_type_for_timeseries([_make_chart("bar", x_field="magType")], plan)
    assert not errors, "bar chart on non-time x-axis should not be flagged"


# ---------------------------------------------------------------------------
# Dashboard completeness check
# ---------------------------------------------------------------------------

def _make_dashboard(metrics: list[str], chart_count: int = 0) -> DashboardSpec:
    tiles: list[DashboardTile] = [
        DashboardTile(kind="kpi_card", title=m, metric=m, chart_ref=None)
        for m in metrics
    ]
    for i in range(chart_count):
        tiles.append(DashboardTile(kind="chart", title=f"chart {i}", metric=None, chart_ref=i))
    return DashboardSpec(title="Dashboard", tiles=tiles)


def test_dashboard_completeness_passes_when_all_metrics_covered() -> None:
    dashboard = _make_dashboard(["quake_count", "avg_mag"])
    errors = check_dashboard_metric_coverage(
        requested_metrics=["quake_count", "avg_mag"],
        dashboard=dashboard,
        charts=[],
    )
    assert not errors


def test_dashboard_completeness_flags_missing_metric() -> None:
    dashboard = _make_dashboard(["quake_count"])
    errors = check_dashboard_metric_coverage(
        requested_metrics=["quake_count", "avg_mag"],
        dashboard=dashboard,
        charts=[],
    )
    assert any("avg_mag" in e for e in errors)


def test_dashboard_completeness_accepts_metric_in_chart_spec() -> None:
    chart = _make_chart("line", x_field="day")
    tile = DashboardTile(kind="chart", title="trend", metric=None, chart_ref=0)
    dashboard = DashboardSpec(title="D", tiles=[tile])
    # chart.y_field == "quake_count" satisfies coverage
    errors = check_dashboard_metric_coverage(
        requested_metrics=["quake_count"],
        dashboard=dashboard,
        charts=[chart],
    )
    assert not errors


def test_dashboard_completeness_accepts_explicit_omission_note() -> None:
    dashboard = _make_dashboard(["quake_count"])
    errors = check_dashboard_metric_coverage(
        requested_metrics=["quake_count", "avg_mag"],
        dashboard=dashboard,
        charts=[],
        omission_notes=["avg_mag intentionally omitted: insufficient data"],
    )
    assert not errors


def test_dashboard_completeness_skipped_when_no_dashboard() -> None:
    errors = check_dashboard_metric_coverage(
        requested_metrics=["quake_count"],
        dashboard=None,
        charts=[],
    )
    assert not errors


# ---------------------------------------------------------------------------
# Causal language in descriptive claims
# ---------------------------------------------------------------------------

def _descriptive_insight(claim: str, confidence: float = 0.8) -> Insight:
    return Insight(
        claim=claim,
        claim_strength="descriptive",
        confidence=confidence,
        evidence_fields=["x"],
        evidence_values={},
        grounded=True,
    )


def test_causal_language_flagged_in_descriptive_claim() -> None:
    report = InsightReport(
        summary="ok",
        insights=[_descriptive_insight("quake_count increases drive avg_mag upward")],
        blocked_claims=[],
    )
    errors = check_causal_language_in_descriptive_claims(report)
    assert errors, "causal verb 'drives' in descriptive claim should be flagged"


def test_causal_language_flagged_because_keyword() -> None:
    report = InsightReport(
        summary="ok",
        insights=[_descriptive_insight("avg_mag is high because of deep faults")],
        blocked_claims=[],
    )
    errors = check_causal_language_in_descriptive_claims(report)
    assert errors


def test_causal_language_clean_descriptive_passes() -> None:
    report = InsightReport(
        summary="ok",
        insights=[_descriptive_insight("quake_count averaged 12 per day")],
        blocked_claims=[],
    )
    errors = check_causal_language_in_descriptive_claims(report)
    assert not errors


def test_causal_strength_claim_not_flagged_by_language_check() -> None:
    # A 'causal' claim is already declared as such — the language check should skip it.
    insight = Insight(
        claim="rainfall causes flooding",
        claim_strength="causal",
        confidence=0.9,
        evidence_fields=["rainfall"],
        evidence_values={},
        grounded=True,
    )
    report = InsightReport(summary="ok", insights=[insight], blocked_claims=[])
    errors = check_causal_language_in_descriptive_claims(report)
    assert not errors


# ---------------------------------------------------------------------------
# Weak-evidence check
# ---------------------------------------------------------------------------

def test_weak_evidence_flagged_below_threshold() -> None:
    report = InsightReport(
        summary="ok",
        insights=[_descriptive_insight("slight pattern maybe", confidence=0.3)],
        blocked_claims=[],
    )
    errors = check_weak_evidence(report)
    assert errors


def test_weak_evidence_passes_above_threshold() -> None:
    report = InsightReport(
        summary="ok",
        insights=[_descriptive_insight("clear trend", confidence=0.7)],
        blocked_claims=[],
    )
    errors = check_weak_evidence(report)
    assert not errors


def test_weak_evidence_boundary_exactly_05_passes() -> None:
    report = InsightReport(
        summary="ok",
        insights=[_descriptive_insight("borderline", confidence=0.5)],
        blocked_claims=[],
    )
    errors = check_weak_evidence(report)
    assert not errors


# ---------------------------------------------------------------------------
# Follow-up analysis: new benchmark questions inferred correctly
# ---------------------------------------------------------------------------

def test_family_inference_monthly_instead_is_followup() -> None:
    family = AnalysisCoordinator._infer_family(
        "From that trend, can we see this monthly instead?",
        has_prior=True,
    )
    assert family == "follow_up_analysis"


def test_family_inference_zoom_into_spike_is_followup() -> None:
    family = AnalysisCoordinator._infer_family(
        "From that analysis, zoom into the spike — what happened during the peak day?",
        has_prior=True,
    )
    assert family == "follow_up_analysis"


def test_family_inference_compare_top3_periods_is_followup() -> None:
    family = AnalysisCoordinator._infer_family(
        "From that previous analysis, compare only the top 3 periods by event_count.",
        has_prior=True,
    )
    assert family == "follow_up_analysis"
