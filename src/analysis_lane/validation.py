from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

import pandas as pd

from .models import AnalysisPlan, AnalysisProfile, ChartSpec, DashboardSpec, InsightReport

ALLOWED_FAMILIES = {
    "kpi_summary",
    "trend",
    "segment_comparison",
    "dashboard_design",
    "follow_up_analysis",
    "guardrail",
}

ALLOWED_OUTPUTS = {"insight_report", "chart_spec", "dashboard_spec"}
ALLOWED_CHARTS = {"line", "bar", "scatter", "table"}
ALLOWED_AGGREGATIONS = {"sum", "avg", "count", "max", "min", "none"}
ALLOWED_STRENGTHS = {"descriptive", "correlative", "causal"}

# Time-aware phrasing: these vague segment terms are banned in time-series summaries.
# Use time-aware equivalents instead (peak day, highest month, etc.).
_VAGUE_SEGMENT_TERMS = re.compile(
    r"\b(top segment|leading segment|best segment|highest segment|lowest segment|bottom segment)\b",
    re.IGNORECASE,
)

# Chart types that are inappropriate for time-series data (monotonic temporal x-axis).
_BAD_TIMESERIES_CHARTS = {"bar", "scatter"}

# Chart types required when a time dimension is present and a chart is requested.
_GOOD_TIMESERIES_CHARTS = {"line"}

# Dashboard requested-metric coverage: tile kinds that can satisfy a metric requirement.
_METRIC_TILE_KINDS = {"kpi_card", "chart"}


def validate_analysis_profile(profile: AnalysisProfile) -> List[str]:
    errors: List[str] = []
    if profile.row_count < 0:
        errors.append("row_count must be >= 0")
    if profile.column_count < 0:
        errors.append("column_count must be >= 0")
    if profile.column_count == 0:
        errors.append("column_count must be > 0")
    names = {x.name for x in profile.nulls}
    if len(names) != len(profile.nulls):
        errors.append("nulls contains duplicate column names")
    return errors


def validate_analysis_plan(plan: AnalysisPlan) -> List[str]:
    errors: List[str] = []
    if plan.family not in ALLOWED_FAMILIES:
        errors.append(f"unsupported family: {plan.family}")
    if not plan.question.strip():
        errors.append("question is empty")
    for out in plan.outputs:
        if out not in ALLOWED_OUTPUTS:
            errors.append(f"unsupported output type: {out}")
    return errors


def validate_field_references(
    *,
    plan: AnalysisPlan,
    charts: List[ChartSpec],
    report: InsightReport,
    df: pd.DataFrame,
) -> List[str]:
    errors: List[str] = []
    cols = set(str(c) for c in df.columns)

    for col in plan.selected_dimensions + plan.selected_metrics:
        if col not in cols:
            errors.append(f"plan references unknown field: {col}")
    if plan.time_dimension and plan.time_dimension not in cols:
        errors.append(f"plan references unknown time_dimension: {plan.time_dimension}")

    for idx, chart in enumerate(charts):
        for field_name in [chart.x_field, chart.y_field, chart.series_field]:
            if field_name is None:
                continue
            if field_name not in cols:
                errors.append(f"chart[{idx}] references unknown field: {field_name}")

    for idx, insight in enumerate(report.insights):
        for field_name in insight.evidence_fields:
            if field_name not in cols:
                errors.append(f"insight[{idx}] references unknown evidence field: {field_name}")

    return errors


def validate_chart_spec(chart: ChartSpec) -> List[str]:
    errors: List[str] = []
    if chart.chart_type not in ALLOWED_CHARTS:
        errors.append(f"unsupported chart_type: {chart.chart_type}")
    if chart.aggregation not in ALLOWED_AGGREGATIONS:
        errors.append(f"unsupported aggregation: {chart.aggregation}")
    if not chart.title.strip():
        errors.append("chart title is empty")
    if not chart.x_field.strip() or not chart.y_field.strip():
        errors.append("x_field and y_field must be non-empty")
    return errors


def validate_dashboard_spec(spec: Optional[DashboardSpec], charts: List[ChartSpec]) -> List[str]:
    if spec is None:
        return []
    errors: List[str] = []
    if not spec.title.strip():
        errors.append("dashboard title is empty")
    for i, tile in enumerate(spec.tiles):
        if tile.kind not in {"kpi_card", "chart"}:
            errors.append(f"tile[{i}] has unsupported kind: {tile.kind}")
        if tile.kind == "kpi_card" and not tile.metric:
            errors.append(f"tile[{i}] kpi_card missing metric")
        if tile.kind == "chart":
            if tile.chart_ref is None:
                errors.append(f"tile[{i}] chart tile missing chart_ref")
            elif tile.chart_ref < 0 or tile.chart_ref >= len(charts):
                errors.append(f"tile[{i}] chart_ref out of range")
    return errors


def validate_insight_report(report: InsightReport) -> List[str]:
    errors: List[str] = []
    if not report.summary.strip():
        errors.append("report summary is empty")
    for i, insight in enumerate(report.insights):
        if insight.claim_strength not in ALLOWED_STRENGTHS:
            errors.append(f"insight[{i}] has unsupported claim_strength")
        if not (0.0 <= insight.confidence <= 1.0):
            errors.append(f"insight[{i}] confidence out of range")
        if not insight.evidence_fields:
            errors.append(f"insight[{i}] missing evidence_fields")
    return errors


def check_claim_strength_policy(
    report: InsightReport,
    *,
    evidence_contract_supports_causal: bool,
) -> bool:
    for insight in report.insights:
        if insight.claim_strength == "causal" and not evidence_contract_supports_causal:
            return False
    return True


def check_insight_grounding(report: InsightReport, df: pd.DataFrame) -> bool:
    cols = set(str(c) for c in df.columns)
    for insight in report.insights:
        if not insight.grounded:
            return False
        if any(field not in cols for field in insight.evidence_fields):
            return False
    return True


# ---------------------------------------------------------------------------
# Time-series phrasing guardrail
# ---------------------------------------------------------------------------

def check_timeseries_phrasing(report: InsightReport, plan: AnalysisPlan) -> List[str]:
    """Return error strings if time-series summaries use vague segment phrasing.

    When the plan has a time_dimension the summary and insight claims must use
    time-aware wording (e.g. "peak day", "highest month") rather than generic
    segment language like "top segment".
    """
    if not plan.time_dimension:
        return []
    errors: List[str] = []
    if _VAGUE_SEGMENT_TERMS.search(report.summary):
        errors.append(
            f"report summary uses vague segment phrasing in a time-series context; "
            "use time-aware wording such as 'peak day' or 'highest month'"
        )
    for i, insight in enumerate(report.insights):
        if _VAGUE_SEGMENT_TERMS.search(insight.claim):
            errors.append(
                f"insight[{i}] uses vague segment phrasing in a time-series context; "
                "use time-aware wording such as 'peak day' or 'highest month'"
            )
    return errors


def check_chart_type_for_timeseries(charts: List[ChartSpec], plan: AnalysisPlan) -> List[str]:
    """Return error strings if a chart uses a bad chart type for time-series data."""
    if not plan.time_dimension:
        return []
    errors: List[str] = []
    for i, chart in enumerate(charts):
        if chart.x_field == plan.time_dimension and chart.chart_type in _BAD_TIMESERIES_CHARTS:
            errors.append(
                f"chart[{i}] uses '{chart.chart_type}' for time-series x-axis '{plan.time_dimension}'; "
                "prefer 'line' for monotonic temporal data"
            )
    return errors


# ---------------------------------------------------------------------------
# Dashboard completeness check
# ---------------------------------------------------------------------------

def check_dashboard_metric_coverage(
    *,
    requested_metrics: List[str],
    dashboard: Optional[DashboardSpec],
    charts: List[ChartSpec],
    omission_notes: Optional[List[str]] = None,
) -> List[str]:
    """Return error strings for any requested metric not covered by the dashboard.

    A metric is considered covered if:
    - a kpi_card tile names it, OR
    - a chart tile's underlying ChartSpec references it as x_field or y_field, OR
    - the metric appears in an explicit omission_notes entry.
    """
    if not requested_metrics:
        return []
    if dashboard is None:
        # No dashboard requested — nothing to check here.
        return []

    covered: Set[str] = set()
    notes_text = " ".join(omission_notes or []).lower()

    for tile in dashboard.tiles:
        if tile.kind == "kpi_card" and tile.metric:
            covered.add(tile.metric)
        elif tile.kind == "chart" and tile.chart_ref is not None:
            if 0 <= tile.chart_ref < len(charts):
                chart = charts[tile.chart_ref]
                covered.add(chart.x_field)
                covered.add(chart.y_field)

    errors: List[str] = []
    for metric in requested_metrics:
        if metric not in covered:
            # Accept if metric mentioned in explicit omission notes
            if metric.lower() in notes_text or metric.lower().replace("_", " ") in notes_text:
                continue
            errors.append(
                f"requested metric '{metric}' is absent from dashboard tiles and has no omission note"
            )
    return errors


# ---------------------------------------------------------------------------
# Descriptive-vs-causal claim strength guardrail
# ---------------------------------------------------------------------------

_CAUSAL_LANGUAGE = re.compile(
    r"\b(causes?|caused by|drives?|driven by|leads? to|results? in|because|due to|"
    r"responsible for|attributable to|explains?|accounts? for)\b",
    re.IGNORECASE,
)


def check_causal_language_in_descriptive_claims(report: InsightReport) -> List[str]:
    """Return error strings if descriptive/correlative insights contain causal language.

    Causal-strength claims require an explicit evidence contract.  This check
    catches the silent promotion pattern where a descriptive claim uses causal
    wording while the claim_strength field stays 'descriptive' or 'correlative'.
    """
    errors: List[str] = []
    for i, insight in enumerate(report.insights):
        if insight.claim_strength in {"descriptive", "correlative"}:
            if _CAUSAL_LANGUAGE.search(insight.claim):
                errors.append(
                    f"insight[{i}] has claim_strength='{insight.claim_strength}' but uses "
                    f"causal language in the claim text; either upgrade to 'causal' "
                    "(and provide an evidence contract) or reword descriptively"
                )
    return errors


def check_weak_evidence(report: InsightReport) -> List[str]:
    """Return error strings for insights with suspiciously low confidence.

    A descriptive insight with confidence below 0.5 should not be presented
    as a concrete finding — the engine should block or flag it.
    """
    errors: List[str] = []
    for i, insight in enumerate(report.insights):
        if insight.confidence < 0.5 and insight.claim_strength != "causal":
            errors.append(
                f"insight[{i}] has low confidence ({insight.confidence:.2f}) "
                "below the 0.5 weak-evidence threshold; block or explicitly note uncertainty"
            )
    return errors
