from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .models import (
    AnalysisPlan,
    AnalysisProfile,
    ChartSpec,
    DashboardSpec,
    DashboardTile,
    Insight,
    InsightReport,
)
from .profiler import build_analysis_profile
from .validation import (
    check_causal_language_in_descriptive_claims,
    check_chart_type_for_timeseries,
    check_claim_strength_policy,
    check_dashboard_metric_coverage,
    check_insight_grounding,
    check_timeseries_phrasing,
    check_weak_evidence,
    validate_analysis_plan,
    validate_analysis_profile,
    validate_chart_spec,
    validate_dashboard_spec,
    validate_field_references,
    validate_insight_report,
)


@dataclass(frozen=True)
class DistilledHandle:
    handle_id: str
    path: str
    schema: Dict[str, str]
    description: str


@dataclass(frozen=True)
class WorkerHop:
    worker: str
    input_handle: str
    output_handle: Optional[str]
    bytes_materialized: int
    validation_result: bool
    note: str


@dataclass(frozen=True)
class AnalysisRun:
    plan: AnalysisPlan
    profile: AnalysisProfile
    charts: List[ChartSpec]
    dashboard: Optional[DashboardSpec]
    report: InsightReport
    hops: List[WorkerHop]
    analysis_plan_valid: bool
    field_reference_valid: bool
    chart_spec_valid: bool
    insight_grounded: bool
    claim_strength_appropriate: bool
    followup_continuity_ok: bool
    render_success: bool
    guardrail_errors: List[str]
    guardrails_clean: bool


class AnalysisHandleStore:
    def __init__(self) -> None:
        self._values: Dict[str, Any] = {}
        self._counter = 0

    def put(self, value: Any) -> Tuple[str, int]:
        self._counter += 1
        hid = f"ahl_{self._counter:012x}"
        if isinstance(value, pd.DataFrame):
            b = int(value.memory_usage(index=False, deep=True).sum())
        else:
            b = len(json.dumps(value, default=str).encode("utf-8"))
        self._values[hid] = value
        return hid, b

    def get(self, handle_id: str) -> Any:
        return self._values[handle_id]

    def has(self, handle_id: Optional[str]) -> bool:
        return bool(handle_id and handle_id in self._values)


def load_distilled_handles(path: Path) -> Dict[str, DistilledHandle]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("distilled handles file must be an array")
    out: Dict[str, DistilledHandle] = {}
    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"handles[{idx}] must be an object")
        hid = str(row.get("handle_id") or "").strip()
        artifact_path = str(row.get("path") or "").strip()
        schema = row.get("schema") or {}
        desc = str(row.get("description") or "").strip()
        if not hid or not artifact_path or not isinstance(schema, dict):
            raise ValueError(f"handles[{idx}] missing required fields")
        out[hid] = DistilledHandle(handle_id=hid, path=artifact_path, schema={str(k): str(v) for k, v in schema.items()}, description=desc)
    return out


class AnalysisCoordinator:
    def __init__(self) -> None:
        self.store = AnalysisHandleStore()

    def profiling_worker(self, distilled_df: pd.DataFrame, schema: Dict[str, str]) -> Tuple[AnalysisProfile, str, int, bool]:
        profile = build_analysis_profile(distilled_df, schema)
        errors = validate_analysis_profile(profile)
        hid, b = self.store.put(profile.to_dict())
        return profile, hid, b, not errors

    def analysis_worker(
        self,
        *,
        question: str,
        profile: AnalysisProfile,
        distilled_df: pd.DataFrame,
        prior_analysis_handle: Optional[str],
    ) -> Tuple[AnalysisPlan, List[ChartSpec], Optional[DashboardSpec], InsightReport, str, int]:
        family = self._infer_family(question, has_prior=self.store.has(prior_analysis_handle))
        dims = list(profile.dimensions)
        metrics = list(profile.metrics)
        time_col = profile.date_coverage[0].column if profile.date_coverage else None

        selected_metrics = self._pick_metrics(question, metrics, max_items=2 if family in {"kpi_summary", "dashboard_design"} else 1)
        selected_dim = self._pick_dimension(question, dims, fallback=time_col)

        outputs: List[str] = ["insight_report"]
        if family in {"trend", "segment_comparison", "dashboard_design"}:
            outputs.append("chart_spec")
        if family == "dashboard_design":
            outputs.append("dashboard_spec")

        plan = AnalysisPlan(
            family=family,
            question=question,
            selected_dimensions=[selected_dim] if selected_dim else [],
            selected_metrics=selected_metrics,
            time_dimension=time_col if family in {"trend", "dashboard_design"} else None,
            outputs=outputs,
            claim_policy={
                "allow_causal": False,
                "restrict_to_descriptive": True,
                "prior_analysis_handle": prior_analysis_handle,
            },
        )

        charts = self._build_chart_specs(plan, distilled_df)
        report = self._build_insight_report(plan, distilled_df, prior_analysis_handle)
        dashboard = self._build_dashboard(plan, charts)

        payload = {
            "plan": plan.to_dict(),
            "charts": [c.to_dict() for c in charts],
            "dashboard": dashboard.to_dict() if dashboard else None,
            "report": report.to_dict(),
        }
        hid, b = self.store.put(payload)
        return plan, charts, dashboard, report, hid, b

    def chart_render_worker(self, chart: ChartSpec, distilled_df: pd.DataFrame) -> Tuple[bool, str, int]:
        errors = validate_chart_spec(chart)
        if chart.x_field not in distilled_df.columns or chart.y_field not in distilled_df.columns:
            return False, "chart references unknown fields", 0
        if errors:
            return False, "; ".join(errors), 0

        # Deterministic render artifact (typed JSON-ready payload).
        points = int(len(distilled_df[[chart.x_field, chart.y_field]].dropna()))
        artifact = {
            "type": "rendered_chart",
            "chart_type": chart.chart_type,
            "title": chart.title,
            "x_field": chart.x_field,
            "y_field": chart.y_field,
            "points": points,
        }
        _, b = self.store.put(artifact)
        return True, "rendered", b

    def dashboard_render_worker(self, dashboard: Optional[DashboardSpec], charts: List[ChartSpec]) -> Tuple[bool, str, int]:
        if dashboard is None:
            return True, "no dashboard requested", 0
        errors = validate_dashboard_spec(dashboard, charts)
        if errors:
            return False, "; ".join(errors), 0
        artifact = {
            "type": "dashboard",
            "title": dashboard.title,
            "tile_count": len(dashboard.tiles),
        }
        _, b = self.store.put(artifact)
        return True, "dashboard rendered", b

    def run(
        self,
        *,
        question: str,
        distilled_df: pd.DataFrame,
        schema: Dict[str, str],
        prior_analysis_handle: Optional[str],
        expected_followup_from: Optional[str],
    ) -> AnalysisRun:
        hops: List[WorkerHop] = []

        profile, profile_handle, profile_bytes, profile_ok = self.profiling_worker(distilled_df, schema)
        hops.append(
            WorkerHop(
                worker="profiling_worker",
                input_handle="distilled_source",
                output_handle=profile_handle,
                bytes_materialized=profile_bytes,
                validation_result=profile_ok,
                note="profile generated",
            )
        )

        plan, charts, dashboard, report, analysis_handle, analysis_bytes = self.analysis_worker(
            question=question,
            profile=profile,
            distilled_df=distilled_df,
            prior_analysis_handle=prior_analysis_handle,
        )
        hops.append(
            WorkerHop(
                worker="analysis_worker",
                input_handle=profile_handle,
                output_handle=analysis_handle,
                bytes_materialized=analysis_bytes,
                validation_result=True,
                note="typed analysis outputs produced",
            )
        )

        render_ok = True
        render_notes: List[str] = []
        total_render_bytes = 0
        for chart in charts:
            ok, note, b = self.chart_render_worker(chart, distilled_df)
            render_ok = render_ok and ok
            total_render_bytes += b
            render_notes.append(note)
        hops.append(
            WorkerHop(
                worker="chart_render_worker",
                input_handle=analysis_handle,
                output_handle=None,
                bytes_materialized=total_render_bytes,
                validation_result=render_ok,
                note=", ".join(render_notes) if render_notes else "no charts",
            )
        )

        dashboard_ok, dashboard_note, dashboard_bytes = self.dashboard_render_worker(dashboard, charts)
        hops.append(
            WorkerHop(
                worker="dashboard_render_worker",
                input_handle=analysis_handle,
                output_handle=None,
                bytes_materialized=dashboard_bytes,
                validation_result=dashboard_ok,
                note=dashboard_note,
            )
        )

        plan_errors = validate_analysis_plan(plan)
        field_errors = validate_field_references(plan=plan, charts=charts, report=report, df=distilled_df)
        chart_errors = [err for c in charts for err in validate_chart_spec(c)]
        report_errors = validate_insight_report(report)
        dashboard_errors = validate_dashboard_spec(dashboard, charts)

        analysis_plan_valid = not plan_errors
        field_reference_valid = not field_errors
        chart_spec_valid = not chart_errors and not dashboard_errors
        insight_grounded = check_insight_grounding(report, distilled_df) and not report_errors
        claim_strength_appropriate = check_claim_strength_policy(
            report,
            evidence_contract_supports_causal=bool(plan.claim_policy.get("allow_causal", False)),
        )

        followup_continuity_ok = True
        if expected_followup_from is not None:
            followup_continuity_ok = bool(
                prior_analysis_handle and plan.claim_policy.get("prior_analysis_handle") == prior_analysis_handle
            )

        # --- new guardrail checks ---
        guardrail_errors: List[str] = []
        guardrail_errors.extend(check_timeseries_phrasing(report, plan))
        guardrail_errors.extend(check_chart_type_for_timeseries(charts, plan))
        guardrail_errors.extend(
            check_dashboard_metric_coverage(
                requested_metrics=list(plan.selected_metrics),
                dashboard=dashboard,
                charts=charts,
            )
        )
        guardrail_errors.extend(check_causal_language_in_descriptive_claims(report))
        guardrail_errors.extend(check_weak_evidence(report))

        return AnalysisRun(
            plan=plan,
            profile=profile,
            charts=charts,
            dashboard=dashboard,
            report=report,
            hops=hops,
            analysis_plan_valid=analysis_plan_valid,
            field_reference_valid=field_reference_valid,
            chart_spec_valid=chart_spec_valid,
            insight_grounded=insight_grounded,
            claim_strength_appropriate=claim_strength_appropriate,
            followup_continuity_ok=followup_continuity_ok,
            render_success=render_ok and dashboard_ok,
            guardrail_errors=guardrail_errors,
            guardrails_clean=not guardrail_errors,
        )

    @staticmethod
    def _infer_family(question: str, has_prior: bool) -> str:
        q = question.lower()
        if has_prior and any(x in q for x in ["that", "previous", "follow", "same"]):
            return "follow_up_analysis"
        if any(x in q for x in ["why", "cause", "driver", "because"]):
            return "guardrail"
        if "dashboard" in q:
            return "dashboard_design"
        has_kpi = any(x in q for x in ["kpi", "summary", "snapshot"])
        has_strong_trend = any(x in q for x in ["trend", "over time", "time series"])
        has_weak_trend = any(x in q for x in ["monthly", "daily"])
        if has_strong_trend:
            return "trend"
        if has_kpi:
            return "kpi_summary"
        if has_weak_trend:
            return "trend"
        if any(x in q for x in ["segment", "by ", "compare"]):
            return "segment_comparison"
        return "kpi_summary"

    @staticmethod
    def _field_mentioned(question: str, field_name: str) -> bool:
        q = question.lower()
        field = field_name.lower()
        return field in q or field.replace("_", " ") in q

    @classmethod
    def _pick_metrics(cls, question: str, metrics: List[str], *, max_items: int) -> List[str]:
        matched = [m for m in metrics if cls._field_mentioned(question, m)]
        if matched:
            return matched[:max_items]
        return metrics[:max_items]

    @staticmethod
    def _pick_dimension(question: str, dims: List[str], *, fallback: Optional[str] = None) -> Optional[str]:
        q = question.lower()
        for d in dims:
            if d.lower() in q or d.lower().replace("_", " ") in q:
                return d
        if dims:
            return dims[0]
        return fallback

    @staticmethod
    def _build_chart_specs(plan: AnalysisPlan, df: pd.DataFrame) -> List[ChartSpec]:
        charts: List[ChartSpec] = []
        if "chart_spec" not in plan.outputs:
            return charts

        primary_metric = plan.selected_metrics[0] if plan.selected_metrics else None
        secondary_metric = plan.selected_metrics[1] if len(plan.selected_metrics) > 1 else None
        dim = plan.selected_dimensions[0] if plan.selected_dimensions else None
        if plan.family == "trend" and plan.time_dimension and primary_metric:
            charts.append(
                ChartSpec(
                    chart_type="line",
                    title=f"Trend of {primary_metric}",
                    x_field=plan.time_dimension,
                    y_field=primary_metric,
                    series_field=None,
                    aggregation="none",
                )
            )
        elif plan.family == "dashboard_design" and plan.time_dimension and primary_metric:
            charts.append(
                ChartSpec(
                    chart_type="line",
                    title=f"{primary_metric} over {plan.time_dimension}",
                    x_field=plan.time_dimension,
                    y_field=primary_metric,
                    series_field=None,
                    aggregation="none",
                )
            )
            if secondary_metric and dim and dim != plan.time_dimension:
                charts.append(
                    ChartSpec(
                        chart_type="bar",
                        title=f"{secondary_metric} by {dim}",
                        x_field=dim,
                        y_field=secondary_metric,
                        series_field=None,
                        aggregation="avg",
                    )
                )
        elif plan.family == "segment_comparison" and dim and primary_metric:
            charts.append(
                ChartSpec(
                    chart_type="bar",
                    title=f"{primary_metric} by {dim}",
                    x_field=dim,
                    y_field=primary_metric,
                    series_field=None,
                    aggregation="avg",
                )
            )
        elif primary_metric and plan.time_dimension:
            charts.append(
                ChartSpec(
                    chart_type="line" if plan.time_dimension in df.columns else "table",
                    title=f"{primary_metric} over {plan.time_dimension}",
                    x_field=plan.time_dimension,
                    y_field=primary_metric,
                    series_field=None,
                    aggregation="none",
                )
            )
        elif primary_metric:
            charts.append(
                ChartSpec(
                    chart_type="table",
                    title=f"KPI view for {primary_metric}",
                    x_field=primary_metric,
                    y_field=primary_metric,
                    series_field=None,
                    aggregation="none",
                )
            )
        return charts

    @staticmethod
    def _build_dashboard(plan: AnalysisPlan, charts: List[ChartSpec]) -> Optional[DashboardSpec]:
        if "dashboard_spec" not in plan.outputs:
            return None
        tiles: List[DashboardTile] = []
        if plan.selected_metrics:
            tiles.append(DashboardTile(kind="kpi_card", title="Primary KPI", metric=plan.selected_metrics[0], chart_ref=None))
        for i, chart in enumerate(charts):
            tiles.append(DashboardTile(kind="chart", title=chart.title, metric=None, chart_ref=i))
        return DashboardSpec(title="Distilled Data Dashboard", tiles=tiles)

    def _build_insight_report(self, plan: AnalysisPlan, df: pd.DataFrame, prior_analysis_handle: Optional[str]) -> InsightReport:
        blocked: List[str] = []
        insights: List[Insight] = []

        metric = plan.selected_metrics[0] if plan.selected_metrics else None
        dim = plan.selected_dimensions[0] if plan.selected_dimensions else None

        if plan.family == "guardrail":
            blocked.append("Causal/explanatory claims are blocked without explicit evidence contract")
            if metric and metric in df.columns:
                vals = pd.to_numeric(df[metric], errors="coerce").dropna()
                value = float(vals.mean()) if not vals.empty else None
                insights.append(
                    Insight(
                        claim=f"{metric} average is {value:.3f}" if value is not None else f"{metric} has limited numeric evidence",
                        claim_strength="descriptive",
                        confidence=0.7,
                        evidence_fields=[metric],
                        evidence_values={"mean": value},
                        grounded=True,
                    )
                )
            summary = (
                f"Guardrail applied for {metric}: only descriptive evidence-backed claims are allowed."
                if metric
                else "Guardrail applied: only descriptive evidence-backed claims are allowed."
            )
            return InsightReport(summary=summary, insights=insights, blocked_claims=blocked)

        if plan.family == "trend" and plan.time_dimension and metric and metric in df.columns:
            ordered = df[[plan.time_dimension, metric]].copy()
            ordered[plan.time_dimension] = pd.to_datetime(ordered[plan.time_dimension], errors="coerce")
            ordered[metric] = pd.to_numeric(ordered[metric], errors="coerce")
            ordered = ordered.dropna().sort_values(plan.time_dimension)
            if not ordered.empty:
                first = float(ordered.iloc[0][metric])
                last = float(ordered.iloc[-1][metric])
                direction = "increased" if last > first else "decreased" if last < first else "was flat"
                insights.append(
                    Insight(
                        claim=f"{metric} {direction} from {first:.3f} to {last:.3f}",
                        claim_strength="descriptive",
                        confidence=0.8,
                        evidence_fields=[plan.time_dimension, metric],
                        evidence_values={"first": first, "last": last},
                        grounded=True,
                    )
                )
        elif plan.family in {"segment_comparison", "dashboard_design"} and metric and dim and metric in df.columns and dim in df.columns:
            frame = df[[dim, metric]].copy()
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
            grouped = frame.dropna().groupby(dim, dropna=True)[metric].mean().sort_values(ascending=False)
            if not grouped.empty:
                top_dim = str(grouped.index[0])
                top_val = float(grouped.iloc[0])
                insights.append(
                    Insight(
                        claim=f"Top segment by {metric} is {top_dim} ({top_val:.3f})",
                        claim_strength="descriptive",
                        confidence=0.8,
                        evidence_fields=[dim, metric],
                        evidence_values={"top_segment": top_dim, "top_value": top_val},
                        grounded=True,
                    )
                )
        elif plan.family == "follow_up_analysis":
            n = int(len(df))
            prior_fields: List[str] = []
            if prior_analysis_handle and self.store.has(prior_analysis_handle):
                prior_payload = self.store.get(prior_analysis_handle)
                if isinstance(prior_payload, dict):
                    prior_plan = prior_payload.get("plan") or {}
                    if isinstance(prior_plan, dict):
                        prior_fields.extend(str(x) for x in prior_plan.get("selected_metrics", []) if isinstance(x, str))
                        prior_fields.extend(str(x) for x in prior_plan.get("selected_dimensions", []) if isinstance(x, str))
            prior_fields = [f for f in prior_fields if f in df.columns]
            evidence = prior_fields or ([str(df.columns[0])] if len(df.columns) else [])
            insights.append(
                Insight(
                    claim=(
                        f"Follow-up retains focus on {', '.join(prior_fields)} with {n} rows in current context"
                        if prior_fields
                        else f"Follow-up context contains {n} rows with consistent handle continuity"
                    ),
                    claim_strength="descriptive",
                    confidence=0.7,
                    evidence_fields=evidence,
                    evidence_values={"row_count": n, "prior_handle": prior_analysis_handle, "focus_fields": prior_fields},
                    grounded=True,
                )
            )
        else:
            if metric and metric in df.columns:
                vals = pd.to_numeric(df[metric], errors="coerce").dropna()
                if not vals.empty:
                    insights.append(
                        Insight(
                            claim=f"{metric} mean is {float(vals.mean()):.3f} across {int(vals.count())} observed rows",
                            claim_strength="descriptive",
                            confidence=0.75,
                            evidence_fields=[metric],
                            evidence_values={"mean": float(vals.mean())},
                            grounded=True,
                        )
                    )

        summary = "; ".join(i.claim for i in insights) if insights else "No strong grounded insight available"
        return InsightReport(summary=summary, insights=insights, blocked_claims=blocked)
