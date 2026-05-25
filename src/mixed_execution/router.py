from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from .logical_plan import LogicalPlan


RouteName = str


@dataclass(frozen=True)
class RouteDecision:
    route: RouteName
    reason: str
    scores: Dict[str, float]


@dataclass(frozen=True)
class SourceProfile:
    rows_estimate: int
    is_remote: bool = False
    header_confidence: float = 1.0


ANALYTICS_HEAVY_STEPS: Set[str] = {
    "rolling_mean",
    "percentile",
    "cumulative",
    "shape_repair",
    "date_alignment",
}

_SECTION_NUMBER_COL_RE = re.compile(r"^\s*\d+(?:\.\d+){1,3}\s*$")
_DATE_LIKE_NAME_RE = re.compile(r"(date|time|year|month|day|week|quarter)", re.IGNORECASE)


def _has_analytics_post_processing(plan: LogicalPlan) -> bool:
    return any(step.kind in ANALYTICS_HEAVY_STEPS for step in plan.post_processing)


def _unique(seq: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_routing_artifact(
    *,
    question: str,
    schema: Dict[str, str],
    header_confidence: float,
) -> Dict[str, Any]:
    """Build a typed routing artifact for observability and debugging."""
    date_like_column_names = [
        col
        for col, col_type in schema.items()
        if col_type == "date" or _DATE_LIKE_NAME_RE.search(col)
    ]
    section_number_columns = [
        col
        for col in schema
        if _SECTION_NUMBER_COL_RE.match(str(col))
    ]
    has_time_column = bool(
        any(col_type == "date" for col_type in schema.values())
        or any("time" in str(col).lower() for col in schema)
    )

    reason_codes: List[str] = []
    if header_confidence < 0.6:
        table_type = "mixed_header_table"
        reason_codes.append("header_confidence_very_low")
    elif header_confidence < 0.9:
        table_type = "ambiguous_table"
        reason_codes.append("header_confidence_low")
    elif section_number_columns and not has_time_column:
        table_type = "label_indexed_report"
        reason_codes.append("section_index_without_time")
    else:
        table_type = "structured_table"

    if table_type == "structured_table":
        recommended_intents = ["exact_lookup", "ranking", "segment_comparison"]
        if has_time_column:
            recommended_intents.append("trend")
        blocked_intents: List[str] = []
    elif table_type == "label_indexed_report":
        recommended_intents = ["exact_lookup", "ranking", "analysis_summary"]
        blocked_intents = ["trend"] if not has_time_column else []
    else:
        recommended_intents = ["preprocess_then_query", "analysis_summary"]
        blocked_intents = ["exact_lookup", "trend"]

    q = (question or "").strip().lower()
    trend_requested = any(token in q for token in ["trend", "over time", "time series", "year-over-year", "yoy"])
    if trend_requested and not has_time_column:
        reason_codes.append("trend_requested_without_time_column")
        blocked_intents.append("trend")

    blocked_intents = _unique(blocked_intents)
    reason_codes = _unique(reason_codes)

    redirect_reason = ""
    if "trend_requested_without_time_column" in reason_codes:
        redirect_reason = "no_queryable_time_column"
    elif table_type == "mixed_header_table":
        redirect_reason = "mixed_header_table_requires_preprocessing"
    elif table_type == "ambiguous_table":
        redirect_reason = "ambiguous_structure_requires_preprocessing"
    elif table_type == "label_indexed_report" and not has_time_column:
        redirect_reason = "label_indexed_report_no_time_axis"

    gate_triggered = bool(table_type != "structured_table" or redirect_reason)

    return {
        "table_type": table_type,
        "has_time_column": has_time_column,
        "section_number_columns": section_number_columns,
        "date_like_column_names": date_like_column_names,
        "recommended_intents": recommended_intents,
        "blocked_intents": blocked_intents,
        "reason_codes": reason_codes,
        "gate_triggered": gate_triggered,
        "redirect_reason": redirect_reason,
    }


def apply_route_decision_to_artifact(
    artifact: Dict[str, Any],
    *,
    route: str,
    route_reason: str,
) -> Dict[str, Any]:
    out = dict(artifact)
    reason_codes = list(out.get("reason_codes") or [])
    reason_codes.append(f"route_{route}")
    if route_reason:
        reason_codes.append(f"route_reason_{route_reason}")
    if route == "cleaning_first":
        out["gate_triggered"] = True
        if not out.get("redirect_reason"):
            out["redirect_reason"] = "table_requires_preprocessing"
    out["reason_codes"] = _unique([str(x) for x in reason_codes if x])
    return out


def route_plan(plan: LogicalPlan, profile: SourceProfile) -> RouteDecision:
    """Deterministic router for pushdown/hybrid/python/cleaning_first."""
    score_pushdown = 0.0
    score_hybrid = 0.0
    score_python = 0.0
    score_cleaning = 0.0

    if profile.header_confidence < 0.9:
        score_cleaning += 10.0

    has_analytics = _has_analytics_post_processing(plan)
    has_pushdown_ops = bool(plan.filters or plan.group_by or plan.aggregates or plan.order_by or plan.limit)
    selective = bool(plan.filters or plan.limit)

    if has_analytics:
        score_python += 5.0
        score_hybrid += 6.0 if has_pushdown_ops else 2.0
    else:
        score_pushdown += 6.0

    if profile.is_remote:
        score_pushdown += 4.0
        score_hybrid += 3.0

    if profile.rows_estimate >= 50_000:
        score_pushdown += 3.0
        if has_analytics:
            score_hybrid += 3.0
    elif profile.rows_estimate <= 5_000:
        score_python += 2.0

    if selective:
        score_pushdown += 2.0
        score_hybrid += 2.0
    else:
        score_python += 1.0

    if score_cleaning >= max(score_pushdown, score_hybrid, score_python):
        return RouteDecision(
            route="cleaning_first",
            reason="header_confidence_below_threshold",
            scores={
                "pushdown": score_pushdown,
                "hybrid": score_hybrid,
                "python": score_python,
                "cleaning_first": score_cleaning,
            },
        )

    if score_hybrid >= max(score_pushdown, score_python):
        reason = "analytics_after_selective_pushdown" if has_pushdown_ops else "analytics_heavy"
        return RouteDecision(
            route="hybrid",
            reason=reason,
            scores={
                "pushdown": score_pushdown,
                "hybrid": score_hybrid,
                "python": score_python,
                "cleaning_first": score_cleaning,
            },
        )

    if score_pushdown >= score_python:
        return RouteDecision(
            route="pushdown",
            reason="simple_pushdown_operations",
            scores={
                "pushdown": score_pushdown,
                "hybrid": score_hybrid,
                "python": score_python,
                "cleaning_first": score_cleaning,
            },
        )

    return RouteDecision(
        route="python",
        reason="analytics_heavy_small_local_source",
        scores={
            "pushdown": score_pushdown,
            "hybrid": score_hybrid,
            "python": score_python,
            "cleaning_first": score_cleaning,
        },
    )


def route_matches_expectation(route: str, expected_family: Optional[str]) -> bool:
    if expected_family is None:
        return True
    if expected_family == "pushdown":
        return route == "pushdown"
    if expected_family == "hybrid_or_python":
        return route in {"hybrid", "python"}
    if expected_family == "cleaning_first":
        return route == "cleaning_first"
    return False
