from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set

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


def _has_analytics_post_processing(plan: LogicalPlan) -> bool:
    return any(step.kind in ANALYTICS_HEAVY_STEPS for step in plan.post_processing)


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

