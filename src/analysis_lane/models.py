from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ColumnSummary:
    name: str
    type: str
    non_null: int
    null_count: int
    null_rate: float


@dataclass(frozen=True)
class MetricSummary:
    column: str
    count: int
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    p50: Optional[float]
    p90: Optional[float]
    max: Optional[float]


@dataclass(frozen=True)
class DateCoverage:
    column: str
    min_date: str
    max_date: str
    periods: int


@dataclass(frozen=True)
class CategorySummary:
    column: str
    top_values: List[Dict[str, Any]]


@dataclass(frozen=True)
class Hint:
    kind: str
    column: str
    message: str


@dataclass(frozen=True)
class AnalysisProfile:
    row_count: int
    column_count: int
    dimensions: List[str]
    metrics: List[str]
    nulls: List[ColumnSummary]
    summary_stats: List[MetricSummary]
    date_coverage: List[DateCoverage]
    top_categories: List[CategorySummary]
    trend_hints: List[Hint]
    anomaly_hints: List[Hint]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "dimensions": list(self.dimensions),
            "metrics": list(self.metrics),
            "nulls": [
                {
                    "name": x.name,
                    "type": x.type,
                    "non_null": x.non_null,
                    "null_count": x.null_count,
                    "null_rate": x.null_rate,
                }
                for x in self.nulls
            ],
            "summary_stats": [
                {
                    "column": x.column,
                    "count": x.count,
                    "mean": x.mean,
                    "std": x.std,
                    "min": x.min,
                    "p50": x.p50,
                    "p90": x.p90,
                    "max": x.max,
                }
                for x in self.summary_stats
            ],
            "date_coverage": [
                {
                    "column": x.column,
                    "min_date": x.min_date,
                    "max_date": x.max_date,
                    "periods": x.periods,
                }
                for x in self.date_coverage
            ],
            "top_categories": [
                {"column": x.column, "top_values": x.top_values}
                for x in self.top_categories
            ],
            "trend_hints": [
                {"kind": x.kind, "column": x.column, "message": x.message}
                for x in self.trend_hints
            ],
            "anomaly_hints": [
                {"kind": x.kind, "column": x.column, "message": x.message}
                for x in self.anomaly_hints
            ],
        }


@dataclass(frozen=True)
class AnalysisPlan:
    family: str
    question: str
    selected_dimensions: List[str]
    selected_metrics: List[str]
    time_dimension: Optional[str]
    outputs: List[str]
    claim_policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "question": self.question,
            "selected_dimensions": list(self.selected_dimensions),
            "selected_metrics": list(self.selected_metrics),
            "time_dimension": self.time_dimension,
            "outputs": list(self.outputs),
            "claim_policy": dict(self.claim_policy),
        }


@dataclass(frozen=True)
class ChartSpec:
    chart_type: str
    title: str
    x_field: str
    y_field: str
    series_field: Optional[str]
    aggregation: str
    filters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "x_field": self.x_field,
            "y_field": self.y_field,
            "series_field": self.series_field,
            "aggregation": self.aggregation,
            "filters": dict(self.filters),
        }


@dataclass(frozen=True)
class DashboardTile:
    kind: str
    title: str
    metric: Optional[str]
    chart_ref: Optional[int]


@dataclass(frozen=True)
class DashboardSpec:
    title: str
    tiles: List[DashboardTile]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "tiles": [
                {
                    "kind": t.kind,
                    "title": t.title,
                    "metric": t.metric,
                    "chart_ref": t.chart_ref,
                }
                for t in self.tiles
            ],
        }


@dataclass(frozen=True)
class Insight:
    claim: str
    claim_strength: str
    confidence: float
    evidence_fields: List[str]
    evidence_values: Dict[str, Any]
    grounded: bool


@dataclass(frozen=True)
class InsightReport:
    summary: str
    insights: List[Insight]
    blocked_claims: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "insights": [
                {
                    "claim": i.claim,
                    "claim_strength": i.claim_strength,
                    "confidence": i.confidence,
                    "evidence_fields": list(i.evidence_fields),
                    "evidence_values": dict(i.evidence_values),
                    "grounded": i.grounded,
                }
                for i in self.insights
            ],
            "blocked_claims": list(self.blocked_claims),
        }
