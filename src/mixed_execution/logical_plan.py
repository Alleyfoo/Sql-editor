from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PlanSource:
    kind: str
    name: str


@dataclass(frozen=True)
class PlanFilter:
    column: str
    op: str
    value: Any = None


@dataclass(frozen=True)
class PlanAggregate:
    fn: str
    column: str
    alias: str


@dataclass(frozen=True)
class PlanOrder:
    expr: str
    direction: str


@dataclass(frozen=True)
class OutputColumn:
    name: str
    type: str


@dataclass(frozen=True)
class PostProcessingStep:
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LogicalPlan:
    source: PlanSource
    projection: List[str]
    filters: List[PlanFilter]
    group_by: List[str]
    aggregates: List[PlanAggregate]
    order_by: List[PlanOrder]
    post_processing: List[PostProcessingStep]
    expected_output_schema: List[OutputColumn]
    limit: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": {"kind": self.source.kind, "name": self.source.name},
            "projection": list(self.projection),
            "filters": [
                {"column": f.column, "op": f.op, "value": f.value}
                for f in self.filters
            ],
            "group_by": list(self.group_by),
            "aggregates": [
                {"fn": a.fn, "column": a.column, "as": a.alias}
                for a in self.aggregates
            ],
            "order_by": [
                {"expr": o.expr, "direction": o.direction}
                for o in self.order_by
            ],
            "limit": self.limit,
            "post_processing": [
                {"kind": p.kind, **(p.params or {})}
                for p in self.post_processing
            ],
            "expected_output_schema": [
                {"name": c.name, "type": c.type}
                for c in self.expected_output_schema
            ],
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "LogicalPlan":
        source_raw = payload.get("source") or {}
        return LogicalPlan(
            source=PlanSource(
                kind=str(source_raw.get("kind") or ""),
                name=str(source_raw.get("name") or ""),
            ),
            projection=[str(c) for c in (payload.get("projection") or [])],
            filters=[
                PlanFilter(
                    column=str(item.get("column") or ""),
                    op=str(item.get("op") or ""),
                    value=item.get("value"),
                )
                for item in (payload.get("filters") or [])
            ],
            group_by=[str(c) for c in (payload.get("group_by") or [])],
            aggregates=[
                PlanAggregate(
                    fn=str(item.get("fn") or ""),
                    column=str(item.get("column") or ""),
                    alias=str(item.get("as") or ""),
                )
                for item in (payload.get("aggregates") or [])
            ],
            order_by=[
                PlanOrder(
                    expr=str(item.get("expr") or ""),
                    direction=str(item.get("direction") or ""),
                )
                for item in (payload.get("order_by") or [])
            ],
            limit=payload.get("limit"),
            post_processing=[
                PostProcessingStep(
                    kind=str(item.get("kind") or ""),
                    params={k: v for k, v in item.items() if k != "kind"},
                )
                for item in (payload.get("post_processing") or [])
            ],
            expected_output_schema=[
                OutputColumn(
                    name=str(item.get("name") or ""),
                    type=str(item.get("type") or ""),
                )
                for item in (payload.get("expected_output_schema") or [])
            ],
            metadata=dict(payload.get("metadata") or {}),
        )

