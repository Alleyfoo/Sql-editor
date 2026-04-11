from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from .logical_plan import LogicalPlan


ALLOWED_SOURCE_KINDS = {"table_or_file"}
ALLOWED_FILTER_OPS = {
    "=",
    "!=",
    "<",
    ">",
    "<=",
    ">=",
    "BETWEEN",
    "LIKE",
    "NOT LIKE",
    "IS NULL",
    "IS NOT NULL",
}
ALLOWED_AGG_FNS = {"count", "sum", "avg", "min", "max"}
ALLOWED_ORDER_DIRECTIONS = {"asc", "desc"}
ALLOWED_POST_KINDS = {
    "rolling_mean",
    "percentile",
    "cumulative",
    "shape_repair",
    "date_alignment",
}
ALLOWED_OUTPUT_TYPES = {"text", "numeric", "date", "integer", "float"}
EXPR_RE = re.compile(r"^(?:(?:date)\(([A-Za-z_][A-Za-z0-9_]*)\)|([A-Za-z_][A-Za-z0-9_]*))$")


@dataclass(frozen=True)
class PlanValidationResult:
    ok: bool
    errors: List[str]


class PlanValidationError(ValueError):
    pass


def _parse_expr_column(expr: str) -> str | None:
    match = EXPR_RE.match(expr.strip())
    if not match:
        return None
    return match.group(1) or match.group(2)


def validate_logical_plan(plan: LogicalPlan, source_schema: Dict[str, str]) -> PlanValidationResult:
    errors: List[str] = []
    schema_cols: Set[str] = set(source_schema.keys())

    if plan.source.kind not in ALLOWED_SOURCE_KINDS:
        errors.append(f"unsupported source.kind: {plan.source.kind!r}")
    if not plan.source.name:
        errors.append("source.name is required")

    for col in plan.projection:
        if col not in schema_cols:
            errors.append(f"unknown projection column: {col!r}")

    for f in plan.filters:
        if f.column not in schema_cols:
            errors.append(f"unknown filter column: {f.column!r}")
        if f.op not in ALLOWED_FILTER_OPS:
            errors.append(f"unsupported filter op: {f.op!r}")
        if f.op == "BETWEEN":
            if not isinstance(f.value, (list, tuple)) or len(f.value) != 2:
                errors.append("BETWEEN requires 2 values")
        if f.op in {"IS NULL", "IS NOT NULL"} and f.value is not None:
            errors.append(f"{f.op} must not include value")

    for expr in plan.group_by:
        col = _parse_expr_column(expr)
        if col is None:
            errors.append(f"unsupported group_by expression: {expr!r}")
        elif col not in schema_cols:
            errors.append(f"group_by references unknown column: {col!r}")

    agg_aliases: Set[str] = set()
    for agg in plan.aggregates:
        if agg.fn not in ALLOWED_AGG_FNS:
            errors.append(f"unsupported aggregate fn: {agg.fn!r}")
        if agg.column != "*" and agg.column not in schema_cols:
            errors.append(f"aggregate references unknown column: {agg.column!r}")
        if agg.alias in agg_aliases:
            errors.append(f"duplicate aggregate alias: {agg.alias!r}")
        agg_aliases.add(agg.alias)

    for order in plan.order_by:
        if order.expr in agg_aliases:
            col = None
        else:
            col = _parse_expr_column(order.expr)
            if col is None:
                errors.append(f"unsupported order_by expression: {order.expr!r}")
            elif col not in schema_cols:
                errors.append(f"order_by references unknown column: {col!r}")
        if order.direction not in ALLOWED_ORDER_DIRECTIONS:
            errors.append(f"unsupported order direction: {order.direction!r}")

    if plan.limit is not None and (not isinstance(plan.limit, int) or plan.limit <= 0):
        errors.append(f"limit must be positive int or null, got {plan.limit!r}")

    for step in plan.post_processing:
        if step.kind not in ALLOWED_POST_KINDS:
            errors.append(f"unsupported post_processing kind: {step.kind!r}")

    for col in plan.expected_output_schema:
        if not col.name:
            errors.append("expected_output_schema column requires name")
        if col.type not in ALLOWED_OUTPUT_TYPES:
            errors.append(f"unsupported expected output type: {col.type!r}")

    return PlanValidationResult(ok=not errors, errors=errors)


def assert_valid_logical_plan(plan: LogicalPlan, source_schema: Dict[str, str]) -> None:
    result = validate_logical_plan(plan, source_schema)
    if not result.ok:
        raise PlanValidationError("; ".join(result.errors))


def logical_plan_schema_path() -> Path:
    return Path(__file__).with_name("logical_plan.schema.json")
