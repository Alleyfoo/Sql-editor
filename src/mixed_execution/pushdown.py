from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Protocol

import pandas as pd

from src.executor import execute
from src.query_model import Aggregation, Filter, QueryModel

from .logical_plan import LogicalPlan, PlanFilter


@dataclass(frozen=True)
class CostEstimate:
    rows_scanned: int
    rows_materialized: int
    bytes_fetched: int


@dataclass
class ExecutionResult:
    dataframe: pd.DataFrame
    rows_scanned: int
    rows_materialized: int
    bytes_fetched: int
    backend: str


class PushdownBackend(Protocol):
    def supports(self, plan: LogicalPlan) -> bool:
        ...

    def estimate_cost(self, plan: LogicalPlan, source_rows: int) -> CostEstimate:
        ...

    def execute(self, plan: LogicalPlan, conn: Any, source_rows: int) -> ExecutionResult:
        ...


def _date_placeholder_to_iso(value: object) -> object:
    if not isinstance(value, str):
        return value
    if value == "NOW_MINUS_7_DAYS":
        return (datetime.now(timezone.utc).date() - timedelta(days=6)).isoformat()
    if value == "NOW_MINUS_14_DAYS":
        return (datetime.now(timezone.utc).date() - timedelta(days=13)).isoformat()
    return value


def _resolve_placeholder_from_max_day(placeholder: str, max_day: Optional[str]) -> str:
    if max_day:
        parsed = pd.to_datetime(max_day, errors="coerce")
        if pd.notna(parsed):
            delta = 6 if placeholder == "NOW_MINUS_7_DAYS" else 13
            return (parsed.date() - timedelta(days=delta)).isoformat()
    return str(_date_placeholder_to_iso(placeholder))


def _resolve_relative_date_filters(plan: LogicalPlan, conn: Any) -> LogicalPlan:
    resolved_filters = []
    for f in plan.filters:
        if f.op in {"IS NULL", "IS NOT NULL"}:
            resolved_filters.append(f)
            continue
        if not isinstance(f.value, str):
            resolved_filters.append(f)
            continue
        if f.value not in {"NOW_MINUS_7_DAYS", "NOW_MINUS_14_DAYS"}:
            resolved_filters.append(f)
            continue

        query = f'SELECT MAX(substr("{f.column}", 1, 10)) AS max_day FROM "data"'
        frame = execute(conn, query)
        max_day = None
        if not frame.empty:
            max_day = str(frame.iloc[0]["max_day"]) if frame.iloc[0]["max_day"] is not None else None
        value = _resolve_placeholder_from_max_day(f.value, max_day)
        resolved_filters.append(PlanFilter(column=f.column, op=f.op, value=value))

    return LogicalPlan(
        source=plan.source,
        projection=plan.projection,
        filters=resolved_filters,
        group_by=plan.group_by,
        aggregates=plan.aggregates,
        order_by=plan.order_by,
        post_processing=plan.post_processing,
        expected_output_schema=plan.expected_output_schema,
        limit=plan.limit,
        metadata=plan.metadata,
    )


def _resolve_relative_date_filters_df(plan: LogicalPlan, source_df: pd.DataFrame) -> LogicalPlan:
    resolved_filters = []
    for f in plan.filters:
        if f.op in {"IS NULL", "IS NOT NULL"}:
            resolved_filters.append(f)
            continue
        if not isinstance(f.value, str):
            resolved_filters.append(f)
            continue
        if f.value not in {"NOW_MINUS_7_DAYS", "NOW_MINUS_14_DAYS"}:
            resolved_filters.append(f)
            continue

        max_day: Optional[str] = None
        if f.column in source_df.columns:
            series = source_df[f.column].dropna().astype(str).str.slice(0, 10)
            if not series.empty:
                max_day = str(series.max())
        value = _resolve_placeholder_from_max_day(f.value, max_day)
        resolved_filters.append(PlanFilter(column=f.column, op=f.op, value=value))

    return LogicalPlan(
        source=plan.source,
        projection=plan.projection,
        filters=resolved_filters,
        group_by=plan.group_by,
        aggregates=plan.aggregates,
        order_by=plan.order_by,
        post_processing=plan.post_processing,
        expected_output_schema=plan.expected_output_schema,
        limit=plan.limit,
        metadata=plan.metadata,
    )


def _expr_is_date(expr: str) -> Optional[str]:
    expr = expr.strip()
    if expr.startswith("date(") and expr.endswith(")"):
        return expr[5:-1].strip()
    return None


class SQLPushdownBackend:
    name = "sql"

    def supports(self, plan: LogicalPlan) -> bool:
        # pushdown backend intentionally handles only non-analytics plan bodies.
        analytics_only_steps = {"rolling_mean", "percentile", "cumulative", "shape_repair", "date_alignment"}
        if any(step.kind in analytics_only_steps for step in plan.post_processing):
            return False
        return True

    def estimate_cost(self, plan: LogicalPlan, source_rows: int) -> CostEstimate:
        expected_rows = source_rows
        if plan.limit is not None:
            expected_rows = min(expected_rows, plan.limit)
        elif plan.group_by:
            expected_rows = max(1, source_rows // 10)
        elif plan.filters:
            expected_rows = max(1, source_rows // 5)
        cols = max(1, len(plan.projection) + len(plan.aggregates))
        bytes_fetched = expected_rows * cols * 16
        return CostEstimate(
            rows_scanned=source_rows,
            rows_materialized=expected_rows,
            bytes_fetched=bytes_fetched,
        )

    def execute(self, plan: LogicalPlan, conn: Any, source_rows: int) -> ExecutionResult:
        resolved_plan = _resolve_relative_date_filters(plan, conn)
        query_model = self._compile_to_query_model(resolved_plan)
        sql = query_model.to_sql()
        df = execute(conn, sql)
        return ExecutionResult(
            dataframe=df,
            rows_scanned=source_rows,
            rows_materialized=len(df),
            bytes_fetched=int(df.memory_usage(index=False, deep=True).sum()),
            backend=self.name,
        )

    def _compile_to_query_model(self, plan: LogicalPlan) -> QueryModel:
        date_buckets: Dict[str, str] = {}
        selected_columns = []
        for col in plan.projection:
            date_col = _expr_is_date(col)
            if date_col:
                selected_columns.append(date_col)
                date_buckets[date_col] = "day"
            else:
                selected_columns.append(col)

        filters = []
        for f in plan.filters:
            if f.op == "BETWEEN":
                value = tuple(f.value) if isinstance(f.value, (list, tuple)) else f.value
            else:
                value = _date_placeholder_to_iso(f.value)
            filters.append(Filter(column=f.column, operator=f.op, value=value))

        group_by = []
        for expr in plan.group_by:
            date_col = _expr_is_date(expr)
            if date_col:
                group_by.append(date_col)
                date_buckets[date_col] = "day"
                if date_col not in selected_columns:
                    selected_columns.append(date_col)
            else:
                group_by.append(expr)

        aggs = []
        for agg in plan.aggregates:
            func = agg.fn.upper()
            if func == "COUNT" and agg.column == "*":
                aggs.append(Aggregation(function="COUNT", column="*", alias=agg.alias))
            else:
                aggs.append(Aggregation(function=func, column=agg.column, alias=agg.alias))

        order_by = []
        for order in plan.order_by:
            date_col = _expr_is_date(order.expr)
            order_col = date_col if date_col else order.expr
            order_by.append((order_col, order.direction.upper()))

        return QueryModel(
            selected_columns=selected_columns,
            filters=filters,
            group_by=group_by,
            aggregations=aggs,
            order_by=order_by,
            limit=plan.limit,
            date_buckets=date_buckets,
        )


class DataFramePushdownBackend:
    """Pushdown-capable backend executed directly against an in-memory DataFrame.

    This backend mirrors SQL pushdown semantics for filter/group/order/limit plans
    and is used for backend parity checks.
    """

    name = "dataframe_pushdown"

    def supports(self, plan: LogicalPlan) -> bool:
        analytics_only_steps = {"rolling_mean", "percentile", "cumulative", "shape_repair", "date_alignment"}
        if any(step.kind in analytics_only_steps for step in plan.post_processing):
            return False
        return True

    def estimate_cost(self, plan: LogicalPlan, source_rows: int) -> CostEstimate:
        expected_rows = source_rows
        if plan.limit is not None:
            expected_rows = min(expected_rows, plan.limit)
        elif plan.group_by:
            expected_rows = max(1, source_rows // 10)
        elif plan.filters:
            expected_rows = max(1, source_rows // 5)
        cols = max(1, len(plan.projection) + len(plan.aggregates))
        bytes_fetched = expected_rows * cols * 16
        return CostEstimate(
            rows_scanned=source_rows,
            rows_materialized=expected_rows,
            bytes_fetched=bytes_fetched,
        )

    def execute(self, plan: LogicalPlan, conn: Any, source_rows: int) -> ExecutionResult:
        if not isinstance(conn, pd.DataFrame):
            raise TypeError("DataFramePushdownBackend requires a pandas DataFrame as input")

        working_df = conn.copy()
        for col in working_df.columns:
            series = working_df[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                # Normalize datetime columns to string form so filter comparisons
                # match the SQLite/text semantics used by SQL pushdown.
                parsed = pd.to_datetime(series, errors="coerce", utc=True)
                working_df[col] = parsed.dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
            elif pd.api.types.is_object_dtype(series):
                parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
                if parsed.notna().mean() >= 0.8:
                    working_df[col] = parsed.dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")

        resolved_plan = _resolve_relative_date_filters_df(plan, working_df)
        # Local import avoids module-cycle at import time.
        from .python_analytics import PythonAnalyticsExecutor

        python_exec = PythonAnalyticsExecutor()
        out = python_exec.execute(resolved_plan, working_df, rows_scanned=source_rows)
        return ExecutionResult(
            dataframe=out.dataframe,
            rows_scanned=out.rows_scanned,
            rows_materialized=out.rows_materialized,
            bytes_fetched=out.bytes_fetched,
            backend=self.name,
        )
