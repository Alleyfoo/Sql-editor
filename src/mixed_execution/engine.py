from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.ingestion import load_csv

from .logical_plan import LogicalPlan
from .plan_validator import validate_logical_plan
from .planner import LogicalPlanner
from .pushdown import DataFramePushdownBackend, ExecutionResult, PushdownBackend, SQLPushdownBackend
from .python_analytics import PythonAnalyticsExecutor
from .result_validator import validate_result_schema
from .router import (
    SourceProfile,
    apply_route_decision_to_artifact,
    build_routing_artifact,
    route_matches_expectation,
    route_plan,
)


@dataclass
class EngineRun:
    plan: LogicalPlan
    route: str
    route_reason: str
    route_scores: Dict[str, float]
    plan_valid: bool
    plan_errors: list[str]
    route_correct: bool
    execution_route: str
    result: Optional[ExecutionResult]
    schema_correct: bool
    schema_errors: list[str]
    safety_pass: bool
    fallback_used: bool
    fallback_reason: str
    backend: str
    rows_scanned: int
    rows_materialized: int
    bytes_fetched: int
    peak_memory_mb: float
    routing_artifact: Dict[str, Any]


class MixedExecutionEngine:
    def __init__(self) -> None:
        self.planner = LogicalPlanner()
        self.sql_backend = SQLPushdownBackend()
        self.df_pushdown_backend = DataFramePushdownBackend()
        self.python_executor = PythonAnalyticsExecutor()

    def run(
        self,
        question: str,
        dataset_path: str | Path,
        *,
        expected_route_family: Optional[str] = None,
        header_confidence: float = 1.0,
        source_name: str = "data",
    ) -> EngineRun:
        path = Path(dataset_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        source_df = pd.read_csv(path)
        conn, schema = load_csv(path)
        routing_artifact = build_routing_artifact(
            question=question,
            schema=schema,
            header_confidence=header_confidence,
        )
        try:
            intent = self.planner.plan(question, schema, source_name=source_name)
            plan = intent.plan

            validated = validate_logical_plan(plan, schema)
            plan_valid = validated.ok
            fallback_used = bool(plan.metadata.get("fallback_used", False))
            fallback_reason = str(plan.metadata.get("fallback_reason", ""))
            if not plan_valid:
                artifact_with_route = apply_route_decision_to_artifact(
                    routing_artifact,
                    route="rejected",
                    route_reason="invalid_plan",
                )
                return EngineRun(
                    plan=plan,
                    route="rejected",
                    route_reason="invalid_plan",
                    route_scores={},
                    plan_valid=False,
                    plan_errors=validated.errors,
                    route_correct=False,
                    execution_route="none",
                    result=None,
                    schema_correct=False,
                    schema_errors=["execution skipped due to invalid plan"],
                    safety_pass=False,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason,
                    backend="none",
                    rows_scanned=0,
                    rows_materialized=0,
                    bytes_fetched=0,
                    peak_memory_mb=0.0,
                    routing_artifact=artifact_with_route,
                )

            profile = SourceProfile(
                rows_estimate=len(source_df),
                is_remote=False,
                header_confidence=header_confidence,
            )
            decision = route_plan(plan, profile)
            route = decision.route
            execution_route = route
            reason = decision.reason
            scores = decision.scores
            working_df = source_df

            if route == "cleaning_first":
                working_df = self._run_cleaning_stage(source_df)
                reroute = route_plan(
                    plan,
                    SourceProfile(
                        rows_estimate=len(working_df),
                        is_remote=False,
                        header_confidence=1.0,
                    ),
                )
                execution_route = reroute.route
                reason = f"cleaning_then_{reroute.reason}"
                scores = reroute.scores

            artifact_with_route = apply_route_decision_to_artifact(
                routing_artifact,
                route=route,
                route_reason=reason,
            )

            if execution_route == "pushdown":
                push_backend = self._choose_pushdown_backend(profile)
                backend_input = conn if isinstance(push_backend, SQLPushdownBackend) else source_df
                exec_result = push_backend.execute(plan, backend_input, source_rows=len(source_df))
                backend = exec_result.backend
            elif execution_route == "python":
                exec_result = self.python_executor.execute(plan, working_df, rows_scanned=len(working_df))
                backend = exec_result.backend
            elif execution_route == "hybrid":
                pre_plan = LogicalPlan(
                    source=plan.source,
                    projection=plan.projection,
                    filters=plan.filters,
                    group_by=plan.group_by,
                    aggregates=plan.aggregates,
                    order_by=plan.order_by,
                    limit=plan.limit,
                    post_processing=[],
                    expected_output_schema=[],
                    metadata=plan.metadata,
                )
                push_backend = self._choose_pushdown_backend(profile)
                if push_backend.supports(pre_plan):
                    backend_input = conn if isinstance(push_backend, SQLPushdownBackend) else source_df
                    pre_result = push_backend.execute(pre_plan, backend_input, source_rows=len(source_df))
                    post_plan = LogicalPlan(
                        source=plan.source,
                        projection=list(pre_result.dataframe.columns),
                        filters=[],
                        group_by=[],
                        aggregates=[],
                        order_by=plan.order_by,
                        limit=plan.limit,
                        post_processing=plan.post_processing,
                        expected_output_schema=plan.expected_output_schema,
                        metadata=plan.metadata,
                    )
                    exec_result = self.python_executor.execute(
                        post_plan, pre_result.dataframe, rows_scanned=pre_result.rows_scanned
                    )
                    exec_result = ExecutionResult(
                        dataframe=exec_result.dataframe,
                        rows_scanned=pre_result.rows_scanned,
                        rows_materialized=exec_result.rows_materialized,
                        bytes_fetched=pre_result.bytes_fetched + exec_result.bytes_fetched,
                        backend=f"hybrid({pre_result.backend}+python)",
                    )
                    backend = exec_result.backend
                else:
                    exec_result = self.python_executor.execute(plan, working_df, rows_scanned=len(working_df))
                    backend = f"hybrid_fallback({exec_result.backend})"
            else:
                exec_result = self.python_executor.execute(plan, working_df, rows_scanned=len(working_df))
                backend = f"fallback({exec_result.backend})"

            repaired_df = self._repair_output_schema(plan, exec_result.dataframe)
            if repaired_df is not exec_result.dataframe:
                exec_result = ExecutionResult(
                    dataframe=repaired_df,
                    rows_scanned=exec_result.rows_scanned,
                    rows_materialized=len(repaired_df),
                    bytes_fetched=int(repaired_df.memory_usage(index=False, deep=True).sum()),
                    backend=exec_result.backend,
                )

            schema_validation = validate_result_schema(plan, exec_result.dataframe)
            route_ok = route_matches_expectation(route, expected_route_family)
            safety_pass = plan_valid and (route != "rejected")
            peak_memory_mb = float(exec_result.bytes_fetched) / (1024.0 * 1024.0)

            return EngineRun(
                plan=plan,
                route=route,
                route_reason=reason,
                route_scores=scores,
                plan_valid=plan_valid,
                plan_errors=validated.errors,
                route_correct=route_ok,
                execution_route=execution_route,
                result=exec_result,
                schema_correct=schema_validation.schema_correct,
                schema_errors=schema_validation.errors,
                safety_pass=safety_pass,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                backend=backend,
                rows_scanned=exec_result.rows_scanned,
                rows_materialized=exec_result.rows_materialized,
                bytes_fetched=exec_result.bytes_fetched,
                peak_memory_mb=peak_memory_mb,
                routing_artifact=artifact_with_route,
            )
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _choose_pushdown_backend(self, profile: SourceProfile) -> PushdownBackend:
        # Prefer non-SQL local pushdown for local/smaller sources. SQL remains
        # preferred for remote or very large sources where database pushdown is
        # generally the safer first choice.
        if profile.is_remote or profile.rows_estimate >= 50_000:
            return self.sql_backend
        return self.df_pushdown_backend

    @staticmethod
    def _run_cleaning_stage(df: pd.DataFrame) -> pd.DataFrame:
        """Deterministic high-confidence cleaning stage placeholder.

        Current benchmark-scope behavior is conservative:
        - remove fully-empty columns
        - preserve all non-empty source columns and rows
        """
        out = df.copy()
        drop_cols = [c for c in out.columns if out[c].isna().all()]
        if drop_cols:
            out = out.drop(columns=drop_cols)
        return out

    @staticmethod
    def _repair_output_schema(plan: LogicalPlan, df: pd.DataFrame) -> pd.DataFrame:
        expected_names = {c.name for c in plan.expected_output_schema}
        if not expected_names:
            return df

        out = df.copy()
        cols = set(out.columns)
        for expected in expected_names:
            if expected in cols:
                continue
            if expected in {"day", "date"}:
                day_candidates = [c for c in out.columns if c.lower().endswith("_day") or c.lower() == "day"]
                if len(day_candidates) == 1 and day_candidates[0] not in expected_names:
                    out = out.rename(columns={day_candidates[0]: expected})
                    cols = set(out.columns)
                    continue
            # Allow exact-insensitive rename.
            insensitive = [c for c in out.columns if str(c).lower() == expected.lower()]
            if len(insensitive) == 1:
                out = out.rename(columns={insensitive[0]: expected})
                cols = set(out.columns)
        return out
