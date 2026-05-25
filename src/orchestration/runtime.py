from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.ingestion import infer_schema
from src.mixed_execution import MixedExecutionEngine

from .validation import validate_orchestration_plan

OPAQUE_HANDLE_RE = re.compile(r"^hdl_[a-f0-9]{12}$")

TASK_CLASSES = {
    "pushdown",
    "hybrid",
    "python_first",
    "cleaning_first",
    "follow_up",
    "adversarial",
}


@dataclass(frozen=True)
class SourceManifest:
    source_id: str
    name: str
    kind: str
    path: str
    schema: Dict[str, str]
    rows_estimate: int
    header_confidence: float = 1.0


@dataclass(frozen=True)
class WorkerCapability:
    worker_id: str
    allowed_task_classes: List[str]
    accepted_handle_types: List[str]
    output_handle_type: str


@dataclass(frozen=True)
class CapabilityManifest:
    manifest_id: str
    workers: List[WorkerCapability]

    def worker_ids(self) -> set[str]:
        return {w.worker_id for w in self.workers}

    def by_id(self) -> Dict[str, WorkerCapability]:
        return {w.worker_id: w for w in self.workers}


@dataclass(frozen=True)
class DataHandle:
    handle_id: str
    handle_type: str
    source_id: str
    rows_estimate: int
    bytes_estimate: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanStep:
    worker_id: str
    input_handles: List[str]
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestrationPlan:
    task_class: str
    steps: List[PlanStep]
    final_output_handle: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_class": self.task_class,
            "steps": [
                {
                    "worker_id": s.worker_id,
                    "input_handles": list(s.input_handles),
                    "params": dict(s.params),
                }
                for s in self.steps
            ],
            "final_output_handle": self.final_output_handle,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TypedErrorResult:
    error_code: str
    message: str
    worker_id: str
    retryable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "worker_id": self.worker_id,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class WorkerResult:
    ok: bool
    worker_id: str
    output_handle: Optional[str]
    bytes_materialized: int
    # validation_result is worker-contract validity for this hop.
    # For executor hops, schema_validation_result captures output-schema checks separately.
    validation_result: bool
    validation_scope: str
    schema_validation_result: Optional[bool]
    fallback_reason: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[TypedErrorResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "worker_id": self.worker_id,
            "output_handle": self.output_handle,
            "bytes_materialized": self.bytes_materialized,
            "validation_result": self.validation_result,
            "validation_scope": self.validation_scope,
            "schema_validation_result": self.schema_validation_result,
            "fallback_reason": self.fallback_reason,
            "summary": self.summary,
            "details": dict(self.details),
            "error": self.error.to_dict() if self.error else None,
        }


@dataclass(frozen=True)
class CoordinatorRun:
    plan: OrchestrationPlan
    task_classification_correct: bool
    worker_selection_correct: bool
    sequence_correct: bool
    handle_valid: bool
    payload_pass: bool
    safety_pass: bool
    final_output_correct: bool
    total_bytes_materialized: int
    hops: List[Dict[str, Any]]
    final_handle: Optional[str]
    final_error: Optional[TypedErrorResult]


class HandleStore:
    def __init__(self) -> None:
        self._values: Dict[str, Any] = {}
        self._meta: Dict[str, DataHandle] = {}
        self._counter = 0

    def put(
        self,
        value: Any,
        *,
        handle_type: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataHandle:
        self._counter += 1
        handle_id = f"hdl_{self._counter:012x}"
        rows_estimate = 0
        bytes_estimate = 0
        if isinstance(value, pd.DataFrame):
            rows_estimate = int(len(value))
            bytes_estimate = int(value.memory_usage(index=False, deep=True).sum())
        elif isinstance(value, dict):
            bytes_estimate = len(str(value).encode("utf-8"))

        meta = DataHandle(
            handle_id=handle_id,
            handle_type=handle_type,
            source_id=source_id,
            rows_estimate=rows_estimate,
            bytes_estimate=bytes_estimate,
            metadata=dict(metadata or {}),
        )
        self._values[handle_id] = value
        self._meta[handle_id] = meta
        return meta

    def has(self, handle_id: str) -> bool:
        return handle_id in self._values

    def get_meta(self, handle_id: str) -> DataHandle:
        return self._meta[handle_id]

    def get_value(self, handle_id: str) -> Any:
        return self._values[handle_id]


def _validators():
    """Lazy import so eval module is only needed when validators are actually called."""
    from eval.open_data_sql_vs_python_eval import VALIDATORS  # noqa: PLC0415
    return VALIDATORS


class CentralCoordinator:
    def __init__(self, *, engine: Optional[MixedExecutionEngine] = None) -> None:
        self.engine = engine or MixedExecutionEngine()

    @staticmethod
    def classify_task(question: str, header_confidence: float, has_prior_handle: bool) -> str:
        q = (question or "").strip().lower()
        if any(tok in q for tok in ["drop table", "delete from", "union select", "or 1=1", "exec(", "--"]):
            return "adversarial"
        if has_prior_handle and any(tok in q for tok in ["that result", "previous result", "same result", "from that"]):
            return "follow_up"
        if header_confidence < 0.9 or "header" in q or "clean" in q:
            return "cleaning_first"
        if any(tok in q for tok in ["rolling", "moving average"]) and any(tok in q for tok in ["last", "per day", "date"]):
            return "hybrid"
        if any(tok in q for tok in ["percentile", "quantile", "stdev", "standard deviation", "outlier", "anomaly"]):
            return "python_first"
        return "pushdown"

    def build_plan(
        self,
        *,
        question: str,
        source_handle: str,
        validator: str,
        header_confidence: float,
        prior_result_handle: Optional[str],
    ) -> OrchestrationPlan:
        task_class = self.classify_task(question, header_confidence, bool(prior_result_handle))

        if task_class == "adversarial":
            return OrchestrationPlan(task_class=task_class, steps=[PlanStep(worker_id="reject_worker", input_handles=[], params={})])

        if task_class == "follow_up":
            if not prior_result_handle:
                return OrchestrationPlan(
                    task_class=task_class,
                    steps=[PlanStep(worker_id="reject_worker", input_handles=[], params={"reason": "missing_prior_handle"})],
                )
            return OrchestrationPlan(
                task_class=task_class,
                steps=[
                    PlanStep(
                        worker_id="followup_worker",
                        input_handles=[prior_result_handle],
                        params={"question": question},
                    ),
                    PlanStep(
                        worker_id="validator_worker",
                        input_handles=["$prev"],
                        params={"validator": validator},
                    ),
                ],
            )

        if task_class == "cleaning_first":
            return OrchestrationPlan(
                task_class=task_class,
                steps=[
                    PlanStep(worker_id="cleaning_worker", input_handles=[source_handle], params={}),
                    PlanStep(
                        worker_id="mixed_executor_worker",
                        input_handles=["$prev"],
                        params={
                            "question": question,
                            "expected_route_family": "cleaning_first",
                        },
                    ),
                    PlanStep(
                        worker_id="validator_worker",
                        input_handles=["$prev"],
                        params={"validator": validator},
                    ),
                ],
            )

        route_family = "pushdown" if task_class == "pushdown" else "hybrid_or_python"
        return OrchestrationPlan(
            task_class=task_class,
            steps=[
                PlanStep(
                    worker_id="mixed_executor_worker",
                    input_handles=[source_handle],
                    params={
                        "question": question,
                        "expected_route_family": route_family,
                    },
                ),
                PlanStep(
                    worker_id="validator_worker",
                    input_handles=["$prev"],
                    params={"validator": validator},
                ),
            ],
        )

    def execute_plan(
        self,
        *,
        plan: OrchestrationPlan,
        question: str,
        source_manifest: SourceManifest,
        capability_manifest: CapabilityManifest,
        handle_store: HandleStore,
        source_df: pd.DataFrame,
        source_handle: str,
        payload_budget: Dict[str, int],
        expected_task_class: str,
        expected_workers: List[str],
        validator: str,
    ) -> CoordinatorRun:
        plan_errors = validate_orchestration_plan(plan, capability_manifest)
        hops: List[Dict[str, Any]] = []
        last_output_handle: Optional[str] = None
        total_bytes = 0
        final_error: Optional[TypedErrorResult] = None
        safety_pass = not plan_errors
        handle_valid = True
        final_output_correct = False

        if plan_errors:
            final_error = TypedErrorResult(
                error_code="invalid_plan",
                message="; ".join(plan_errors),
                worker_id="coordinator",
                retryable=False,
            )
        else:
            for step in plan.steps:
                resolved_inputs: List[str] = []
                for hid in step.input_handles:
                    if hid == "$prev":
                        if last_output_handle is None:
                            handle_valid = False
                            final_error = TypedErrorResult(
                                error_code="invalid_handle",
                                message="$prev used before any worker output",
                                worker_id=step.worker_id,
                                retryable=False,
                            )
                            break
                        hid = last_output_handle
                    if not OPAQUE_HANDLE_RE.match(hid):
                        handle_valid = False
                    if not handle_store.has(hid):
                        handle_valid = False
                        final_error = TypedErrorResult(
                            error_code="invalid_handle",
                            message=f"unknown handle {hid}",
                            worker_id=step.worker_id,
                            retryable=False,
                        )
                        break
                    resolved_inputs.append(hid)
                if final_error is not None:
                    break

                result = self._invoke_worker(
                    worker_id=step.worker_id,
                    params=step.params,
                    input_handles=resolved_inputs,
                    handle_store=handle_store,
                    source_manifest=source_manifest,
                    source_df=source_df,
                    source_handle=source_handle,
                    question=question,
                    validator=validator,
                )
                total_bytes += int(result.bytes_materialized)
                if result.output_handle is not None:
                    last_output_handle = result.output_handle

                hops.append(
                    {
                        "chosen_worker": step.worker_id,
                        "input_handle": resolved_inputs,
                        "output_handle": result.output_handle,
                        "bytes_materialized": int(result.bytes_materialized),
                        "validation_result": bool(result.validation_result),
                        "validation_scope": result.validation_scope,
                        "schema_validation_result": result.schema_validation_result,
                        "fallback_reason": result.fallback_reason,
                        "details": dict(result.details),
                    }
                )

                if not result.ok:
                    final_error = result.error
                    break

            if final_error is None and last_output_handle is not None:
                out_df = handle_store.get_value(last_output_handle)
                if isinstance(out_df, pd.DataFrame):
                    try:
                        final_output_correct = _validators()[validator](out_df, source_df)[0]
                    except Exception:
                        final_output_correct = False

        max_bytes = int(payload_budget.get("max_bytes_materialized", 1_000_000_000))
        payload_pass = total_bytes <= max_bytes

        task_classification_correct = plan.task_class == expected_task_class
        planned_workers = [s.worker_id for s in plan.steps]
        worker_selection_correct = set(planned_workers) == set(expected_workers)
        sequence_correct = planned_workers == expected_workers

        return CoordinatorRun(
            plan=plan,
            task_classification_correct=task_classification_correct,
            worker_selection_correct=worker_selection_correct,
            sequence_correct=sequence_correct,
            handle_valid=handle_valid,
            payload_pass=payload_pass,
            safety_pass=safety_pass,
            final_output_correct=final_output_correct,
            total_bytes_materialized=total_bytes,
            hops=hops,
            final_handle=last_output_handle,
            final_error=final_error,
        )

    def _invoke_worker(
        self,
        *,
        worker_id: str,
        params: Dict[str, Any],
        input_handles: List[str],
        handle_store: HandleStore,
        source_manifest: SourceManifest,
        source_df: pd.DataFrame,
        source_handle: str,
        question: str,
        validator: str,
    ) -> WorkerResult:
        if worker_id == "reject_worker":
            err = TypedErrorResult(
                error_code="safety_violation",
                message="request rejected by coordinator safety policy",
                worker_id=worker_id,
                retryable=False,
            )
            return WorkerResult(
                ok=False,
                worker_id=worker_id,
                output_handle=None,
                bytes_materialized=0,
                validation_result=True,
                validation_scope="policy",
                schema_validation_result=None,
                fallback_reason="adversarial_or_unsafe",
                summary="rejected",
                error=err,
            )

        if worker_id == "cleaning_worker":
            in_df = handle_store.get_value(input_handles[0])
            if not isinstance(in_df, pd.DataFrame):
                err = TypedErrorResult("worker_failed", "cleaning input must be dataframe", worker_id, False)
                return WorkerResult(False, worker_id, None, 0, False, "contract", None, "invalid_input", "cleaning failed", err)

            artifact = self._build_cleaning_artifact(in_df)
            out_handle = handle_store.put(
                artifact,
                handle_type="cleaned_source",
                source_id=source_manifest.source_id,
                metadata={
                    "dataset_path": source_manifest.path,
                    "header_confidence": source_manifest.header_confidence,
                    "artifact_kind": "cleaning_metadata",
                    "row_offset": artifact.get("row_offset", 0),
                    "normalized_schema": artifact.get("normalized_schema", {}),
                    "header_map": artifact.get("header_map", {}),
                },
            )
            return WorkerResult(
                ok=True,
                worker_id=worker_id,
                output_handle=out_handle.handle_id,
                bytes_materialized=out_handle.bytes_estimate,
                validation_result=True,
                validation_scope="contract",
                schema_validation_result=None,
                fallback_reason="",
                summary="cleaning metadata artifact emitted",
            )

        if worker_id == "mixed_executor_worker":
            in_meta = handle_store.get_meta(input_handles[0])
            ask = str(params.get("question") or question)
            expected_route_family = str(params.get("expected_route_family") or "hybrid_or_python")
            try:
                run = self.engine.run(
                    question=ask,
                    dataset_path=Path(source_manifest.path),
                    expected_route_family=expected_route_family,
                    header_confidence=float(in_meta.metadata.get("header_confidence", source_manifest.header_confidence)),
                    source_name="data",
                )
            except Exception as exc:
                err = TypedErrorResult("worker_failed", str(exc), worker_id, False)
                return WorkerResult(False, worker_id, None, 0, False, "contract", None, "engine_exception", "execution failed", err)

            if run.result is None:
                err = TypedErrorResult("worker_failed", "mixed executor produced no result", worker_id, False)
                return WorkerResult(False, worker_id, None, 0, False, "contract", None, "empty_result", "execution failed", err)

            out_handle = handle_store.put(
                run.result.dataframe,
                handle_type="result_table",
                source_id=source_manifest.source_id,
                metadata={
                    "route": run.route,
                    "execution_route": run.execution_route,
                    "route_reason": run.route_reason,
                    "route_scores": run.route_scores,
                    "routing_artifact": run.routing_artifact,
                    "backend": run.backend,
                    "schema_correct": run.schema_correct,
                },
            )
            return WorkerResult(
                ok=run.plan_valid and run.safety_pass,
                worker_id=worker_id,
                output_handle=out_handle.handle_id,
                bytes_materialized=int(run.bytes_fetched),
                validation_result=bool(run.plan_valid and run.safety_pass),
                validation_scope="contract",
                schema_validation_result=bool(run.schema_correct),
                fallback_reason=str(run.fallback_reason or ""),
                summary=f"route={run.route}; backend={run.backend}",
                details={
                    "route": run.route,
                    "execution_route": run.execution_route,
                    "route_reason": run.route_reason,
                    "route_scores": run.route_scores,
                    "routing_artifact": run.routing_artifact,
                },
                error=None if (run.plan_valid and run.safety_pass) else TypedErrorResult(
                    "worker_failed",
                    "plan invalid or safety failed",
                    worker_id,
                    False,
                ),
            )

        if worker_id == "followup_worker":
            in_df = handle_store.get_value(input_handles[0])
            if not isinstance(in_df, pd.DataFrame):
                err = TypedErrorResult("worker_failed", "follow-up input must be dataframe", worker_id, False)
                return WorkerResult(False, worker_id, None, 0, False, "contract", None, "invalid_input", "follow-up failed", err)
            q = (str(params.get("question") or "")).lower()
            out_df = in_df.copy()
            if "count" in q or "how many" in q:
                out_df = pd.DataFrame([{"row_count": int(len(in_df))}])
            elif "top" in q:
                match = re.search(r"top\s+(\d+)", q)
                n = int(match.group(1)) if match else 5
                numeric_cols = [c for c in out_df.columns if pd.to_numeric(out_df[c], errors="coerce").notna().mean() >= 0.8]
                if numeric_cols:
                    out_df = out_df.sort_values(numeric_cols[0], ascending=False).head(n)
                else:
                    out_df = out_df.head(n)
            elif "first" in q:
                match = re.search(r"first\s+(\d+)", q)
                n = int(match.group(1)) if match else 5
                out_df = out_df.head(n)
            elif "average" in q or "mean" in q:
                numeric_cols = [c for c in out_df.columns if pd.to_numeric(out_df[c], errors="coerce").notna().mean() >= 0.8]
                if numeric_cols:
                    value = float(pd.to_numeric(out_df[numeric_cols[0]], errors="coerce").mean())
                    out_df = pd.DataFrame([{"mean_value": value}])
                else:
                    out_df = pd.DataFrame([{"row_count": int(len(out_df))}])
            else:
                out_df = out_df.head(5)

            out_df = out_df.reset_index(drop=True)
            out_handle = handle_store.put(
                out_df,
                handle_type="result_table",
                source_id=source_manifest.source_id,
                metadata={"derived_from": input_handles[0]},
            )
            return WorkerResult(
                ok=True,
                worker_id=worker_id,
                output_handle=out_handle.handle_id,
                bytes_materialized=out_handle.bytes_estimate,
                validation_result=True,
                validation_scope="contract",
                schema_validation_result=None,
                fallback_reason="",
                summary="follow-up operation complete",
            )

        if worker_id == "validator_worker":
            in_df = handle_store.get_value(input_handles[0])
            if not isinstance(in_df, pd.DataFrame):
                err = TypedErrorResult("worker_failed", "validator input must be dataframe", worker_id, False)
                return WorkerResult(False, worker_id, None, 0, False, "validator", None, "invalid_input", "validation failed", err)
            validator_name = str(params.get("validator") or validator)
            validators = _validators()
            if validator_name not in validators:
                err = TypedErrorResult("worker_failed", f"unknown validator {validator_name}", worker_id, False)
                return WorkerResult(False, worker_id, None, 0, False, "validator", None, "unknown_validator", "validation failed", err)
            try:
                ok, note = validators[validator_name](in_df, source_df)
            except Exception as exc:
                ok, note = False, f"validator raised: {exc}"
            return WorkerResult(
                ok=bool(ok),
                worker_id=worker_id,
                output_handle=input_handles[0],
                bytes_materialized=0,
                validation_result=bool(ok),
                validation_scope="validator",
                schema_validation_result=None,
                fallback_reason="" if ok else "validator_failed",
                summary=note,
                error=None if ok else TypedErrorResult("worker_failed", note, worker_id, False),
            )

        err = TypedErrorResult("worker_failed", f"unknown worker {worker_id}", worker_id, False)
        return WorkerResult(False, worker_id, None, 0, False, "contract", None, "unknown_worker", "worker missing", err)

    @staticmethod
    def _build_cleaning_artifact(df: pd.DataFrame) -> Dict[str, Any]:
        # Emit a lightweight cleaning artifact instead of materializing a full
        # cleaned table copy; this keeps cleaning-first payload costs bounded.
        non_empty_cols = [c for c in df.columns if df[c].notna().any()]
        header_map = {str(col): str(col) for col in non_empty_cols}
        dropped_empty_columns = [str(c) for c in df.columns if c not in non_empty_cols]

        if non_empty_cols:
            normalized_schema = infer_schema(df[non_empty_cols])
            non_null_counts = df[non_empty_cols].notna().sum(axis=1)
            threshold = max(1, len(non_empty_cols) // 2)
            candidate_rows = non_null_counts[non_null_counts >= threshold]
            row_offset = int(candidate_rows.index[0]) if not candidate_rows.empty else 0
        else:
            normalized_schema = {}
            row_offset = 0

        return {
            "artifact_version": "v1",
            "header_map": header_map,
            "normalized_schema": normalized_schema,
            "row_offset": row_offset,
            "dropped_empty_columns": dropped_empty_columns,
        }


def default_capability_manifest() -> CapabilityManifest:
    return CapabilityManifest(
        manifest_id="local_orchestration_v1",
        workers=[
            WorkerCapability(
                worker_id="cleaning_worker",
                allowed_task_classes=["cleaning_first"],
                accepted_handle_types=["source", "cleaned_source"],
                output_handle_type="cleaned_source",
            ),
            WorkerCapability(
                worker_id="mixed_executor_worker",
                allowed_task_classes=["pushdown", "hybrid", "python_first", "cleaning_first"],
                accepted_handle_types=["source", "cleaned_source"],
                output_handle_type="result_table",
            ),
            WorkerCapability(
                worker_id="followup_worker",
                allowed_task_classes=["follow_up"],
                accepted_handle_types=["result_table"],
                output_handle_type="result_table",
            ),
            WorkerCapability(
                worker_id="validator_worker",
                allowed_task_classes=["pushdown", "hybrid", "python_first", "cleaning_first", "follow_up"],
                accepted_handle_types=["result_table"],
                output_handle_type="result_table",
            ),
            WorkerCapability(
                worker_id="reject_worker",
                allowed_task_classes=["adversarial"],
                accepted_handle_types=[],
                output_handle_type="error",
            ),
        ],
    )
