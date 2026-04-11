"""Planner-first mixed execution benchmark for open-data cases.

Pipeline:
    NL -> typed LogicalPlan -> deterministic router -> pushdown/hybrid/python execution -> schema/result validation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.open_data_sql_vs_python_eval import (  # noqa: E402
    DEFAULT_REPORT_DIR,
    VALIDATORS,
    ProbeCase,
    load_cases,
)
from src.mixed_execution import MixedExecutionEngine  # noqa: E402

DEFAULT_MIXED_CASES = REPO_ROOT / "eval" / "golden" / "open_data" / "mixed_execution_cases.json"
DEFAULT_ROUTE_ORACLE = REPO_ROOT / "eval" / "golden" / "open_data" / "mixed_execution_route_oracle.json"
EXPECTED_ROUTE_FAMILIES = {"pushdown", "hybrid_or_python", "cleaning_first"}


@dataclass(frozen=True)
class RouteOracleCase:
    id: str
    expected_route_family: str
    header_confidence: float
    payload_budget: Dict[str, float]

    @staticmethod
    def from_dict(payload: Mapping[str, Any], index: int) -> "RouteOracleCase":
        if not isinstance(payload, Mapping):
            raise ValueError(f"route_oracle[{index}] must be an object")
        cid = str(payload.get("id") or "").strip()
        if not cid:
            raise ValueError(f"route_oracle[{index}].id must be a non-empty string")
        expected_route_family = str(payload.get("expected_route_family") or "").strip()
        if expected_route_family not in EXPECTED_ROUTE_FAMILIES:
            raise ValueError(
                f"route_oracle[{index}].expected_route_family must be one of {sorted(EXPECTED_ROUTE_FAMILIES)}"
            )
        raw_header_conf = payload.get("header_confidence", 1.0)
        try:
            header_confidence = float(raw_header_conf)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"route_oracle[{index}].header_confidence must be numeric") from exc
        if header_confidence < 0.0 or header_confidence > 1.0:
            raise ValueError(f"route_oracle[{index}].header_confidence must be in [0, 1]")

        payload_budget = payload.get("payload_budget") or {}
        if not isinstance(payload_budget, Mapping):
            raise ValueError(f"route_oracle[{index}].payload_budget must be an object")
        cleaned_budget: Dict[str, float] = {}
        for key, value in payload_budget.items():
            if key not in {"max_rows_scanned", "max_rows_materialized", "max_bytes_fetched", "max_peak_memory_mb"}:
                raise ValueError(f"route_oracle[{index}].payload_budget has unknown key {key!r}")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"route_oracle[{index}].payload_budget.{key} must be numeric") from exc
            if numeric < 0:
                raise ValueError(f"route_oracle[{index}].payload_budget.{key} must be >= 0")
            cleaned_budget[str(key)] = numeric

        return RouteOracleCase(
            id=cid,
            expected_route_family=expected_route_family,
            header_confidence=header_confidence,
            payload_budget=cleaned_budget,
        )


def load_route_oracle(path: Path) -> Dict[str, RouteOracleCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("route oracle file must be a JSON array")
    loaded = [RouteOracleCase.from_dict(item, i) for i, item in enumerate(raw)]
    by_id: Dict[str, RouteOracleCase] = {}
    for row in loaded:
        if row.id in by_id:
            raise ValueError(f"duplicate route oracle id: {row.id!r}")
        by_id[row.id] = row
    return by_id


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _preview_rows(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    return df.head(n).where(pd.notna(df.head(n)), None).to_dict(orient="records")


def _evaluate_payload_budget(
    *,
    rows_scanned: int,
    rows_materialized: int,
    bytes_fetched: int,
    peak_memory_mb: float,
    payload_budget: Mapping[str, float],
) -> Tuple[bool, List[str]]:
    violations: List[str] = []

    max_rows_scanned = payload_budget.get("max_rows_scanned")
    if max_rows_scanned is not None and rows_scanned > int(max_rows_scanned):
        violations.append(f"rows_scanned {rows_scanned} > {int(max_rows_scanned)}")

    max_rows_materialized = payload_budget.get("max_rows_materialized")
    if max_rows_materialized is not None and rows_materialized > int(max_rows_materialized):
        violations.append(f"rows_materialized {rows_materialized} > {int(max_rows_materialized)}")

    max_bytes_fetched = payload_budget.get("max_bytes_fetched")
    if max_bytes_fetched is not None and bytes_fetched > int(max_bytes_fetched):
        violations.append(f"bytes_fetched {bytes_fetched} > {int(max_bytes_fetched)}")

    max_peak_memory_mb = payload_budget.get("max_peak_memory_mb")
    if max_peak_memory_mb is not None and peak_memory_mb > float(max_peak_memory_mb):
        violations.append(f"peak_memory_mb {peak_memory_mb:.4f} > {float(max_peak_memory_mb):.4f}")

    return not violations, violations


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)

    def ratio(key: str) -> float:
        return (sum(1 for row in results if row.get(key)) / total) if total else 0.0

    latencies = [float(row.get("latency_ms", 0.0)) for row in results]
    rows_scanned = [int(row.get("rows_scanned", 0)) for row in results]
    rows_mat = [int(row.get("rows_materialized", 0)) for row in results]
    bytes_fetched = [int(row.get("bytes_fetched", 0)) for row in results]
    peak_mem = [float(row.get("peak_memory_mb", 0.0)) for row in results]

    by_route: Dict[str, int] = {}
    by_execution_route: Dict[str, int] = {}
    by_backend: Dict[str, int] = {}
    fallback_reasons: Dict[str, int] = {}
    for row in results:
        route = str(row.get("route") or "")
        by_route[route] = by_route.get(route, 0) + 1
        execution_route = str(row.get("execution_route") or "")
        by_execution_route[execution_route] = by_execution_route.get(execution_route, 0) + 1
        backend = str(row.get("backend") or "")
        by_backend[backend] = by_backend.get(backend, 0) + 1
        if row.get("fallback_used"):
            reason = str(row.get("fallback_reason") or "unspecified")
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1

    return {
        "cases_total": total,
        "plan_valid_rate": ratio("plan_valid"),
        "route_correct_rate": ratio("route_correct"),
        "execution_correct_rate": ratio("execution_correct"),
        "schema_correct_rate": ratio("schema_correct"),
        "safety_pass_rate": ratio("safety_pass"),
        "payload_pass_rate": ratio("payload_pass"),
        "overall_pass_rate": ratio("overall_pass"),
        "fallback_used_rate": ratio("fallback_used"),
        "latency_ms_avg": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "latency_ms_p50": (sorted(latencies)[len(latencies) // 2]) if latencies else 0.0,
        "rows_scanned_avg": (sum(rows_scanned) / len(rows_scanned)) if rows_scanned else 0.0,
        "rows_materialized_avg": (sum(rows_mat) / len(rows_mat)) if rows_mat else 0.0,
        "bytes_fetched_avg": (sum(bytes_fetched) / len(bytes_fetched)) if bytes_fetched else 0.0,
        "peak_memory_mb_avg": (sum(peak_mem) / len(peak_mem)) if peak_mem else 0.0,
        "route_counts": by_route,
        "execution_route_counts": by_execution_route,
        "backend_counts": by_backend,
        "safety_summary": {
            "cases_passed": sum(1 for row in results if row.get("safety_pass")),
            "cases_failed": sum(1 for row in results if not row.get("safety_pass")),
            "pass_rate": ratio("safety_pass"),
        },
        "fallback_summary": {
            "cases_with_fallback": sum(1 for row in results if row.get("fallback_used")),
            "rate": ratio("fallback_used"),
            "reasons": fallback_reasons,
        },
        "payload_summary": {
            "cases_passed": sum(1 for row in results if row.get("payload_pass")),
            "cases_failed": sum(1 for row in results if not row.get("payload_pass")),
            "pass_rate": ratio("payload_pass"),
            "rows_scanned_avg": (sum(rows_scanned) / len(rows_scanned)) if rows_scanned else 0.0,
            "rows_materialized_avg": (sum(rows_mat) / len(rows_mat)) if rows_mat else 0.0,
            "bytes_fetched_avg": (sum(bytes_fetched) / len(bytes_fetched)) if bytes_fetched else 0.0,
            "peak_memory_mb_avg": (sum(peak_mem) / len(peak_mem)) if peak_mem else 0.0,
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_MIXED_CASES)
    parser.add_argument("--route-oracle", type=Path, default=DEFAULT_ROUTE_ORACLE)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args(argv)

    if not args.cases.exists():
        raise SystemExit(f"cases file not found: {args.cases}")
    if not args.route_oracle.exists():
        raise SystemExit(f"route oracle file not found: {args.route_oracle}")

    cases = load_cases(args.cases)
    route_oracle = load_route_oracle(args.route_oracle)

    case_ids = {case.id for case in cases}
    missing_oracle = sorted(case_ids - set(route_oracle.keys()))
    if missing_oracle:
        raise SystemExit(f"route oracle missing case ids: {missing_oracle}")

    engine = MixedExecutionEngine()
    results: List[Dict[str, Any]] = []

    for case in cases:
        if case.validator not in VALIDATORS:
            raise SystemExit(f"unknown validator {case.validator!r} for case {case.id!r}")
        dataset_path = Path(case.dataset)
        if not dataset_path.is_absolute():
            dataset_path = REPO_ROOT / dataset_path
        if not dataset_path.exists():
            raise SystemExit(f"dataset not found: {case.dataset}")
        source_df = pd.read_csv(dataset_path)

        started = time.perf_counter()
        oracle = route_oracle[case.id]
        run = engine.run(
            question=case.question,
            dataset_path=dataset_path,
            expected_route_family=oracle.expected_route_family,
            header_confidence=oracle.header_confidence,
            source_name="data",
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        execution_correct = False
        validation_note = "execution skipped"
        error = None
        preview: List[Dict[str, Any]] = []
        row_count = 0
        if run.result is not None:
            row_count = int(len(run.result.dataframe))
            preview = _preview_rows(run.result.dataframe)
            try:
                execution_correct, validation_note = VALIDATORS[case.validator](run.result.dataframe, source_df)
            except Exception as exc:
                execution_correct = False
                validation_note = "validator raised exception"
                error = str(exc)

        payload_pass, payload_violations = _evaluate_payload_budget(
            rows_scanned=run.rows_scanned,
            rows_materialized=run.rows_materialized,
            bytes_fetched=run.bytes_fetched,
            peak_memory_mb=run.peak_memory_mb,
            payload_budget=oracle.payload_budget,
        )
        overall_pass = bool(
            run.plan_valid
            and run.route_correct
            and execution_correct
            and run.schema_correct
            and run.safety_pass
            and payload_pass
        )

        results.append(
            {
                "id": case.id,
                "track": case.track,
                "dataset": case.dataset,
                "question": case.question,
                "validator": case.validator,
                "expected_route_family": oracle.expected_route_family,
                "header_confidence": oracle.header_confidence,
                "route": run.route,
                "execution_route": run.execution_route,
                "route_reason": run.route_reason,
                "route_scores": run.route_scores,
                "plan": run.plan.to_dict(),
                "plan_valid": run.plan_valid,
                "plan_errors": run.plan_errors,
                "route_correct": run.route_correct,
                "execution_correct": execution_correct,
                "schema_correct": run.schema_correct,
                "schema_errors": run.schema_errors,
                "safety_pass": run.safety_pass,
                "fallback_used": run.fallback_used,
                "fallback_reason": run.fallback_reason,
                "backend": run.backend,
                "rows_scanned": run.rows_scanned,
                "rows_materialized": run.rows_materialized,
                "bytes_fetched": run.bytes_fetched,
                "peak_memory_mb": run.peak_memory_mb,
                "payload_budget": oracle.payload_budget,
                "payload_pass": payload_pass,
                "payload_violations": payload_violations,
                "latency_ms": latency_ms,
                "validation_note": validation_note,
                "result_preview": preview,
                "row_count": row_count,
                "error": error,
                "overall_pass": overall_pass,
            }
        )

    summary = _aggregate(results)
    route_recommendations = [
        {
            "id": row["id"],
            "track": row["track"],
            "recommended_route": row["route"],
            "route_reason": row["route_reason"],
            "expected_route_family": row["expected_route_family"],
            "header_confidence": row["header_confidence"],
        }
        for row in results
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(args.cases),
        "route_oracle_path": str(args.route_oracle),
        "summary": summary,
        "route_recommendations": route_recommendations,
        "results": results,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"open_data_mixed_execution_{_utc_stamp()}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "report_path": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
