"""Central-LLM orchestration benchmark (local-only).

This evaluates coordinator behavior, not executor intelligence:
- central output is orchestration JSON only
- workers run typed contracts using opaque handles
- scoring emphasizes task classification, worker choice, sequence, handle discipline,
  payload discipline, safety, and final correctness
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.open_data_sql_vs_python_eval import DEFAULT_REPORT_DIR, VALIDATORS  # noqa: E402
from src.orchestration import (  # noqa: E402
    CentralCoordinator,
    HandleStore,
    SourceManifest,
    default_capability_manifest,
)

DEFAULT_CASES = REPO_ROOT / "eval" / "golden" / "open_data" / "orchestration_cases.json"


@dataclass(frozen=True)
class OrchestrationCase:
    id: str
    category: str
    dataset: str
    question: str
    validator: str
    session_id: str
    expected_task_class: str
    expected_workers: List[str]
    expected_route_family: Optional[str]
    payload_budget: Dict[str, int]
    header_confidence: float
    followup_from: Optional[str]
    expect_rejected: bool

    @staticmethod
    def from_dict(payload: Dict[str, Any], index: int) -> "OrchestrationCase":
        if not isinstance(payload, dict):
            raise ValueError(f"cases[{index}] must be an object")

        def _req_str(key: str) -> str:
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"cases[{index}].{key} must be a non-empty string")
            return value.strip()

        cid = _req_str("id")
        category = _req_str("category")
        allowed_categories = {"pushdown", "hybrid", "python_first", "cleaning_first", "follow_up", "adversarial"}
        if category not in allowed_categories:
            raise ValueError(f"cases[{index}].category must be one of {sorted(allowed_categories)}")

        dataset = _req_str("dataset")
        question = _req_str("question")
        validator = _req_str("validator")
        if validator not in VALIDATORS:
            raise ValueError(f"cases[{index}].validator is unknown: {validator!r}")

        session_id = _req_str("session_id")
        expected_task_class = _req_str("expected_task_class")

        expected_workers = payload.get("expected_workers")
        if not isinstance(expected_workers, list) or not expected_workers or not all(isinstance(x, str) for x in expected_workers):
            raise ValueError(f"cases[{index}].expected_workers must be a non-empty string array")

        expected_route_family = payload.get("expected_route_family")
        if expected_route_family is not None and expected_route_family not in {"pushdown", "hybrid_or_python", "cleaning_first"}:
            raise ValueError(f"cases[{index}].expected_route_family invalid")

        payload_budget = payload.get("payload_budget") or {}
        if not isinstance(payload_budget, dict):
            raise ValueError(f"cases[{index}].payload_budget must be an object")
        max_bytes = int(payload_budget.get("max_bytes_materialized", 1000000000))

        header_confidence = float(payload.get("header_confidence", 1.0))
        if header_confidence < 0.0 or header_confidence > 1.0:
            raise ValueError(f"cases[{index}].header_confidence must be in [0,1]")

        followup_from = payload.get("followup_from")
        if followup_from is not None and not isinstance(followup_from, str):
            raise ValueError(f"cases[{index}].followup_from must be string when provided")

        expect_rejected = bool(payload.get("expect_rejected", False))

        return OrchestrationCase(
            id=cid,
            category=category,
            dataset=dataset,
            question=question,
            validator=validator,
            session_id=session_id,
            expected_task_class=expected_task_class,
            expected_workers=list(expected_workers),
            expected_route_family=expected_route_family,
            payload_budget={"max_bytes_materialized": max_bytes},
            header_confidence=header_confidence,
            followup_from=followup_from,
            expect_rejected=expect_rejected,
        )


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_cases(path: Path) -> List[OrchestrationCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("cases file must be a JSON array")
    cases = [OrchestrationCase.from_dict(item, i) for i, item in enumerate(raw)]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in orchestration cases")
    return cases


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)

    def ratio(key: str) -> float:
        return (sum(1 for r in results if r.get(key)) / total) if total else 0.0

    payload = [int(r.get("total_bytes_materialized", 0)) for r in results]

    return {
        "cases_total": total,
        "task_classification_correct_rate": ratio("task_classification_correct"),
        "worker_selection_correct_rate": ratio("worker_selection_correct"),
        "sequence_correct_rate": ratio("sequence_correct"),
        "handle_valid_rate": ratio("handle_valid"),
        "payload_pass_rate": ratio("payload_pass"),
        "safety_pass_rate": ratio("safety_pass"),
        "final_output_correct_rate": ratio("final_output_correct"),
        "overall_pass_rate": ratio("overall_pass"),
        "fallback_rate": ratio("fallback_used"),
        "payload_bytes_avg": (sum(payload) / len(payload)) if payload else 0.0,
        "payload_bytes_p50": (sorted(payload)[len(payload) // 2]) if payload else 0.0,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args(argv)

    if not args.cases.exists():
        raise SystemExit(f"cases file not found: {args.cases}")

    cases = load_cases(args.cases)
    capability_manifest = default_capability_manifest()
    coordinator = CentralCoordinator()

    session_ctx: Dict[str, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []

    for case in cases:
        dataset_path = Path(case.dataset)
        if not dataset_path.is_absolute():
            dataset_path = REPO_ROOT / dataset_path
        if not dataset_path.exists():
            raise SystemExit(f"dataset not found: {case.dataset}")

        ctx = session_ctx.get(case.session_id)
        if ctx is None:
            source_df = pd.read_csv(dataset_path)
            source_manifest = SourceManifest(
                source_id=f"src_{case.session_id}",
                name="data",
                kind="csv",
                path=str(dataset_path),
                schema={c: "text" for c in source_df.columns},
                rows_estimate=int(len(source_df)),
                header_confidence=case.header_confidence,
            )
            store = HandleStore()
            source_handle = store.put(
                source_df,
                handle_type="source",
                source_id=source_manifest.source_id,
                metadata={
                    "dataset_path": str(dataset_path),
                    "header_confidence": case.header_confidence,
                },
            )
            source_handle_id = source_handle.handle_id
            ctx = {
                "source_df": source_df,
                "source_manifest": source_manifest,
                "handle_store": store,
                "source_handle": source_handle_id,
                "last_result_handle": None,
                "case_to_handle": {},
            }
            session_ctx[case.session_id] = ctx
        else:
            source_df = ctx["source_df"]
            source_manifest = ctx["source_manifest"]
            store = ctx["handle_store"]
            source_handle_id = ctx["source_handle"]

        if case.followup_from:
            prior_handle = ctx["case_to_handle"].get(case.followup_from)
        else:
            prior_handle = ctx.get("last_result_handle")

        plan = coordinator.build_plan(
            question=case.question,
            source_handle=source_handle_id,
            validator=case.validator,
            header_confidence=case.header_confidence,
            prior_result_handle=prior_handle,
        )

        run = coordinator.execute_plan(
            plan=plan,
            question=case.question,
            source_manifest=source_manifest,
            capability_manifest=capability_manifest,
            handle_store=store,
            source_df=source_df,
            source_handle=source_handle_id,
            payload_budget=case.payload_budget,
            expected_task_class=case.expected_task_class,
            expected_workers=case.expected_workers,
            validator=case.validator,
        )

        continuity_ok = True
        if case.category == "follow_up" and case.followup_from:
            expected_handle = ctx["case_to_handle"].get(case.followup_from)
            first_input = run.hops[0]["input_handle"][0] if run.hops and run.hops[0]["input_handle"] else None
            continuity_ok = bool(expected_handle and first_input == expected_handle)

        final_output_correct = run.final_output_correct
        if case.expect_rejected:
            final_output_correct = bool(run.final_error and run.final_error.error_code == "safety_violation")

        handle_valid = run.handle_valid and continuity_ok
        fallback_used = any(bool(h.get("fallback_reason")) for h in run.hops)

        overall_pass = bool(
            run.task_classification_correct
            and run.worker_selection_correct
            and run.sequence_correct
            and handle_valid
            and run.payload_pass
            and run.safety_pass
            and final_output_correct
        )

        results.append(
            {
                "id": case.id,
                "category": case.category,
                "session_id": case.session_id,
                "dataset": case.dataset,
                "question": case.question,
                "validator": case.validator,
                "expected_task_class": case.expected_task_class,
                "expected_workers": case.expected_workers,
                "plan": plan.to_dict(),
                "task_classification_correct": run.task_classification_correct,
                "worker_selection_correct": run.worker_selection_correct,
                "sequence_correct": run.sequence_correct,
                "handle_valid": handle_valid,
                "payload_pass": run.payload_pass,
                "safety_pass": run.safety_pass,
                "final_output_correct": final_output_correct,
                "overall_pass": overall_pass,
                "hops": run.hops,
                "total_bytes_materialized": run.total_bytes_materialized,
                "payload_budget": case.payload_budget,
                "fallback_used": fallback_used,
                "final_handle": run.final_handle,
                "final_error": run.final_error.to_dict() if run.final_error else None,
                "followup_from": case.followup_from,
                "continuity_ok": continuity_ok,
                "expect_rejected": case.expect_rejected,
            }
        )

        if run.final_handle:
            ctx["last_result_handle"] = run.final_handle
            ctx["case_to_handle"][case.id] = run.final_handle

    summary = _aggregate(results)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(args.cases),
        "summary": summary,
        "results": results,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"open_data_orchestration_{_utc_stamp()}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "report_path": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
