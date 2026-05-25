"""Analysis-only benchmark on distilled data handles.

This benchmark starts AFTER validated distilled-data production and evaluates
analysis orchestration quality independently from extraction/mixed execution.
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

from eval.open_data_sql_vs_python_eval import DEFAULT_REPORT_DIR  # noqa: E402
from src.analysis_lane import AnalysisCoordinator, load_distilled_handles  # noqa: E402

DEFAULT_CASES = REPO_ROOT / "eval" / "golden" / "open_data" / "analysis_cases.json"
DEFAULT_HANDLES = REPO_ROOT / "eval" / "golden" / "open_data" / "analysis_distilled_handles.json"


@dataclass(frozen=True)
class AnalysisCase:
    id: str
    family: str
    handle_id: str
    question: str
    session_id: str
    followup_from: Optional[str]
    expect_guardrail: bool

    @staticmethod
    def from_dict(payload: Dict[str, Any], index: int) -> "AnalysisCase":
        if not isinstance(payload, dict):
            raise ValueError(f"cases[{index}] must be an object")

        def _req_str(key: str) -> str:
            val = payload.get(key)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"cases[{index}].{key} must be non-empty string")
            return val.strip()

        family = _req_str("family")
        allowed = {"kpi_summary", "trend", "segment_comparison", "dashboard_design", "follow_up_analysis", "guardrail"}
        if family not in allowed:
            raise ValueError(f"cases[{index}].family must be one of {sorted(allowed)}")

        followup_from = payload.get("followup_from")
        if followup_from is not None and not isinstance(followup_from, str):
            raise ValueError(f"cases[{index}].followup_from must be string or null")

        return AnalysisCase(
            id=_req_str("id"),
            family=family,
            handle_id=_req_str("handle_id"),
            question=_req_str("question"),
            session_id=_req_str("session_id"),
            followup_from=followup_from,
            expect_guardrail=bool(payload.get("expect_guardrail", False)),
        )


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_cases(path: Path) -> List[AnalysisCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("analysis cases file must be an array")
    cases = [AnalysisCase.from_dict(item, i) for i, item in enumerate(raw)]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in analysis cases")
    return cases


def _ratio(results: List[Dict[str, Any]], key: str) -> float:
    total = len(results)
    return (sum(1 for r in results if r.get(key)) / total) if total else 0.0


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    return {
        "cases_total": total,
        "analysis_plan_valid_rate": _ratio(results, "analysis_plan_valid"),
        "field_reference_valid_rate": _ratio(results, "field_reference_valid"),
        "chart_spec_valid_rate": _ratio(results, "chart_spec_valid"),
        "insight_grounded_rate": _ratio(results, "insight_grounded"),
        "claim_strength_appropriate_rate": _ratio(results, "claim_strength_appropriate"),
        "followup_continuity_ok_rate": _ratio(results, "followup_continuity_ok"),
        "render_success_rate": _ratio(results, "render_success"),
        "overall_pass_rate": _ratio(results, "overall_pass"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--handles", type=Path, default=DEFAULT_HANDLES)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args(argv)

    if not args.cases.exists():
        raise SystemExit(f"cases file not found: {args.cases}")
    if not args.handles.exists():
        raise SystemExit(f"handles file not found: {args.handles}")

    cases = load_cases(args.cases)
    handles = load_distilled_handles(args.handles)

    coordinator = AnalysisCoordinator()
    session_last_analysis_handle: Dict[str, str] = {}
    case_to_analysis_handle: Dict[str, str] = {}

    results: List[Dict[str, Any]] = []

    for case in cases:
        if case.handle_id not in handles:
            raise SystemExit(f"unknown handle_id in case {case.id}: {case.handle_id}")
        handle = handles[case.handle_id]

        csv_path = Path(handle.path)
        if not csv_path.is_absolute():
            csv_path = REPO_ROOT / csv_path
        if not csv_path.exists():
            raise SystemExit(f"distilled artifact not found: {handle.path}")

        df = pd.read_csv(csv_path)

        prior_handle: Optional[str] = None
        if case.followup_from:
            prior_handle = case_to_analysis_handle.get(case.followup_from)
        elif case.session_id in session_last_analysis_handle:
            prior_handle = session_last_analysis_handle[case.session_id]

        run = coordinator.run(
            question=case.question,
            distilled_df=df,
            schema=handle.schema,
            prior_analysis_handle=prior_handle,
            expected_followup_from=case.followup_from,
        )

        family_match = run.plan.family == case.family
        guardrail_ok = True
        if case.expect_guardrail:
            guardrail_ok = bool(run.report.blocked_claims)

        overall_pass = bool(
            family_match
            and run.analysis_plan_valid
            and run.field_reference_valid
            and run.chart_spec_valid
            and run.insight_grounded
            and run.claim_strength_appropriate
            and run.followup_continuity_ok
            and run.render_success
            and guardrail_ok
        )

        analysis_output_handle = run.hops[1].output_handle if len(run.hops) > 1 else None
        if analysis_output_handle:
            session_last_analysis_handle[case.session_id] = analysis_output_handle
            case_to_analysis_handle[case.id] = analysis_output_handle

        results.append(
            {
                "id": case.id,
                "family": case.family,
                "family_match": family_match,
                "handle_id": case.handle_id,
                "question": case.question,
                "session_id": case.session_id,
                "followup_from": case.followup_from,
                "analysis_plan": run.plan.to_dict(),
                "chart_specs": [c.to_dict() for c in run.charts],
                "dashboard_spec": run.dashboard.to_dict() if run.dashboard else None,
                "insight_report": run.report.to_dict(),
                "hops": [
                    {
                        "worker": h.worker,
                        "input_handle": h.input_handle,
                        "output_handle": h.output_handle,
                        "bytes_materialized": h.bytes_materialized,
                        "validation_result": h.validation_result,
                        "note": h.note,
                    }
                    for h in run.hops
                ],
                "analysis_plan_valid": run.analysis_plan_valid,
                "field_reference_valid": run.field_reference_valid,
                "chart_spec_valid": run.chart_spec_valid,
                "insight_grounded": run.insight_grounded,
                "claim_strength_appropriate": run.claim_strength_appropriate,
                "followup_continuity_ok": run.followup_continuity_ok,
                "render_success": run.render_success,
                "guardrail_enforced": guardrail_ok,
                "overall_pass": overall_pass,
            }
        )

    summary = _aggregate(results)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(args.cases),
        "handles_path": str(args.handles),
        "summary": summary,
        "results": results,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"open_data_analysis_{_utc_stamp()}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "report_path": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
