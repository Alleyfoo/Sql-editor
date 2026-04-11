"""Presentation-oriented analysis benchmark on distilled handles.

This benchmark is intentionally separate from numeric-ground-truth packs.
It evaluates analysis usefulness and presentation quality signals:

- summary usefulness
- chart choice appropriateness
- dashboard layout coherence
- insight grounding
- claim-strength policy compliance
- follow-up continuity
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

DEFAULT_CASES = REPO_ROOT / "eval" / "golden" / "open_data" / "analysis_presentation_cases.json"
DEFAULT_HANDLES = REPO_ROOT / "eval" / "golden" / "open_data" / "analysis_distilled_handles.json"


@dataclass(frozen=True)
class PresentationCase:
    id: str
    handle_id: str
    question: str
    session_id: str
    followup_from: Optional[str]
    expected_family: str
    expected_focus_fields: List[str]
    expected_chart_types: List[str]
    require_chart: bool
    require_dashboard: bool

    @staticmethod
    def from_dict(payload: Dict[str, Any], index: int) -> "PresentationCase":
        if not isinstance(payload, dict):
            raise ValueError(f"cases[{index}] must be an object")

        def _req_str(key: str) -> str:
            val = payload.get(key)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"cases[{index}].{key} must be non-empty string")
            return val.strip()

        expected_family = _req_str("expected_family")
        allowed = {"kpi_summary", "trend", "segment_comparison", "dashboard_design", "follow_up_analysis", "guardrail"}
        if expected_family not in allowed:
            raise ValueError(f"cases[{index}].expected_family must be one of {sorted(allowed)}")

        focus_fields = payload.get("expected_focus_fields") or []
        if not isinstance(focus_fields, list) or not all(isinstance(x, str) and x.strip() for x in focus_fields):
            raise ValueError(f"cases[{index}].expected_focus_fields must be a string array")

        chart_types = payload.get("expected_chart_types") or []
        if not isinstance(chart_types, list) or not all(isinstance(x, str) and x.strip() for x in chart_types):
            raise ValueError(f"cases[{index}].expected_chart_types must be a string array")

        followup_from = payload.get("followup_from")
        if followup_from is not None and not isinstance(followup_from, str):
            raise ValueError(f"cases[{index}].followup_from must be string or null")

        return PresentationCase(
            id=_req_str("id"),
            handle_id=_req_str("handle_id"),
            question=_req_str("question"),
            session_id=_req_str("session_id"),
            followup_from=followup_from,
            expected_family=expected_family,
            expected_focus_fields=list(focus_fields),
            expected_chart_types=list(chart_types),
            require_chart=bool(payload.get("require_chart", False)),
            require_dashboard=bool(payload.get("require_dashboard", False)),
        )


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_cases(path: Path) -> List[PresentationCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("presentation cases file must be an array")
    cases = [PresentationCase.from_dict(item, i) for i, item in enumerate(raw)]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in presentation cases")
    return cases


def _summary_useful(*, summary: str, expected_focus_fields: List[str], evidence_fields: List[str]) -> bool:
    words = [w for w in summary.strip().split() if w]
    if len(words) < 6:
        return False
    if not expected_focus_fields:
        return True
    focus = {f.lower() for f in expected_focus_fields}
    sum_tokens = summary.lower()
    if any(f in sum_tokens for f in focus):
        return True
    ev = {e.lower() for e in evidence_fields}
    return bool(focus.intersection(ev))


def _chart_choice_appropriate(
    *,
    chart_types: List[str],
    expected_chart_types: List[str],
    require_chart: bool,
) -> bool:
    if require_chart and not chart_types:
        return False
    if not expected_chart_types:
        return True
    if not chart_types:
        return not require_chart
    expected = {x.lower() for x in expected_chart_types}
    actual = {x.lower() for x in chart_types}
    return bool(expected.intersection(actual))


def _dashboard_layout_coherent(dashboard: Optional[Dict[str, Any]], chart_count: int, require_dashboard: bool) -> bool:
    if not require_dashboard:
        return True
    if not dashboard:
        return False
    tiles = dashboard.get("tiles")
    if not isinstance(tiles, list) or len(tiles) < 2:
        return False
    kinds = [str(t.get("kind") or "") for t in tiles if isinstance(t, dict)]
    if "kpi_card" not in kinds or "chart" not in kinds:
        return False
    for tile in tiles:
        if not isinstance(tile, dict):
            return False
        if tile.get("kind") == "chart":
            ref = tile.get("chart_ref")
            if not isinstance(ref, int) or ref < 0 or ref >= chart_count:
                return False
    return True


def _ratio(results: List[Dict[str, Any]], key: str) -> float:
    total = len(results)
    return (sum(1 for r in results if r.get(key)) / total) if total else 0.0


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "cases_total": len(results),
        "family_match_rate": _ratio(results, "family_match"),
        "summary_useful_rate": _ratio(results, "summary_useful"),
        "chart_choice_appropriate_rate": _ratio(results, "chart_choice_appropriate"),
        "dashboard_layout_coherent_rate": _ratio(results, "dashboard_layout_coherent"),
        "insight_grounded_rate": _ratio(results, "insight_grounded"),
        "claim_strength_appropriate_rate": _ratio(results, "claim_strength_appropriate"),
        "followup_continuity_ok_rate": _ratio(results, "followup_continuity_ok"),
        "render_success_rate": _ratio(results, "render_success"),
        "guardrails_clean_rate": _ratio(results, "guardrails_clean"),
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

        family_match = run.plan.family == case.expected_family
        chart_types = [c.chart_type for c in run.charts]
        dashboard_payload = run.dashboard.to_dict() if run.dashboard else None
        evidence_fields = [field for insight in run.report.insights for field in insight.evidence_fields]

        summary_useful = _summary_useful(
            summary=run.report.summary,
            expected_focus_fields=case.expected_focus_fields,
            evidence_fields=evidence_fields,
        )
        chart_choice_appropriate = _chart_choice_appropriate(
            chart_types=chart_types,
            expected_chart_types=case.expected_chart_types,
            require_chart=case.require_chart,
        )
        dashboard_layout_coherent = _dashboard_layout_coherent(
            dashboard=dashboard_payload,
            chart_count=len(run.charts),
            require_dashboard=case.require_dashboard,
        )

        overall_pass = bool(
            family_match
            and summary_useful
            and chart_choice_appropriate
            and dashboard_layout_coherent
            and run.insight_grounded
            and run.claim_strength_appropriate
            and run.followup_continuity_ok
            and run.render_success
            and run.guardrails_clean
        )

        analysis_output_handle = run.hops[1].output_handle if len(run.hops) > 1 else None
        if analysis_output_handle:
            session_last_analysis_handle[case.session_id] = analysis_output_handle
            case_to_analysis_handle[case.id] = analysis_output_handle

        results.append(
            {
                "id": case.id,
                "session_id": case.session_id,
                "followup_from": case.followup_from,
                "handle_id": case.handle_id,
                "question": case.question,
                "expected_family": case.expected_family,
                "family_actual": run.plan.family,
                "family_match": family_match,
                "summary_useful": summary_useful,
                "chart_choice_appropriate": chart_choice_appropriate,
                "dashboard_layout_coherent": dashboard_layout_coherent,
                "insight_grounded": run.insight_grounded,
                "claim_strength_appropriate": run.claim_strength_appropriate,
                "followup_continuity_ok": run.followup_continuity_ok,
                "render_success": run.render_success,
                "guardrails_clean": run.guardrails_clean,
                "guardrail_errors": list(run.guardrail_errors),
                "overall_pass": overall_pass,
                "analysis_plan": run.plan.to_dict(),
                "chart_specs": [c.to_dict() for c in run.charts],
                "dashboard_spec": dashboard_payload,
                "insight_report": run.report.to_dict(),
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
    report_path = args.report_dir / f"open_data_analysis_presentation_{_utc_stamp()}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "report_path": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

