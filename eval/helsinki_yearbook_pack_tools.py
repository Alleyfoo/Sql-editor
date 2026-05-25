from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "open_data" / "helsinki_yearbook_2024_analysis_test_set.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval" / "golden" / "open_data"


@dataclass(frozen=True)
class SplitResult:
    easy: List[Dict[str, Any]]
    mid: List[Dict[str, Any]]
    hard: List[Dict[str, Any]]


def load_pack(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("pack must be a JSON object")
    meta = payload.get("meta")
    cases = payload.get("cases")
    if not isinstance(meta, dict) or not isinstance(cases, list):
        raise ValueError("pack must contain 'meta' object and 'cases' array")
    return payload


def classify_case(case: Dict[str, Any]) -> str:
    category = str(case.get("category") or "").strip()
    output_type = str(case.get("output_type") or "").strip()
    validator = str(case.get("validator") or "").strip()

    if category == "lookup" and output_type == "scalar" and validator.endswith("_exact"):
        return "easy"

    if category in {"ranking", "comparison"}:
        return "mid"
    if output_type in {"ranked_list", "label_plus_value", "year_plus_value"}:
        return "mid"

    # Hard bucket focuses on derived analytics: ratios, shares, deltas, projections,
    # and other computations that are more sensitive to interpretation/rounding.
    return "hard"


def split_cases(cases: List[Dict[str, Any]]) -> SplitResult:
    easy: List[Dict[str, Any]] = []
    mid: List[Dict[str, Any]] = []
    hard: List[Dict[str, Any]] = []

    for case in cases:
        bucket = classify_case(case)
        if bucket == "easy":
            easy.append(case)
        elif bucket == "mid":
            mid.append(case)
        else:
            hard.append(case)

    return SplitResult(easy=easy, mid=mid, hard=hard)


def _write_subset(path: Path, *, source_pack: Path, bucket: str, cases: List[Dict[str, Any]]) -> None:
    out = {
        "meta": {
            "source_pack": str(source_pack),
            "subset": bucket,
            "cases_total": len(cases),
            "split_rule": "deterministic category/output_type/validator mapping",
        },
        "cases": cases,
    }
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def _write_summary(path: Path, *, source_pack: Path, split: SplitResult) -> None:
    lines = [
        "# Helsinki Yearbook Analysis Pack Split",
        "",
        f"Source: `{source_pack}`",
        "",
        f"- easy: **{len(split.easy)}**",
        f"- mid: **{len(split.mid)}**",
        f"- hard: **{len(split.hard)}**",
        f"- total: **{len(split.easy) + len(split.mid) + len(split.hard)}**",
        "",
        "## Rules",
        "",
        "- easy: lookup + scalar + exact validator",
        "- mid: ranking/comparison/label-plus-value/ranked-list/year-plus-value",
        "- hard: ratio/share/change-over-time/projection/derived metrics and remaining cases",
        "",
        "## IDs By Subset",
        "",
        "### Easy",
    ]
    lines.extend([f"- `{c['id']}`" for c in split.easy] or ["- (none)"])
    lines.append("")
    lines.append("### Mid")
    lines.extend([f"- `{c['id']}`" for c in split.mid] or ["- (none)"])
    lines.append("")
    lines.append("### Hard")
    lines.extend([f"- `{c['id']}`" for c in split.hard] or ["- (none)"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_and_write(input_path: Path, output_dir: Path) -> Dict[str, int]:
    payload = load_pack(input_path)
    cases = payload["cases"]
    split = split_cases(cases)

    output_dir.mkdir(parents=True, exist_ok=True)
    easy_path = output_dir / "helsinki_yearbook_2024_analysis_easy.json"
    mid_path = output_dir / "helsinki_yearbook_2024_analysis_mid.json"
    hard_path = output_dir / "helsinki_yearbook_2024_analysis_hard.json"
    summary_path = output_dir / "helsinki_yearbook_2024_analysis_split_summary.md"

    _write_subset(easy_path, source_pack=input_path, bucket="easy", cases=split.easy)
    _write_subset(mid_path, source_pack=input_path, bucket="mid", cases=split.mid)
    _write_subset(hard_path, source_pack=input_path, bucket="hard", cases=split.hard)
    _write_summary(summary_path, source_pack=input_path, split=split)

    return {
        "easy": len(split.easy),
        "mid": len(split.mid),
        "hard": len(split.hard),
        "total": len(cases),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"input pack not found: {args.input}")

    counts = build_and_write(args.input, args.output_dir)
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
