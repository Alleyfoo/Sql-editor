"""Capability spike evaluator for dirty Excel header extraction.

This evaluates whether a model can identify displaced headers in messy
Excel/CSV files and return normalized header names.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.llm.natural_language import OllamaClient, load_llm_config


SYSTEM_PROMPT = (
    "You identify the header row in messy spreadsheet-like data. "
    "Return ONLY one JSON object with keys: "
    '"header_row_index" (0-based integer), '
    '"headers" (array of normalized snake_case header names with empty cells removed), '
    '"confidence" (number 0..1), '
    '"notes" (short string). '
    "Do not include markdown."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - rank) + ordered[high] * (rank - low))


def _normalize_header(text: str) -> str:
    out = []
    prev_sep = False
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_sep = False
        else:
            if not prev_sep:
                out.append("_")
                prev_sep = True
    normalized = "".join(out).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


@dataclass
class CleaningCapabilityCase:
    id: str
    file: str
    expected_header_row: int
    expected_headers: List[str]
    tags: List[str] = field(default_factory=list)
    mock_response: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_dict(payload: Dict[str, Any], index: int) -> "CleaningCapabilityCase":
        if not isinstance(payload, dict):
            raise ValueError(f"cases[{index}] must be an object")
        cid = payload.get("id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"cases[{index}].id must be a non-empty string")

        file_name = payload.get("file")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError(f"cases[{index}].file must be a non-empty string")

        header_row = payload.get("expected_header_row")
        if isinstance(header_row, bool) or not isinstance(header_row, int) or header_row < 0:
            raise ValueError(f"cases[{index}].expected_header_row must be >= 0 integer")

        expected_headers_raw = payload.get("expected_headers")
        if not isinstance(expected_headers_raw, list) or not expected_headers_raw:
            raise ValueError(f"cases[{index}].expected_headers must be a non-empty array")
        expected_headers: List[str] = []
        for h in expected_headers_raw:
            if not isinstance(h, str) or not h.strip():
                raise ValueError(f"cases[{index}].expected_headers has an invalid value")
            expected_headers.append(_normalize_header(h))

        tags_raw = payload.get("tags", [])
        if tags_raw is None:
            tags_raw = []
        if not isinstance(tags_raw, list) or not all(isinstance(t, str) for t in tags_raw):
            raise ValueError(f"cases[{index}].tags must be an array of strings")

        mock_response = payload.get("mock_response")
        if mock_response is not None and not isinstance(mock_response, dict):
            raise ValueError(f"cases[{index}].mock_response must be an object")

        return CleaningCapabilityCase(
            id=cid.strip(),
            file=file_name.strip(),
            expected_header_row=header_row,
            expected_headers=expected_headers,
            tags=list(tags_raw),
            mock_response=mock_response,
        )


def load_cases(path: Path) -> List[CleaningCapabilityCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("cases file must be a JSON array")
    return [CleaningCapabilityCase.from_dict(item, i) for i, item in enumerate(raw)]


def _load_preview_rows(path: Path, max_rows: int, max_cols: int) -> List[List[str]]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, header=None, dtype=str, nrows=max_rows)
    elif suffix == ".csv":
        df = pd.read_csv(path, header=None, dtype=str, nrows=max_rows)
    else:
        raise ValueError(f"unsupported file type for cleaning eval: {path.suffix}")
    df = df.fillna("")
    rows: List[List[str]] = []
    for _, row in df.iterrows():
        cells = [str(v)[:80] for v in row.tolist()[:max_cols]]
        rows.append(cells)
    return rows


def _build_user_prompt(rows: List[List[str]]) -> str:
    lines = []
    for idx, row in enumerate(rows):
        rendered = " | ".join(cell if cell != "" else "<EMPTY>" for cell in row)
        lines.append(f"{idx}: {rendered}")
    preview = "\n".join(lines)
    return (
        "Find the header row index from this spreadsheet preview. "
        "Header row is where column names begin. Data rows follow it.\n\n"
        "Preview rows (0-based):\n"
        f"{preview}\n\n"
        "Return normalized snake_case headers, remove empty cells, keep order."
    )


@dataclass
class ProviderCallResult:
    payload: Dict[str, Any]


class MockEvalProvider:
    provider = "mock"

    def __init__(self, model: str) -> None:
        self.model = model

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        case: CleaningCapabilityCase,
    ) -> ProviderCallResult:
        if case.mock_response is not None:
            return ProviderCallResult(payload=case.mock_response)
        # Fallback keeps mock runs deterministic without verbose fixture JSON.
        return ProviderCallResult(
            payload={
                "header_row_index": case.expected_header_row,
                "headers": case.expected_headers,
                "confidence": 1.0,
                "notes": "mock fallback from expected case metadata",
            }
        )


class OllamaEvalProvider:
    provider = "ollama"

    def __init__(self, host: str, model: str, timeout: float) -> None:
        self.model = model
        self._client = OllamaClient(host=host, model=model, timeout=timeout)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        case: CleaningCapabilityCase,
    ) -> ProviderCallResult:
        payload = self._client.generate_json(system_prompt, user_prompt)
        return ProviderCallResult(payload=payload)


class OpenAIEvalProvider:
    provider = "openai"

    def __init__(self, model: str, timeout: float, base_url: str) -> None:
        self.model = model
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for --provider openai")

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        case: CleaningCapabilityCase,
    ) -> ProviderCallResult:
        req_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(req_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            raise RuntimeError(f"OpenAI returned HTTP {exc.code}: {exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenAI transport error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"OpenAI request timed out after {self.timeout:.0f}s"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"OpenAI OS transport error: {exc}") from exc

        envelope = json.loads(raw)
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI response had no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError("OpenAI choice payload shape was invalid")
        message = first.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("OpenAI message content was empty")
        payload = json.loads(message["content"])
        if not isinstance(payload, dict):
            raise RuntimeError("OpenAI JSON payload top-level must be object")
        return ProviderCallResult(payload=payload)


def _build_provider(args: argparse.Namespace):
    if args.provider == "mock":
        return MockEvalProvider(model=args.model or "mock")
    if args.provider == "ollama":
        return OllamaEvalProvider(host=args.host, model=args.model, timeout=args.timeout)
    if args.provider == "openai":
        return OpenAIEvalProvider(
            model=args.model,
            timeout=args.timeout,
            base_url=args.openai_base_url,
        )
    raise ValueError(f"unsupported provider: {args.provider}")


def _parse_prediction(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("prediction payload must be an object")

    raw_row = payload.get("header_row_index")
    if isinstance(raw_row, bool) or not isinstance(raw_row, int) or raw_row < 0:
        raise ValueError("header_row_index must be >= 0 integer")
    raw_headers = payload.get("headers")
    if not isinstance(raw_headers, list) or not raw_headers:
        raise ValueError("headers must be a non-empty array")
    headers: List[str] = []
    for idx, h in enumerate(raw_headers):
        if not isinstance(h, str):
            raise ValueError(f"headers[{idx}] must be a string")
        norm = _normalize_header(h)
        if norm:
            headers.append(norm)
    if not headers:
        raise ValueError("headers are empty after normalization")

    confidence = payload.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        conf = float(confidence)
    else:
        conf = None
    notes = payload.get("notes")
    notes_str = str(notes).strip() if notes is not None else ""
    return {
        "header_row_index": raw_row,
        "headers": headers,
        "confidence": conf,
        "notes": notes_str,
    }


def _header_positional_match_ratio(expected: List[str], predicted: List[str]) -> float:
    if not expected:
        return 0.0
    matched = 0
    for idx, exp in enumerate(expected):
        if idx < len(predicted) and predicted[idx] == exp:
            matched += 1
    return matched / len(expected)


def evaluate_cases(
    *,
    provider: Any,
    cases: List[CleaningCapabilityCase],
    base_dir: Path,
    max_rows: int,
    max_cols: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    latency: List[float] = []
    failures: Dict[str, int] = {}

    json_object_count = 0
    valid_prediction_count = 0
    exact_header_row_count = 0
    exact_headers_count = 0
    pass_count = 0

    for case in cases:
        data_path = (base_dir / case.file).resolve()
        preview_rows = _load_preview_rows(data_path, max_rows=max_rows, max_cols=max_cols)
        user_prompt = _build_user_prompt(preview_rows)

        started = time.perf_counter()
        prediction: Optional[Dict[str, Any]] = None
        failure_type: Optional[str] = None
        error_message: Optional[str] = None

        try:
            call = provider.generate_json(SYSTEM_PROMPT, user_prompt, case)
            json_object_count += 1
            prediction = _parse_prediction(call.payload)
            valid_prediction_count += 1
        except Exception as exc:
            error_message = str(exc)
            if "header_row_index" in error_message or "headers" in error_message:
                failure_type = "prediction_schema_error"
            elif "unsupported file type" in error_message:
                failure_type = "unsupported_file"
            else:
                failure_type = "provider_or_parse_error"
            failures[failure_type] = failures.get(failure_type, 0) + 1

        elapsed = (time.perf_counter() - started) * 1000.0
        latency.append(elapsed)

        expected_headers = case.expected_headers
        row_ok = False
        headers_exact = False
        header_match_ratio = 0.0
        if prediction is not None:
            predicted_headers = prediction["headers"]
            predicted_row = prediction["header_row_index"]
            row_ok = predicted_row == case.expected_header_row
            headers_exact = predicted_headers == expected_headers
            header_match_ratio = _header_positional_match_ratio(
                expected_headers, predicted_headers
            )

            if row_ok:
                exact_header_row_count += 1
            if headers_exact:
                exact_headers_count += 1
            if row_ok and headers_exact:
                pass_count += 1

        rows.append(
            {
                "id": case.id,
                "file": case.file,
                "tags": case.tags,
                "latency_ms": round(elapsed, 3),
                "valid_prediction": prediction is not None,
                "row_correct": row_ok,
                "headers_exact": headers_exact,
                "header_match_ratio": round(header_match_ratio, 4),
                "expected": {
                    "header_row_index": case.expected_header_row,
                    "headers": expected_headers,
                },
                "predicted": prediction,
                "failure_type": failure_type,
                "error": error_message,
            }
        )

    total = len(cases)
    return {
        "generated_at": _utc_now_iso(),
        "provider": provider.provider,
        "model": provider.model,
        "cases_total": total,
        "metrics": {
            "json_object_rate": (json_object_count / total) if total else 0.0,
            "valid_prediction_rate": (valid_prediction_count / total) if total else 0.0,
            "exact_header_row_rate": (exact_header_row_count / total) if total else 0.0,
            "exact_headers_rate": (exact_headers_count / total) if total else 0.0,
            "case_pass_rate": (pass_count / total) if total else 0.0,
            "latency_ms": {
                "mean": round(statistics.fmean(latency), 3) if latency else 0.0,
                "p50": round(_percentile(latency, 0.50), 3),
                "p95": round(_percentile(latency, 0.95), 3),
                "max": round(max(latency), 3) if latency else 0.0,
            },
        },
        "failures": failures,
        "results": rows,
    }


def _default_cases_path() -> Path:
    return Path("eval/golden/cleaning/dirty_excel_cases.json")


def _default_output_path(provider: str, model: str) -> Path:
    safe_model = model.replace("/", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(f"eval/reports/cleaning_capability_{provider}_{safe_model}_{stamp}.json")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dirty-Excel header extraction capability eval."
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "ollama", "openai"],
        default="mock",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--host", default="")
    parser.add_argument("--timeout", type=float, default=0.0)
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com"),
    )
    parser.add_argument("--cases", type=Path, default=_default_cases_path())
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=12)
    parser.add_argument("--max-cols", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config = load_config()
    llm_cfg = load_llm_config(config)
    if not args.model:
        if args.provider == "ollama":
            args.model = llm_cfg.model
        elif args.provider == "openai":
            args.model = "gpt-4o-mini"
        else:
            args.model = "mock-model"
    if not args.host:
        args.host = llm_cfg.host
    if not args.timeout or args.timeout <= 0:
        args.timeout = llm_cfg.timeout

    cases = load_cases(args.cases)
    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]

    provider = _build_provider(args)
    base_dir = args.cases.parent
    report = evaluate_cases(
        provider=provider,
        cases=cases,
        base_dir=base_dir,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
    )

    out_path = args.output or _default_output_path(provider.provider, provider.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = {
        "provider": provider.provider,
        "model": provider.model,
        "cases_total": report["cases_total"],
        "case_pass_rate": report["metrics"]["case_pass_rate"],
        "exact_header_row_rate": report["metrics"]["exact_header_row_rate"],
        "exact_headers_rate": report["metrics"]["exact_headers_rate"],
        "report_path": str(out_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
