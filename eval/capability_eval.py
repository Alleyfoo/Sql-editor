"""Capability spike evaluator for NL -> JSON plan model behavior.

This script measures whether a model can reliably produce valid query plans
for the existing ``sql-editor`` trust boundary, before we commit to larger
agent work.

It intentionally evaluates only capability-level behavior:
- JSON object response rate
- Query-plan validity rate (parse + SQL generation safety checks)
- Hallucination rate (unknown columns/operators)
- Latency
- Token usage (actual when provider returns it, otherwise estimated)

Usage examples:

    python eval/capability_eval.py --provider mock
    python eval/capability_eval.py --provider ollama --model gemma3
    python eval/capability_eval.py --provider openai --model gpt-4o-mini
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
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Ensure "python eval/capability_eval.py" can import src/* modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.llm.natural_language import (
    LLMError,
    OllamaClient,
    SYSTEM_PROMPT,
    build_user_prompt,
    load_llm_config,
    parse_query_plan,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    """Simple character-based token estimate when provider usage is missing."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


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
    left = ordered[low] * (high - rank)
    right = ordered[high] * (rank - low)
    return float(left + right)


@dataclass
class CapabilityCase:
    id: str
    question: str
    schema: Dict[str, str]
    tags: List[str] = field(default_factory=list)
    selected_columns: List[str] = field(default_factory=list)
    history: List[Tuple[str, str]] = field(default_factory=list)
    mock_response: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_dict(payload: Dict[str, Any], index: int) -> "CapabilityCase":
        if not isinstance(payload, dict):
            raise ValueError(f"cases[{index}] must be an object")

        cid = payload.get("id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"cases[{index}].id must be a non-empty string")

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"cases[{index}].question must be a non-empty string")

        schema = payload.get("schema")
        if not isinstance(schema, dict) or not schema:
            raise ValueError(f"cases[{index}].schema must be a non-empty object")
        clean_schema: Dict[str, str] = {}
        for col, ctype in schema.items():
            if not isinstance(col, str) or not col.strip():
                raise ValueError(f"cases[{index}].schema contains an invalid column")
            if ctype not in {"text", "numeric", "date"}:
                raise ValueError(
                    f"cases[{index}].schema[{col!r}] must be text|numeric|date"
                )
            clean_schema[col] = ctype

        tags_raw = payload.get("tags", [])
        if tags_raw is None:
            tags_raw = []
        if not isinstance(tags_raw, list) or not all(
            isinstance(t, str) for t in tags_raw
        ):
            raise ValueError(f"cases[{index}].tags must be an array of strings")

        selected_columns_raw = payload.get("selected_columns", [])
        if selected_columns_raw is None:
            selected_columns_raw = []
        if not isinstance(selected_columns_raw, list) or not all(
            isinstance(c, str) for c in selected_columns_raw
        ):
            raise ValueError(
                f"cases[{index}].selected_columns must be an array of strings"
            )

        history_raw = payload.get("history", [])
        if history_raw is None:
            history_raw = []
        if not isinstance(history_raw, list):
            raise ValueError(f"cases[{index}].history must be an array")
        history: List[Tuple[str, str]] = []
        for h_index, pair in enumerate(history_raw):
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or not isinstance(pair[1], str)
            ):
                raise ValueError(
                    f"cases[{index}].history[{h_index}] must be [question, reply]"
                )
            history.append((pair[0], pair[1]))

        mock_response = payload.get("mock_response")
        if mock_response is not None and not isinstance(mock_response, dict):
            raise ValueError(f"cases[{index}].mock_response must be an object")

        return CapabilityCase(
            id=cid.strip(),
            question=question.strip(),
            schema=clean_schema,
            tags=list(tags_raw),
            selected_columns=list(selected_columns_raw),
            history=history,
            mock_response=mock_response,
        )


def load_cases(path: Path) -> List[CapabilityCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("capability cases file must be a JSON array")
    return [CapabilityCase.from_dict(item, i) for i, item in enumerate(raw)]


@dataclass
class ProviderCallResult:
    payload: Dict[str, Any]
    usage_prompt_tokens: Optional[int] = None
    usage_completion_tokens: Optional[int] = None
    usage_total_tokens: Optional[int] = None


class EvalProvider(Protocol):
    provider: str
    model: str

    def generate_json(
        self, system_prompt: str, user_prompt: str, case: CapabilityCase
    ) -> ProviderCallResult:
        """Execute one model call and return parsed JSON payload + optional usage."""


class OllamaEvalProvider:
    provider = "ollama"

    def __init__(self, host: str, model: str, timeout: float) -> None:
        self.model = model
        self._client = OllamaClient(host=host, model=model, timeout=timeout)

    def generate_json(
        self, system_prompt: str, user_prompt: str, case: CapabilityCase
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
            raise RuntimeError(
                "OPENAI_API_KEY is required for --provider openai capability eval"
            )

    def generate_json(
        self, system_prompt: str, user_prompt: str, case: CapabilityCase
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
        body = json.dumps(req_payload).encode("utf-8")
        request = Request(
            self.base_url + "/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            text = ""
            try:
                text = exc.read().decode("utf-8", errors="replace")
            except Exception:
                text = ""
            msg = f"OpenAI returned HTTP {exc.code}: {exc.reason}"
            if text:
                msg += f" ({text})"
            raise RuntimeError(msg) from exc
        except URLError as exc:
            raise RuntimeError(f"OpenAI transport error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"OpenAI request timed out after {self.timeout:.0f}s"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"OpenAI OS transport error: {exc}") from exc

        envelope = json.loads(raw)
        if not isinstance(envelope, dict):
            raise RuntimeError("OpenAI response envelope was not a JSON object")

        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI response had no choices")

        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError("OpenAI choice payload shape was invalid")
        message = first.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("OpenAI choice has no message object")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenAI message content was empty")

        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise RuntimeError("OpenAI JSON payload top-level must be an object")

        usage = envelope.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        return ProviderCallResult(
            payload=payload,
            usage_prompt_tokens=int(prompt_tokens)
            if isinstance(prompt_tokens, int)
            else None,
            usage_completion_tokens=int(completion_tokens)
            if isinstance(completion_tokens, int)
            else None,
            usage_total_tokens=int(total_tokens) if isinstance(total_tokens, int) else None,
        )


class MockEvalProvider:
    provider = "mock"

    def __init__(self, model: str) -> None:
        self.model = model

    def generate_json(
        self, system_prompt: str, user_prompt: str, case: CapabilityCase
    ) -> ProviderCallResult:
        if case.mock_response is None:
            raise RuntimeError(
                f"case {case.id!r} has no mock_response; required for mock provider"
            )
        return ProviderCallResult(payload=case.mock_response)


def _classify_error(stage: str, message: str) -> str:
    low = message.lower()
    if stage == "provider":
        if "valid json" in low or "not json" in low:
            return "invalid_json"
        return "provider_error"

    if "not in the dataset schema" in low:
        return "hallucinated_column"
    if ".operator" in low and "not valid" in low:
        return "hallucinated_operator"
    if "must be an array" in low or "must be an object" in low:
        return "json_schema_mismatch"
    if "blocked sql keyword" in low or "only select statements are allowed" in low:
        return "security_violation"
    return "other_validation_error"


def _build_provider(args: argparse.Namespace, cases: List[CapabilityCase]) -> EvalProvider:
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


def evaluate_cases(
    *,
    provider: EvalProvider,
    cases: List[CapabilityCase],
    input_cost_per_1m: Optional[float] = None,
    output_cost_per_1m: Optional[float] = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    failures: Dict[str, int] = {}
    latency_ms: List[float] = []

    prompt_tokens_sum = 0
    completion_tokens_sum = 0
    total_tokens_sum = 0
    actual_usage_rows = 0

    valid_plan_count = 0
    json_object_count = 0
    security_probe_count = 0
    security_probe_safe_count = 0

    for case in cases:
        user_prompt = build_user_prompt(
            case.question,
            case.schema,
            selected_columns=case.selected_columns or None,
            history=case.history or None,
        )
        started = time.perf_counter()

        payload: Optional[Dict[str, Any]] = None
        model_sql = ""
        error_type = ""
        error_message = ""
        parse_valid = False
        sql_safe = False

        usage_prompt_tokens: Optional[int] = None
        usage_completion_tokens: Optional[int] = None
        usage_total_tokens: Optional[int] = None

        try:
            call = provider.generate_json(SYSTEM_PROMPT, user_prompt, case)
            payload = call.payload
            json_object_count += 1
            usage_prompt_tokens = call.usage_prompt_tokens
            usage_completion_tokens = call.usage_completion_tokens
            usage_total_tokens = call.usage_total_tokens
        except Exception as exc:
            error_message = str(exc)
            error_type = _classify_error("provider", error_message)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency_ms.append(elapsed_ms)

        if payload is not None:
            try:
                model = parse_query_plan(payload, case.schema)
                model_sql = model.to_sql()
                parse_valid = True
                sql_safe = True
                valid_plan_count += 1
            except (LLMError, ValueError) as exc:
                error_message = str(exc)
                error_type = _classify_error("parser", error_message)

        if "security_probe" in case.tags:
            security_probe_count += 1
            if sql_safe:
                security_probe_safe_count += 1

        if usage_prompt_tokens is not None and usage_completion_tokens is not None:
            actual_usage_rows += 1
            prompt_tokens = usage_prompt_tokens
            completion_tokens = usage_completion_tokens
            total_tokens = (
                usage_total_tokens
                if usage_total_tokens is not None
                else usage_prompt_tokens + usage_completion_tokens
            )
        else:
            completion_text = json.dumps(payload, sort_keys=True) if payload else ""
            prompt_tokens = _estimate_tokens(SYSTEM_PROMPT + "\n" + user_prompt)
            completion_tokens = _estimate_tokens(completion_text)
            total_tokens = prompt_tokens + completion_tokens

        prompt_tokens_sum += prompt_tokens
        completion_tokens_sum += completion_tokens
        total_tokens_sum += total_tokens

        if error_type:
            failures[error_type] = failures.get(error_type, 0) + 1

        rows.append(
            {
                "id": case.id,
                "question": case.question,
                "tags": case.tags,
                "latency_ms": round(elapsed_ms, 3),
                "json_object": payload is not None,
                "valid_plan": parse_valid,
                "sql_safe": sql_safe,
                "failure_type": error_type or None,
                "error": error_message or None,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "source": "actual"
                    if usage_prompt_tokens is not None and usage_completion_tokens is not None
                    else "estimated",
                },
                "sql_preview": model_sql,
            }
        )

    total = len(cases)
    hallucinations = failures.get("hallucinated_column", 0) + failures.get(
        "hallucinated_operator", 0
    )

    usage_source = "actual" if actual_usage_rows == total else "estimated"
    if 0 < actual_usage_rows < total:
        usage_source = "mixed"

    cost_usd: Optional[float] = None
    if input_cost_per_1m is not None and output_cost_per_1m is not None:
        cost_usd = (
            (prompt_tokens_sum / 1_000_000.0) * input_cost_per_1m
            + (completion_tokens_sum / 1_000_000.0) * output_cost_per_1m
        )

    return {
        "generated_at": _utc_now_iso(),
        "provider": provider.provider,
        "model": provider.model,
        "cases_total": total,
        "metrics": {
            "json_object_rate": (json_object_count / total) if total else 0.0,
            "valid_plan_rate": (valid_plan_count / total) if total else 0.0,
            "hallucination_rate": (hallucinations / total) if total else 0.0,
            "security_probe_safe_rate": (
                (security_probe_safe_count / security_probe_count)
                if security_probe_count
                else 0.0
            ),
            "latency_ms": {
                "mean": round(statistics.fmean(latency_ms), 3) if latency_ms else 0.0,
                "p50": round(_percentile(latency_ms, 0.50), 3),
                "p95": round(_percentile(latency_ms, 0.95), 3),
                "max": round(max(latency_ms), 3) if latency_ms else 0.0,
            },
            "token_usage": {
                "prompt_tokens": prompt_tokens_sum,
                "completion_tokens": completion_tokens_sum,
                "total_tokens": total_tokens_sum,
                "source": usage_source,
            },
            "estimated_cost_usd": round(cost_usd, 6) if cost_usd is not None else None,
        },
        "failures": failures,
        "results": rows,
    }


def _default_cases_path() -> Path:
    return Path("eval/golden/capability/starter_cases.json")


def _default_report_path(provider: str, model: str) -> Path:
    safe_model = model.replace("/", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(f"eval/reports/capability_{provider}_{safe_model}_{stamp}.json")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NL->JSON plan capability eval against configured provider."
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "ollama", "openai"],
        default="mock",
        help="Model provider for this run.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Provider model name. If omitted for ollama, config/env defaults are used.",
    )
    parser.add_argument(
        "--host",
        default="",
        help="Ollama host override (for --provider ollama).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Request timeout in seconds (0 means use config/default).",
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com"),
        help="Base URL for OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=_default_cases_path(),
        help="Path to capability cases JSON.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Limit number of cases (0 means all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path. Defaults to eval/reports/capability_<provider>_<model>_<ts>.json",
    )
    parser.add_argument(
        "--input-cost-per-1m",
        type=float,
        default=None,
        help="Optional USD price per 1M input tokens for cost estimate.",
    )
    parser.add_argument(
        "--output-cost-per-1m",
        type=float,
        default=None,
        help="Optional USD price per 1M output tokens for cost estimate.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config = load_config()
    llm_cfg = load_llm_config(config)

    if not args.model:
        args.model = llm_cfg.model if args.provider == "ollama" else "mock-model"
    if not args.host:
        args.host = llm_cfg.host
    if not args.timeout or args.timeout <= 0:
        args.timeout = llm_cfg.timeout

    cases = load_cases(args.cases)
    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]

    provider = _build_provider(args, cases)
    report = evaluate_cases(
        provider=provider,
        cases=cases,
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
    )

    output_path = args.output or _default_report_path(provider.provider, provider.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = {
        "provider": provider.provider,
        "model": provider.model,
        "cases_total": report["cases_total"],
        "valid_plan_rate": report["metrics"]["valid_plan_rate"],
        "hallucination_rate": report["metrics"]["hallucination_rate"],
        "json_object_rate": report["metrics"]["json_object_rate"],
        "latency_ms_p50": report["metrics"]["latency_ms"]["p50"],
        "report_path": str(output_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
