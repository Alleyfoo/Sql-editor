"""Result-set distillation for the NL agent path.

Given an executed SQL query and its result DataFrame, this module asks the
LLM for a compact analysis block that can be shown in the UI chat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .natural_language import LLMConfig, LLMError, OllamaClient, load_llm_config


class AnalysisError(RuntimeError):
    """Raised when result analysis cannot be generated or parsed."""


@dataclass
class ResultAnalysis:
    summary: str
    insights: List[str] = field(default_factory=list)
    next_questions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


SYSTEM_ANALYSIS_PROMPT = (
    "You are a data analyst. Given SQL, result preview, and basic dataset stats, "
    "return ONLY JSON with keys: summary (string), insights (string array), "
    "next_questions (string array), warnings (string array). "
    "Do not claim values not supported by the provided rows/stats."
)


def _to_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _coerce_str_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AnalysisError("analysis list field must be an array")
    out: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _parse_analysis_payload(payload: Any) -> ResultAnalysis:
    if not isinstance(payload, dict):
        raise AnalysisError("analysis payload must be a JSON object")
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise AnalysisError("analysis summary must be a non-empty string")
    return ResultAnalysis(
        summary=summary.strip(),
        insights=_coerce_str_list(payload.get("insights")),
        next_questions=_coerce_str_list(payload.get("next_questions")),
        warnings=_coerce_str_list(payload.get("warnings")),
    )


def _result_preview(df: pd.DataFrame, max_rows: int = 20) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    sample = df.head(max_rows).copy()
    rows: List[Dict[str, Any]] = []
    for _, row in sample.iterrows():
        rows.append({str(col): _to_json_safe(row[col]) for col in sample.columns})
    return rows


def _numeric_stats(df: pd.DataFrame, max_cols: int = 8) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    numeric_cols = [str(c) for c in df.select_dtypes(include=["number"]).columns]
    for col in numeric_cols[:max_cols]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
        }
    return stats


def _build_user_prompt(
    question: str,
    sql: str,
    df: pd.DataFrame,
    schema: Dict[str, str],
    max_preview_rows: int,
) -> str:
    payload = {
        "question": question,
        "sql": sql,
        "row_count": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "schema": schema,
        "numeric_stats": _numeric_stats(df),
        "preview_rows": _result_preview(df, max_rows=max_preview_rows),
    }
    return "Analyze this executed query result:\n" + json.dumps(payload, ensure_ascii=True)


def fallback_result_analysis(
    question: str,
    sql: str,
    df: pd.DataFrame,
    *,
    warning: str = "",
) -> ResultAnalysis:
    _ = (question, sql)
    summary = f"Query returned {len(df)} rows across {len(df.columns)} columns."
    insights: List[str] = []
    numeric_cols = [str(c) for c in df.select_dtypes(include=["number"]).columns]
    if numeric_cols:
        col = numeric_cols[0]
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if not series.empty:
            insights.append(
                f"{col}: min={series.min():.3f}, mean={series.mean():.3f}, max={series.max():.3f}"
            )
    next_questions: List[str] = []
    if df.columns.size:
        next_questions.append(f"Would you like a breakdown by {df.columns[0]}?")
    warnings = [warning] if warning else []
    return ResultAnalysis(
        summary=summary,
        insights=insights,
        next_questions=next_questions,
        warnings=warnings,
    )


def analyze_result_with_llm(
    question: str,
    sql: str,
    df: pd.DataFrame,
    schema: Dict[str, str],
    *,
    client: Optional[OllamaClient] = None,
    config: Optional[LLMConfig] = None,
    max_preview_rows: int = 20,
) -> ResultAnalysis:
    """Return a concise analysis of the executed SQL result."""
    if df is None:
        raise AnalysisError("result dataframe is missing")
    if client is None:
        cfg = config or load_llm_config({})
        client = OllamaClient(host=cfg.host, model=cfg.model, timeout=cfg.timeout)
    prompt = _build_user_prompt(question, sql, df, schema, max_preview_rows=max_preview_rows)
    try:
        payload = client.generate_json(SYSTEM_ANALYSIS_PROMPT, prompt)
    except LLMError as exc:
        raise AnalysisError(str(exc)) from exc
    return _parse_analysis_payload(payload)


__all__ = [
    "AnalysisError",
    "ResultAnalysis",
    "SYSTEM_ANALYSIS_PROMPT",
    "analyze_result_with_llm",
    "fallback_result_analysis",
]
