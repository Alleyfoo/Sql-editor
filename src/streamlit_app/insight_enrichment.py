"""Phase 4c — LLM enrichment for deterministic analysis results.

Called only when the user clicks 'Ask + Analyze'. Takes the already-computed
:class:`DeterministicAnalysis` (headlines + cards) and asks the local Qwen /
Ollama model to write 2–3 sentences of plain-English narrative and up to 3
additional follow-up questions.

Contract:
- Never replaces ``insights`` or ``headline`` (those are deterministic truth).
- Only sets ``prose`` and extends ``next_questions`` (capped at 5 total).
- Any failure (network, parse, timeout) is swallowed silently; the
  deterministic results are still shown intact.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from src.streamlit_app.insight_engine import DeterministicAnalysis


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def enrich_analysis(
    det: DeterministicAnalysis,
    *,
    sql: str,
    user_text: str,
    results_sample: str,
    config,  # LLMConfig from src.llm.natural_language
) -> DeterministicAnalysis:
    """Return a new DeterministicAnalysis with LLM-generated prose + follow-ups.

    ``results_sample`` should be a short CSV string (≤15 rows) of the
    query result. The caller is responsible for truncating.
    """
    try:
        prose, follow_ups = _call_llm(
            det,
            sql=sql,
            user_text=user_text,
            results_sample=results_sample,
            config=config,
        )
    except Exception:
        return det  # enrichment failure is always silent

    # Merge follow-ups: deterministic ones come first, LLM ones extend the list
    questions = list(det.next_questions)
    for q in follow_ups:
        if q and q not in questions:
            questions.append(q)
    questions = questions[:5]  # never show more than 5 chips

    return DeterministicAnalysis(
        headline=det.headline,
        insights=det.insights,
        next_questions=questions,
        warnings=det.warnings,
        prose=prose or None,
        pattern=det.pattern,
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a concise data analyst assistant. "
    "Respond only with valid JSON — no markdown fences, no extra text."
)

_USER_TEMPLATE = """\
A user asked: "{user_text}"

SQL executed:
{sql}

Result sample (up to 15 rows, CSV):
{results_sample}

Pre-computed findings:
{det_summary}

Task:
1. Write a "narrative" of 2–3 sentences interpreting what these results mean. \
Be specific — cite actual values (numbers, names) from the data above.
2. Suggest up to 3 "follow_ups" as short question strings the user might \
want to ask next. Each must be a natural-language question, ≤60 chars.

Return exactly this JSON shape:
{{"narrative": "...", "follow_ups": ["...", "...", "..."]}}
"""


def _det_summary(det: DeterministicAnalysis) -> str:
    lines: List[str] = []
    if det.headline:
        lines.append(f"Headline: {det.headline.text}")
    for ins in det.insights:
        lines.append(f"- {ins.label}: {ins.value}  ({ins.delta})")
    if not lines:
        lines.append("No pre-computed findings.")
    return "\n".join(lines)


def _call_llm(
    det: DeterministicAnalysis,
    *,
    sql: str,
    user_text: str,
    results_sample: str,
    config,
) -> Tuple[Optional[str], List[str]]:
    from src.llm.natural_language import LLMError, make_llm_client
    import dataclasses

    # Use a shorter timeout for enrichment — it's non-critical
    cfg_short = dataclasses.replace(config, timeout=min(config.timeout, 90.0))
    client = make_llm_client(cfg_short)

    user_msg = _USER_TEMPLATE.format(
        user_text=user_text,
        sql=sql.strip(),
        results_sample=results_sample.strip(),
        det_summary=_det_summary(det),
    )

    data = client.generate_json(_SYSTEM, user_msg)

    narrative = str(data.get("narrative") or "").strip()
    raw_fups = data.get("follow_ups") or []
    follow_ups = [str(q).strip() for q in raw_fups if q][:3]

    return narrative or None, follow_ups


# ---------------------------------------------------------------------------
# Helper: build results_sample CSV string
# ---------------------------------------------------------------------------


def results_to_sample_csv(df, max_rows: int = 15) -> str:
    """Return a compact CSV string of the first ``max_rows`` rows.

    Floats are rounded to 2 decimal places so the prompt stays short.
    """
    if df is None or len(df) == 0:
        return "(no rows)"
    sample = df.head(max_rows).copy()
    for col in sample.select_dtypes(include="float").columns:
        sample[col] = sample[col].round(2)
    return sample.to_csv(index=False)


__all__ = ["enrich_analysis", "results_to_sample_csv"]
