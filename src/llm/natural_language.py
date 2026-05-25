"""Natural-language → QueryModel via a local Ollama model.

This module is the Phase 3 trust boundary. It:

1. Talks to an Ollama HTTP endpoint (default ``http://localhost:11434``)
   using only the Python standard library — no extra dependency.
2. Asks the model to return a JSON *query plan*, **not** SQL.
3. Parses that JSON into a fully validated :class:`QueryModel`.

The parser is the untrusted-input boundary: every column name is
checked against the active schema, every operator and aggregation
function is allow-listed against the same constants the visual
composer uses, and the resulting model still has to survive
``QueryModel.to_sql()`` → ``_assert_select_only()`` → the executor
blocklist before any SQL touches SQLite. Three layers of defense.

The Ollama HTTP pattern (stdlib ``urllib``, POST to ``/api/chat`` with
``stream: false``) is adapted from
``Alleyfoo/Support-triage-llm/app/slm_ollama.py``. The env-var layering
(``OLLAMA_HOST`` / ``OLLAMA_MODEL`` / ``OLLAMA_TIMEOUT``) mirrors
``Alleyfoo/Support-triage-llm/app/config.py``. Neither file was vendored
in — see ``VENDOR.md`` for the reference-only attribution.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..query_model import (
    AGGREGATION_FUNCTIONS,
    Aggregation,
    ColumnFormat,
    DATE_BUCKET_GRAINS,
    Filter,
    MAX_FORMAT_DECIMALS,
    OPERATORS_BY_TYPE,
    ORDER_DIRECTIONS,
    QueryModel,
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Raised when the LLM call fails or returns an unusable plan.

    The UI catches this and shows it in a ``messagebox.showerror``; the
    message is intentionally human-readable so the user can tell whether
    the problem is transport (Ollama down), schema (model hallucinated a
    column), or validation (bad operator, etc.).
    """


class RouteToPythonError(LLMError):
    """Raised when NL intent should be handled by Python analytics."""

    def __init__(
        self,
        reason: str,
        *,
        blocked_intent: str,
        next_actions: List[str],
        note: str = "",
    ) -> None:
        self.reason = reason
        self.blocked_intent = blocked_intent
        self.next_actions = list(next_actions)
        self.note = note
        lines = [
            "This request was routed away from SQL generation.",
            f"Why: {reason}.",
            f"Blocked in SQL mode: {blocked_intent}.",
        ]
        if note:
            lines.append(f"Note: {note}.")
        if self.next_actions:
            lines.append("Next best actions:")
            lines.extend([f"- {action}" for action in self.next_actions])
        super().__init__("\n".join(lines))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    host: str = "http://localhost:11434"
    model: str = "gemma3"
    timeout: float = 60.0
    provider: str = "ollama"  # "ollama" | "groq" | "openai_compatible"
    history_depth: int = 6
    api_key: str = ""  # for Groq / OpenAI-compatible providers


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_llm_config(app_config: Optional[Dict[str, Any]] = None) -> LLMConfig:
    """Build an ``LLMConfig`` from the already-loaded ``config.yaml``
    dict, applying env-var overrides.

    Precedence (first non-empty wins):

    1. ``OLLAMA_HOST`` / ``OLLAMA_MODEL`` / ``OLLAMA_TIMEOUT`` env vars
    2. ``st.session_state["_llm_overrides"]`` — runtime UI changes
    3. ``st.secrets["llm"]`` — Streamlit Community Cloud secrets
    4. ``llm:`` section of ``config.yaml`` — local development
    5. Hard-coded defaults
    """
    section: Dict[str, Any] = {}
    if app_config:
        raw = app_config.get("llm") or {}
        if isinstance(raw, dict):
            section = raw

    # Layer in st.secrets (works on Streamlit Community Cloud; no-op locally)
    try:
        import streamlit as st
        st_secrets = dict(st.secrets.get("llm", {}))  # type: ignore[attr-defined]
        if st_secrets:
            section = {**section, **st_secrets}
    except Exception:
        pass

    # Layer in runtime UI overrides (always available, survives ephemeral filesystem)
    try:
        import streamlit as st
        overrides: Dict[str, Any] = st.session_state.get("_llm_overrides", {})
        if overrides:
            section = {**section, **overrides}
    except Exception:
        pass

    host = (
        os.environ.get("OLLAMA_HOST") or section.get("host") or "http://localhost:11434"
    )
    model = os.environ.get("OLLAMA_MODEL") or section.get("model") or "gemma3"
    timeout_default = float(section.get("timeout", 60.0) or 60.0)
    timeout = _env_float("OLLAMA_TIMEOUT", timeout_default)
    provider = str(section.get("provider") or "ollama")
    # API key: env var takes precedence over config file / secrets
    api_key = (
        os.environ.get("GROQ_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or str(section.get("api_key") or "")
    )
    try:
        history_depth = int(section.get("history_depth", 6))
    except (TypeError, ValueError):
        history_depth = 6
    history_depth = max(0, history_depth)

    return LLMConfig(
        host=host,
        model=model,
        timeout=timeout,
        provider=provider,
        history_depth=history_depth,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------


class OllamaClient:
    """Minimal stdlib-only Ollama chat client in JSON mode.

    Not a full SDK — just enough to POST one prompt and parse the
    response. Injectable into :func:`nl_to_query_model` so tests can
    substitute a fake without touching the network.
    """

    def __init__(self, host: str, model: str, timeout: float = 60.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = float(timeout)

    def generate_json(self, system: str, user: str) -> Dict[str, Any]:
        """POST a chat request and return the model's JSON payload.

        Raises :class:`LLMError` on any transport error, non-JSON
        response, or unexpected envelope shape.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
            "think": False,   # disable extended thinking (deepseek-r1, qwq, etc.)
            "options": {"temperature": 0.0},
        }
        data = json.dumps(payload).encode("utf-8")
        url = self.host + "/api/chat"
        request = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(
                request, timeout=self.timeout
            ) as response:  # nosec - local inference endpoint
                body = response.read()
        except HTTPError as exc:
            raise LLMError(f"Ollama returned HTTP {exc.code}: {exc.reason}") from exc
        except URLError as exc:
            raise LLMError(
                f"could not reach Ollama at {self.host}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise LLMError(
                f"Ollama request timed out after {self.timeout:.0f}s"
            ) from exc
        except OSError as exc:
            raise LLMError(f"Ollama transport error: {exc}") from exc

        try:
            envelope = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Ollama response was not JSON: {exc}") from exc

        if not isinstance(envelope, dict):
            raise LLMError("Ollama response envelope is not an object")

        message = envelope.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Ollama response contained no message content")

        # Some models wrap JSON in markdown fences (```json … ```) even when
        # asked for raw JSON.  Strip fences before parsing.
        content_stripped = content.strip()
        if content_stripped.startswith("```"):
            # Drop opening fence line and closing fence.
            lines = content_stripped.splitlines()
            # Remove first line (``` or ```json) and last ``` line.
            inner_lines = lines[1:]
            if inner_lines and inner_lines[-1].strip() == "```":
                inner_lines = inner_lines[:-1]
            content_stripped = "\n".join(inner_lines).strip()

        try:
            parsed = json.loads(content_stripped)
        except json.JSONDecodeError as exc:
            raise LLMError(f"model did not return valid JSON: {exc.msg}") from exc

        if not isinstance(parsed, dict):
            raise LLMError("model JSON must be an object at the top level")

        return parsed

    def generate_text(self, system: str, user: str) -> str:
        """POST a chat request and return the model's plain-text response.

        Like ``generate_json`` but without ``format: json`` — used for
        multi-table SQL generation where the model should return raw SQL.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0},
        }
        data = json.dumps(payload).encode("utf-8")
        url = self.host + "/api/chat"
        request = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # nosec
                body = response.read()
        except HTTPError as exc:
            raise LLMError(f"Ollama returned HTTP {exc.code}: {exc.reason}") from exc
        except URLError as exc:
            raise LLMError(
                f"could not reach Ollama at {self.host}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise LLMError(
                f"Ollama request timed out after {self.timeout:.0f}s"
            ) from exc
        except OSError as exc:
            raise LLMError(f"Ollama transport error: {exc}") from exc

        try:
            envelope = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Ollama response was not JSON: {exc}") from exc

        message = envelope.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Ollama response contained no message content")
        return content.strip()


# ---------------------------------------------------------------------------
# OpenAI-compatible client (Groq, OpenAI, any /v1/chat/completions endpoint)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def make_llm_client(cfg: LLMConfig):
    """Return an OllamaClient for the configured host/model."""
    return OllamaClient(host=cfg.host, model=cfg.model, timeout=cfg.timeout)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You translate a user's natural-language request about a tabular "
    "dataset into a JSON query plan. Output ONLY a single JSON object "
    "matching the schema given. Do not invent columns. Do not emit SQL. "
    'Always include a "reply" field: one short plain-English sentence '
    "confirming what you understood and what the query will do. "
    "If the user asks for numeric formatting (e.g. 'round to 2 decimals', "
    "'show prices with 2 decimal places'), include a \"column_formats\" "
    'object mapping each affected column name to {"round": N} where N is '
    "the number of decimal places. Only apply formatting to numeric columns. "
    "If the user asks for per-day/group-by-day output, include "
    '"date_buckets": {"<date_column>": "day"} and keep selected/group columns '
    "as real schema columns (no SQL expressions in selected_columns)."
)


_PYTHON_ROUTE_PATTERNS: List[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bpercentile(s)?\b", re.IGNORECASE), "percentile"),
    (re.compile(r"\bquantile(s)?\b", re.IGNORECASE), "quantile"),
    (
        re.compile(r"\b(rolling|moving)\s+(average|avg|window)\b", re.IGNORECASE),
        "rolling window",
    ),
    (
        re.compile(r"\b(stdev|stddev|std dev|standard deviation)\b", re.IGNORECASE),
        "standard deviation",
    ),
    (re.compile(r"\boutlier(s)?\b", re.IGNORECASE), "outlier detection"),
    (re.compile(r"\banomal(y|ies)\b", re.IGNORECASE), "anomaly detection"),
    (re.compile(r"\b(hour of day|departure hour|per hour)\b", re.IGNORECASE), "hour extraction"),
    (
        re.compile(
            r"\b(same station|start and end at the same station|departure station id equals return station id)\b",
            re.IGNORECASE,
        ),
        "column-to-column comparison",
    ),
    (
        re.compile(
            r"\b(more departures than returns|receive more trips than they send)\b",
            re.IGNORECASE,
        ),
        "cross-role station balance",
    ),
    (
        re.compile(
            r"\b(percentage|percent|ratio|share)\s+of\b|\bwhat\s+(percentage|percent|ratio|share)\b",
            re.IGNORECASE,
        ),
        "percentage/rate",
    ),
]


def detect_python_route_reason(nl: str) -> Optional[str]:
    """Return the matched reason when a prompt should route to Python."""
    text = (nl or "").strip()
    if not text:
        return None
    for pattern, reason in _PYTHON_ROUTE_PATTERNS:
        if pattern.search(text):
            return reason
    return None


def _has_queryable_time_column(schema: Dict[str, str]) -> bool:
    for col, col_type in schema.items():
        name = str(col).lower()
        if col_type == "date":
            return True
        if any(tok in name for tok in ["date", "time", "year", "month", "day", "week"]):
            return True
    return False


def _build_python_redirect_details(
    *,
    reason: str,
    schema: Dict[str, str],
    nl: str,
) -> tuple[str, List[str], str]:
    blocked_intent_map = {
        "percentile": "percentile/quantile computations",
        "quantile": "percentile/quantile computations",
        "rolling window": "rolling-window analytics",
        "standard deviation": "dispersion/statistical computations",
        "outlier detection": "outlier detection workflows",
        "anomaly detection": "anomaly detection workflows",
        "hour extraction": "timestamp-part extraction and bucketing",
        "column-to-column comparison": "column-to-column row comparisons",
        "cross-role station balance": "cross-role station balance analysis",
        "percentage/rate": "ratio/share/percentage calculations",
    }
    blocked_intent = blocked_intent_map.get(reason, "analytics-heavy computation")

    next_actions = [
        "Use the Python analytics route for an exact computed result.",
        "If you only need a descriptive readout, run a simpler SQL query and then use Ask + Analyze.",
    ]

    note = ""
    has_time = _has_queryable_time_column(schema)
    if reason in {"rolling window", "hour extraction"} and not has_time:
        note = "no queryable time column was detected in the current schema"
        next_actions.insert(0, "Add or normalize a time/date column first (preprocessing), then retry.")

    low_nl = (nl or "").lower()
    if any(tok in low_nl for tok in ["trend", "over time", "time series"]) and not has_time:
        note = "no queryable time column was detected in the current schema"
        if "Add or normalize a time/date column first (preprocessing), then retry." not in next_actions:
            next_actions.insert(0, "Add or normalize a time/date column first (preprocessing), then retry.")

    return blocked_intent, next_actions, note


def _format_schema(schema: Dict[str, str]) -> str:
    return "\n".join(f"  - {col} ({ctype})" for col, ctype in schema.items())


def build_user_prompt(
    nl: str,
    schema: Dict[str, str],
    selected_columns: Optional[List[str]] = None,
    history: Optional[List[tuple]] = None,
    table_name: str = "data",
) -> str:
    """Compose the user turn: schema, allowed values, target JSON shape.

    If ``selected_columns`` is provided (columns already ticked in the
    visual composer), they are added as a hint so the model can refine
    or extend the current selection rather than starting from scratch.

    If ``history`` is provided it must be a list of ``(question, reply)``
    tuples (oldest first, already trimmed to the desired depth by the
    caller).  They are embedded as a compact Q/A block so the model can
    resolve follow-up references like "also filter by that column".
    """
    ops = sorted({op for ops in OPERATORS_BY_TYPE.values() for op in ops})
    context_hint = ""
    if selected_columns:
        context_hint = (
            f"The user currently has these columns selected: "
            f"{', '.join(selected_columns)}\n\n"
        )
    history_block = ""
    if history:
        lines = ["Conversation so far (oldest first):"]
        for q, a in history:
            lines.append(f"  Q: {q}")
            lines.append(f"  A: {a}")
        history_block = "\n".join(lines) + "\n\n"
    return (
        f"You are given a single table named `{table_name}` with these columns:\n"
        f"{_format_schema(schema)}\n"
        "\n"
        f"{history_block}"
        f"{context_hint}"
        "Produce a JSON object with this exact shape (fields are optional; "
        "omit or use [] / null when not needed):\n"
        "{\n"
        '  "reply": "One sentence confirming what the query will do.",\n'
        '  "selected_columns": ["col", ...],\n'
        '  "filters": [{"column": "col", "operator": "=", "value": "x", '
        '"logical": "AND"}, ...],\n'
        '  "group_by": ["col", ...],\n'
        '  "aggregations": [{"function": "SUM", "column": "col", '
        '"alias": "total"}, ...],\n'
        '  "having": [{"column": "total", "operator": ">", "value": 100}, ...],\n'
        '  "order_by": [["col_or_alias", "ASC"], ...],\n'
        '  "limit": 10,\n'
        '  "column_formats": {"price": {"round": 2}},\n'
        '  "date_buckets": {"created_at": "day"}\n'
        "}\n"
        "\n"
        f"Allowed operators: {ops}\n"
        f"Allowed aggregation functions: {list(AGGREGATION_FUNCTIONS)}\n"
        f"Allowed order directions: {list(ORDER_DIRECTIONS)}\n"
        'Allowed logical joiners: ["AND", "OR"]\n'
        "For BETWEEN, value must be a 2-element array [low, high].\n"
        "For IS NULL / IS NOT NULL, omit value.\n"
        f"Allowed date_buckets grains: {list(DATE_BUCKET_GRAINS)}.\n"
        f"column_formats keys must be numeric columns from the schema above; "
        f'"round" must be an integer between 0 and {MAX_FORMAT_DECIMALS}.\n'
        "When aggregations are present, every non-aggregated column in "
        "selected_columns MUST appear in group_by. "
        "NEVER put aggregation aliases (e.g. 'total_sales') into "
        "selected_columns — aggregation expressions are added automatically.\n"
        "\n"
        f"User request: {nl}\n"
        "JSON:"
    )


# ---------------------------------------------------------------------------
# Parser — the untrusted-input boundary
# ---------------------------------------------------------------------------


_VALID_LOGICAL = {"AND", "OR"}
_NULLARY_OPS = {"IS NULL", "IS NOT NULL"}
_RANGE_OPS = {"BETWEEN"}
_INLINE_AGG_RE = re.compile(
    r"^(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(\*|[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*(?:AS\s+([A-Za-z_][A-Za-z0-9_]*))?$",
    re.IGNORECASE,
)
_INLINE_COUNT_DISTINCT_RE = re.compile(
    r"^COUNT\s+DISTINCT\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*(?:AS\s+([A-Za-z_][A-Za-z0-9_]*))?$",
    re.IGNORECASE,
)


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise LLMError(f"{field!r} must be a non-empty string (got {value!r})")
    return value


def _require_column(value: Any, schema: Dict[str, str], field: str) -> str:
    col = _require_str(value, field).strip()
    if col in schema:
        return col
    lower = col.lower()
    for key in schema:
        if key.lower() == lower:
            return key
    raise LLMError(f"{field}: column {col!r} is not in the dataset schema")


def _parse_filter(
    entry: Any,
    schema: Dict[str, str],
    *,
    scope: str,
) -> Filter:
    if not isinstance(entry, dict):
        raise LLMError(f"{scope} entries must be objects (got {type(entry).__name__})")

    col = _require_column(entry.get("column"), schema, f"{scope}.column")
    col_type = schema[col]
    allowed_ops = OPERATORS_BY_TYPE.get(col_type, OPERATORS_BY_TYPE["text"])

    op_raw = entry.get("operator")
    if not isinstance(op_raw, str) or not op_raw:
        raise LLMError(f"{scope}.operator must be a non-empty string")
    op = op_raw.strip().upper()
    if op not in allowed_ops:
        raise LLMError(
            f"{scope}.operator {op!r} is not valid for column {col!r} "
            f"(type {col_type}); allowed: {list(allowed_ops)}"
        )

    logical_raw = entry.get("logical", "AND")
    logical = (
        logical_raw.strip().upper()
        if isinstance(logical_raw, str) and logical_raw
        else "AND"
    )
    if logical not in _VALID_LOGICAL:
        raise LLMError(f"{scope}.logical must be 'AND' or 'OR' (got {logical_raw!r})")

    value: Any
    if op in _NULLARY_OPS:
        value = None
    elif op in _RANGE_OPS:
        raw = entry.get("value")
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise LLMError(
                f"{scope}.value for BETWEEN must be a 2-element array " f"(got {raw!r})"
            )
        lo, hi = raw
        _require_scalar(lo, f"{scope}.value[0]")
        _require_scalar(hi, f"{scope}.value[1]")
        value = (lo, hi)
    else:
        raw = entry.get("value")
        _require_scalar(raw, f"{scope}.value")
        value = raw

    return Filter(column=col, operator=op, value=value, logical=logical)


def _require_scalar(value: Any, field: str) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        raise LLMError(f"{field} must be a string or number, not a bool")
    if isinstance(value, (str, int, float)):
        return
    raise LLMError(
        f"{field} must be a string, number, or null (got {type(value).__name__})"
    )


def _parse_aggregation(
    entry: Any, schema: Dict[str, str], *, index: int
) -> Aggregation:
    if not isinstance(entry, dict):
        raise LLMError(
            f"aggregations[{index}] must be an object (got {type(entry).__name__})"
        )
    func_raw = entry.get("function")
    if not isinstance(func_raw, str) or not func_raw:
        raise LLMError(f"aggregations[{index}].function must be a non-empty string")
    func = func_raw.strip().upper()
    if func not in AGGREGATION_FUNCTIONS:
        raise LLMError(
            f"aggregations[{index}].function {func!r} is not allowed; "
            f"must be one of {list(AGGREGATION_FUNCTIONS)}"
        )

    col_raw = entry.get("column")
    if not isinstance(col_raw, str) or not col_raw:
        raise LLMError(f"aggregations[{index}].column must be a non-empty string")
    col = col_raw.strip()
    if col == "*":
        if func != "COUNT":
            raise LLMError(
                f"aggregations[{index}]: '*' is only valid with COUNT, not {func}"
            )
    else:
        if col not in schema:
            raise LLMError(
                f"aggregations[{index}].column {col!r} is not in the dataset schema"
            )

    alias_raw = entry.get("alias")
    if alias_raw is not None and (not isinstance(alias_raw, str) or not alias_raw):
        raise LLMError(
            f"aggregations[{index}].alias must be a non-empty string or omitted"
        )
    alias = alias_raw if isinstance(alias_raw, str) and alias_raw else None

    return Aggregation(column=col, function=func, alias=alias)


def _parse_order_by(
    entries: Any,
    schema: Dict[str, str],
    agg_names: set,
) -> List[tuple]:
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise LLMError(f"order_by must be an array (got {type(entries).__name__})")
    out: List[tuple] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise LLMError(
                f"order_by[{i}] must be a [column, direction] 2-element array "
                f"(got {entry!r})"
            )
        col_raw, dir_raw = entry
        if not isinstance(col_raw, str) or not col_raw:
            raise LLMError(f"order_by[{i}][0] must be a non-empty string")
        col = col_raw.strip()
        if col not in schema and col not in agg_names:
            raise LLMError(
                f"order_by[{i}] column {col!r} is not in the schema or "
                f"aggregation aliases"
            )
        if not isinstance(dir_raw, str) or not dir_raw:
            raise LLMError(f"order_by[{i}][1] must be a non-empty string")
        direction = dir_raw.strip().upper()
        if direction not in ORDER_DIRECTIONS:
            raise LLMError(
                f"order_by[{i}] direction {dir_raw!r} must be one of "
                f"{list(ORDER_DIRECTIONS)}"
            )
        out.append((col, direction))
    return out


def _parse_limit(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise LLMError("limit must be an integer, not a bool")
    if not isinstance(raw, (int, float)):
        raise LLMError(f"limit must be a non-negative integer (got {raw!r})")
    n = int(raw)
    if n < 0:
        raise LLMError(f"limit must be non-negative (got {n})")
    return None if n == 0 else n


def _parse_column_formats(raw: Any, schema: Dict[str, str]) -> Dict[str, ColumnFormat]:
    """Parse and validate the optional ``column_formats`` dict from the LLM.

    Accepted shape::

        {"price": {"round": 2}, "quantity": {"round": 0}}

    Raises :class:`LLMError` if:
    - the value is not a dict (when present)
    - any key is not a numeric column in the schema
    - ``round`` is not an integer in ``[0, MAX_FORMAT_DECIMALS]``
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise LLMError(f"column_formats must be an object (got {type(raw).__name__})")
    result: Dict[str, ColumnFormat] = {}
    for col, spec in raw.items():
        if col not in schema:
            raise LLMError(
                f"column_formats: column {col!r} is not in the dataset schema"
            )
        if schema[col] != "numeric":
            raise LLMError(
                f"column_formats: column {col!r} is type {schema[col]!r}; "
                "formatting is only supported for numeric columns"
            )
        if not isinstance(spec, dict):
            raise LLMError(
                f"column_formats[{col!r}] must be an object (got {type(spec).__name__})"
            )
        round_raw = spec.get("round")
        if round_raw is None:
            # No recognised keys — skip silently (forward-compatible).
            continue
        if isinstance(round_raw, bool) or not isinstance(round_raw, (int, float)):
            raise LLMError(
                f"column_formats[{col!r}].round must be an integer "
                f"(got {round_raw!r})"
            )
        decimals = int(round_raw)
        if decimals < 0 or decimals > MAX_FORMAT_DECIMALS:
            raise LLMError(
                f"column_formats[{col!r}].round must be between 0 and "
                f"{MAX_FORMAT_DECIMALS} (got {decimals})"
            )
        result[col] = ColumnFormat(column=col, round=decimals)
    return result


def _parse_inline_aggregation(
    raw: str,
    schema: Dict[str, str],
) -> Optional[Aggregation]:
    text = raw.strip()
    if not text:
        return None
    m = _INLINE_AGG_RE.match(text)
    if m:
        func = m.group(1).upper()
        col = m.group(2)
        alias = m.group(3)
        if col != "*" and col not in schema:
            return None
        if col == "*" and func != "COUNT":
            return None
        return Aggregation(column=col, function=func, alias=alias)

    m2 = _INLINE_COUNT_DISTINCT_RE.match(text)
    if m2:
        col = m2.group(1)
        alias = m2.group(2)
        if col not in schema:
            return None
        return Aggregation(column=col, function="COUNT DISTINCT", alias=alias)
    return None


def _parse_date_buckets(raw: Any, schema: Dict[str, str]) -> Dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise LLMError(f"date_buckets must be an object (got {type(raw).__name__})")
    out: Dict[str, str] = {}
    for col, grain_raw in raw.items():
        if col not in schema:
            raise LLMError(f"date_buckets: column {col!r} is not in the dataset schema")
        if schema[col] != "date":
            raise LLMError(
                f"date_buckets: column {col!r} is type {schema[col]!r}; only date columns are allowed"
            )
        if not isinstance(grain_raw, str) or not grain_raw.strip():
            raise LLMError(f"date_buckets[{col!r}] must be a non-empty string")
        grain = grain_raw.strip().lower()
        if grain not in DATE_BUCKET_GRAINS:
            raise LLMError(
                f"date_buckets[{col!r}] grain {grain!r} must be one of {list(DATE_BUCKET_GRAINS)}"
            )
        out[col] = grain
    return out


def parse_query_plan(payload: Any, schema: Dict[str, str], table_name: str = "data") -> QueryModel:
    """Convert a raw JSON dict from the LLM into a validated ``QueryModel``.

    Every field is checked against the active dataset schema and the
    same operator / function / direction allow-lists the visual composer
    uses. Raises :class:`LLMError` on any violation.
    """
    if not isinstance(payload, dict):
        raise LLMError(
            f"query plan must be a JSON object (got {type(payload).__name__})"
        )

    # reply — optional plain-text confirmation from the model
    reply_raw = payload.get("reply", "")
    reply: str = str(reply_raw).strip() if reply_raw else ""

    # aggregations — parsed early so their display names are available for
    # selected_columns and order_by validation below.
    agg_raw = payload.get("aggregations", [])
    if agg_raw is None:
        agg_raw = []
    if not isinstance(agg_raw, list):
        raise LLMError("aggregations must be an array")
    aggregations: List[Aggregation] = [
        _parse_aggregation(entry, schema, index=i) for i, entry in enumerate(agg_raw)
    ]
    agg_names = {a.display_name for a in aggregations}
    count_all_alias: Optional[str] = None
    for agg in aggregations:
        if agg.function.upper() == "COUNT" and agg.column == "*":
            count_all_alias = agg.display_name
            break

    # selected_columns — schema columns only; aggregation aliases are emitted
    # automatically and must NOT appear here.  We silently drop any entry that
    # matches an aggregation display name so models that include them don't
    # cause a hard failure.
    sel_raw = payload.get("selected_columns", [])
    if sel_raw is None:
        sel_raw = []
    if not isinstance(sel_raw, list):
        raise LLMError("selected_columns must be an array")
    selected_columns: List[str] = []
    agg_signature = {(a.function, a.column, a.alias or "") for a in aggregations}
    for i, col in enumerate(sel_raw):
        if not isinstance(col, str) or not col:
            raise LLMError(
                f"selected_columns[{i}] must be a non-empty string (got {col!r})"
            )
        if col in agg_names:
            # Model hallucinated the alias into selected_columns — skip it.
            continue
        inline_agg = _parse_inline_aggregation(col, schema)
        if inline_agg is not None:
            sig = (inline_agg.function, inline_agg.column, inline_agg.alias or "")
            if sig not in agg_signature:
                aggregations.append(inline_agg)
                agg_signature.add(sig)
                agg_names.add(inline_agg.display_name)
            continue
        selected_columns.append(_require_column(col, schema, f"selected_columns[{i}]"))

    # filters
    filter_raw = payload.get("filters", [])
    if filter_raw is None:
        filter_raw = []
    if not isinstance(filter_raw, list):
        raise LLMError("filters must be an array")
    filters: List[Filter] = [
        _parse_filter(entry, schema, scope=f"filters[{i}]")
        for i, entry in enumerate(filter_raw)
    ]

    # group_by
    gb_raw = payload.get("group_by", [])
    if gb_raw is None:
        gb_raw = []
    if not isinstance(gb_raw, list):
        raise LLMError("group_by must be an array")
    group_by: List[str] = [
        _require_column(col, schema, f"group_by[{i}]") for i, col in enumerate(gb_raw)
    ]

    # having — validated against group cols + aggregation display names
    having_raw = payload.get("having", [])
    if having_raw is None:
        having_raw = []
    if not isinstance(having_raw, list):
        raise LLMError("having must be an array")
    if not count_all_alias:
        has_count_star_having = any(
            isinstance(entry, dict)
            and isinstance(entry.get("column"), str)
            and re.match(r"^\s*count\s*\(\s*\*\s*\)\s*$", entry.get("column", ""), flags=re.IGNORECASE)
            for entry in having_raw
        )
        if has_count_star_having:
            auto_count = Aggregation(column="*", function="COUNT", alias="count_all")
            aggregations.append(auto_count)
            agg_names.add(auto_count.display_name)
            count_all_alias = auto_count.display_name
    if count_all_alias:
        normalized_having: List[Any] = []
        for entry in having_raw:
            if isinstance(entry, dict):
                col_raw = entry.get("column")
                if isinstance(col_raw, str) and re.match(
                    r"^\s*count\s*\(\s*\*\s*\)\s*$", col_raw, flags=re.IGNORECASE
                ):
                    patched = dict(entry)
                    patched["column"] = count_all_alias
                    normalized_having.append(patched)
                    continue
            normalized_having.append(entry)
        having_raw = normalized_having
    if having_raw and not group_by:
        raise LLMError("having requires at least one group_by column")
    having_schema: Dict[str, str] = {c: schema.get(c, "text") for c in group_by}
    having_expr_to_name: Dict[str, str] = {}
    for name in agg_names:
        having_schema[name] = "numeric"
    for agg in aggregations:
        func = agg.function.upper()
        if func == "COUNT DISTINCT":
            expr = f"COUNT(DISTINCT {agg.column})"
        elif func == "COUNT" and agg.column == "*":
            expr = "COUNT(*)"
        else:
            expr = f"{func}({agg.column})"
        variants = {
            expr,
            expr.lower(),
            expr.upper(),
            expr.replace(" ", ""),
            expr.lower().replace(" ", ""),
            expr.upper().replace(" ", ""),
        }
        for variant in variants:
            having_schema[variant] = "numeric"
            having_expr_to_name[variant] = agg.display_name
    having: List[Filter] = [
        _parse_filter(entry, having_schema, scope=f"having[{i}]")
        for i, entry in enumerate(having_raw)
    ]
    for h in having:
        if h.column in having_expr_to_name:
            h.column = having_expr_to_name[h.column]

    # order_by — accepts schema columns OR aggregation display names
    order_by = _parse_order_by(payload.get("order_by"), schema, agg_names)

    # limit
    limit = _parse_limit(payload.get("limit"))

    # column_formats — optional per-column numeric formatting
    column_formats = _parse_column_formats(payload.get("column_formats"), schema)
    date_buckets = _parse_date_buckets(payload.get("date_buckets"), schema)

    return QueryModel(
        table=table_name,
        selected_columns=selected_columns,
        filters=filters,
        group_by=group_by,
        aggregations=aggregations,
        having=having,
        order_by=order_by,
        limit=limit,
        reply=reply,
        column_formats=column_formats,
        date_buckets=date_buckets,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def nl_to_query_model(
    nl: str,
    schema: Dict[str, str],
    *,
    client: Optional[OllamaClient] = None,
    config: Optional[LLMConfig] = None,
    selected_columns: Optional[List[str]] = None,
    history: Optional[List[tuple]] = None,
    table_name: str = "data",
) -> QueryModel:
    """Translate a natural-language request into a validated ``QueryModel``.

    ``client`` is injectable so tests can pass a fake without a network
    call. If neither ``client`` nor ``config`` is provided, a default
    ``LLMConfig`` is loaded from env vars only.

    ``selected_columns`` — columns currently ticked in the visual
    composer — are passed as context so the model can refine or extend
    the active selection rather than starting from scratch.

    ``history`` — list of ``(question, reply)`` tuples from prior turns
    (oldest first, already trimmed to the configured depth).  Included
    in the prompt so the model can resolve follow-up references.
    """
    if not isinstance(nl, str) or not nl.strip():
        raise LLMError("natural-language request is empty")
    if not isinstance(schema, dict) or not schema:
        raise LLMError("schema is empty — open a CSV first")
    route_reason = detect_python_route_reason(nl.strip())
    if route_reason:
        blocked_intent, next_actions, note = _build_python_redirect_details(
            reason=route_reason,
            schema=schema,
            nl=nl.strip(),
        )
        raise RouteToPythonError(
            route_reason,
            blocked_intent=blocked_intent,
            next_actions=next_actions,
            note=note,
        )

    if client is None:
        cfg = config or load_llm_config({})
        client = make_llm_client(cfg)

    user_prompt = build_user_prompt(
        nl.strip(), schema, selected_columns=selected_columns, history=history,
        table_name=table_name,
    )
    payload = client.generate_json(SYSTEM_PROMPT, user_prompt)
    try:
        model = parse_query_plan(payload, schema, table_name=table_name)
    except LLMError as first_exc:
        msg = str(first_exc).lower()
        schema_error = "not in the dataset schema" in msg
        if not schema_error:
            raise
        # One constrained retry: remind the model to use schema columns only.
        fix_prompt = (
            user_prompt
            + "\n\nYour previous JSON was invalid because it referenced columns not in the schema."
            + " Return corrected JSON using ONLY exact schema column names."
        )
        payload_retry = client.generate_json(SYSTEM_PROMPT, fix_prompt)
        try:
            model = parse_query_plan(payload_retry, schema, table_name=table_name)
        except LLMError as retry_exc:
            raise LLMError(
                f"{first_exc}; retry failed: {retry_exc}. "
                f"Available columns: {', '.join(schema.keys())}"
            ) from retry_exc

    # Post-parse repair for common analytics phrasing:
    # "per day"/"by day" should bucket date columns by day, not full timestamp.
    low_nl = nl.lower()
    if (
        ("per day" in low_nl or "by day" in low_nl or "each day" in low_nl)
        and model.aggregations
    ):
        bucket_col: Optional[str] = None
        for col in model.group_by:
            if schema.get(col) == "date":
                bucket_col = col
                break
        if bucket_col is None:
            for col in model.selected_columns:
                if schema.get(col) == "date":
                    bucket_col = col
                    break
        if bucket_col is None and "time" in schema and schema.get("time") == "date":
            bucket_col = "time"
        if bucket_col is not None:
            if bucket_col not in model.group_by:
                model.group_by.insert(0, bucket_col)
            if bucket_col not in model.selected_columns:
                model.selected_columns.insert(0, bucket_col)
            model.date_buckets[bucket_col] = "day"
            if not model.order_by:
                model.order_by = [(bucket_col, "ASC")]

    if model.aggregations:
        # If the model leaves raw measure columns in selected_columns while
        # grouping, drop them to keep SQL structurally valid.
        keep = set(model.group_by)
        model.selected_columns = [c for c in model.selected_columns if c in keep]

    # Replace common date placeholders with concrete ISO dates.
    today = datetime.now(timezone.utc).date()
    seven_days_ago = (today - timedelta(days=7)).isoformat()
    seven_days_window_start = (today - timedelta(days=6)).isoformat()
    today_iso = today.isoformat()
    tomorrow_iso = (today + timedelta(days=1)).isoformat()
    placeholder_map = {
        "date_seven_days_ago": seven_days_ago,
        "date_7_days_ago": seven_days_ago,
        "current_date": today_iso,
    }
    has_current_upper_bound = any(
        isinstance(f.column, str)
        and schema.get(f.column) == "date"
        and isinstance(f.value, str)
        and str(f.value).lower() == "current_date"
        for f in model.filters
    )
    for f in model.filters:
        if not isinstance(f.column, str) or schema.get(f.column) != "date":
            continue
        if isinstance(f.value, str):
            key = f.value.strip().lower()
            if key in placeholder_map:
                f.value = placeholder_map[key]
                if key == "current_date":
                    has_current_upper_bound = True

    has_last_7_days_intent = "last 7 days" in low_nl or "past 7 days" in low_nl
    if has_last_7_days_intent and any(schema.get(f.column) == "date" for f in model.filters):
        # Normalize to an inclusive seven-day window [today-6, today].
        date_col = next(
            (f.column for f in model.filters if schema.get(f.column) == "date"),
            None,
        )
        if date_col is not None:
            lower_normalized = False
            upper_normalized = False
            for f in model.filters:
                if f.column != date_col:
                    continue
                if f.operator in {">", ">="}:
                    f.operator = ">="
                    f.value = seven_days_window_start
                    lower_normalized = True
                elif f.operator in {"<", "<="}:
                    f.operator = "<"
                    f.value = tomorrow_iso
                    upper_normalized = True
            if not lower_normalized:
                model.filters.append(
                    Filter(
                        column=date_col,
                        operator=">=",
                        value=seven_days_window_start,
                        logical="AND",
                    )
                )
            if not upper_normalized:
                model.filters.append(
                    Filter(column=date_col, operator="<", value=tomorrow_iso, logical="AND")
                )
            has_current_upper_bound = True

    # Scalar count intent repair: for questions asking for one number, avoid
    # unnecessary GROUP BY payload from model output.
    scalar_count_intent = (
        "one number" in low_nl
        or (
            "how many" in low_nl
            and " per " not in low_nl
            and " by " not in low_nl
            and "top" not in low_nl
        )
    )
    if scalar_count_intent and any(a.function.startswith("COUNT") for a in model.aggregations):
        model.selected_columns = []
        model.group_by = []
        model.order_by = []

    return model


# ---------------------------------------------------------------------------
# Multi-table SQL generation
# ---------------------------------------------------------------------------

_MULTITABLE_SYSTEM = (
    "You are a SQLite expert. Given a database schema and a user question, "
    "write a single read-only SELECT query. "
    "Return ONLY the SQL statement — no explanation, no markdown, no code fences. "
    "Use table aliases. Only use SELECT."
)

_SQL_DANGER = frozenset(
    w.upper() for w in
    ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE",
     "TRUNCATE", "ATTACH", "DETACH", "PRAGMA", "VACUUM")
)


def _detect_join_hints(tables_schema: Dict[str, Dict[str, str]]) -> List[str]:
    """Return human-readable JOIN hints auto-detected from shared _id columns.

    Only columns whose names end in ``_id`` (or equal ``id``) and that
    appear in more than one table are considered foreign keys.  The first
    table in insertion order that owns the column is treated as the primary
    (referenced) side.
    """
    id_col_tables: Dict[str, List[str]] = {}
    for table, schema in tables_schema.items():
        for col in schema:
            if col == "id" or col.endswith("_id"):
                id_col_tables.setdefault(col, []).append(table)

    hints: List[str] = []
    for col, tables in id_col_tables.items():
        if len(tables) > 1:
            primary = tables[0]
            for other in tables[1:]:
                hints.append(f"{other}.{col} = {primary}.{col}")
    return hints


def _build_multitable_prompt(
    nl: str,
    tables_schema: Dict[str, Dict[str, str]],
    history: Optional[List[tuple]] = None,
) -> str:
    """Compose the user turn for multi-table SQL generation."""
    lines: List[str] = ["Available tables:"]
    for table, schema in tables_schema.items():
        lines.append(f"\nTABLE {table}:")
        for col, ctype in schema.items():
            lines.append(f"  {col} ({ctype})")

    join_hints = _detect_join_hints(tables_schema)
    if join_hints:
        lines.append("\nDetected relationships (use for JOINs):")
        for hint in join_hints:
            lines.append(f"  {hint}")

    if history:
        lines.append("\nRecent conversation (oldest first):")
        for q, sql in history:
            lines.append(f"  Q: {q}")
            lines.append(f"  SQL: {sql[:120]}{'...' if len(sql) > 120 else ''}")

    lines.append(f"\nQuestion: {nl}")
    lines.append("\nWrite a single SQLite SELECT query:")
    return "\n".join(lines)


def _validate_raw_sql(sql: str) -> str:
    """Strip markdown fences, assert SELECT-only.  Returns cleaned SQL."""
    s = sql.strip()
    # Strip ```sql … ``` or ``` … ``` fences
    if s.startswith("```"):
        inner = s.splitlines()
        inner = inner[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        s = "\n".join(inner).strip()
    first_word = s.split()[0].upper() if s.split() else ""
    if first_word != "SELECT":
        raise LLMError(f"model returned non-SELECT statement (starts with '{first_word}')")
    tokens = {t.upper().rstrip(";,(") for t in s.split()}
    blocked = _SQL_DANGER & tokens
    if blocked:
        raise LLMError(f"model SQL contains disallowed keyword(s): {blocked}")
    return s


def nl_to_raw_sql(
    nl: str,
    tables_schema: Dict[str, Dict[str, str]],
    *,
    client: Optional[OllamaClient] = None,
    config: Optional["LLMConfig"] = None,
    history: Optional[List[tuple]] = None,
) -> str:
    """Translate a natural-language question into a raw SQLite SELECT query
    for a multi-table dataset.

    Returns the validated SQL string. Raises :class:`LLMError` on failure.
    ``history`` should be a list of ``(question, sql)`` tuples (recent first
    is fine; the prompt shows them oldest-first by reversing).
    """
    if not isinstance(nl, str) or not nl.strip():
        raise LLMError("natural-language request is empty")
    if not tables_schema:
        raise LLMError("no table schemas provided")

    if client is None:
        cfg = config or load_llm_config({})
        client = make_llm_client(cfg)

    ordered_history = list(reversed(history)) if history else []
    user_prompt = _build_multitable_prompt(nl.strip(), tables_schema, history=ordered_history)
    raw = client.generate_text(_MULTITABLE_SYSTEM, user_prompt)
    return _validate_raw_sql(raw)


__all__ = [
    "LLMError",
    "RouteToPythonError",
    "LLMConfig",
    "OllamaClient",
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "detect_python_route_reason",
    "load_llm_config",
    "nl_to_query_model",
    "nl_to_raw_sql",
    "parse_query_plan",
]
