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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..query_model import (
    AGGREGATION_FUNCTIONS,
    Aggregation,
    ColumnFormat,
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    host: str = "http://localhost:11434"
    model: str = "gemma3"
    timeout: float = 60.0
    provider: str = "ollama"
    history_depth: int = 6


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

    1. ``OLLAMA_HOST`` / ``OLLAMA_MODEL`` / ``OLLAMA_TIMEOUT``
    2. ``llm:`` section of ``config.yaml``
    3. Hard-coded defaults
    """
    section: Dict[str, Any] = {}
    if app_config:
        raw = app_config.get("llm") or {}
        if isinstance(raw, dict):
            section = raw

    host = (
        os.environ.get("OLLAMA_HOST") or section.get("host") or "http://localhost:11434"
    )
    model = os.environ.get("OLLAMA_MODEL") or section.get("model") or "gemma3"
    timeout_default = float(section.get("timeout", 60.0) or 60.0)
    timeout = _env_float("OLLAMA_TIMEOUT", timeout_default)
    provider = str(section.get("provider") or "ollama")
    try:
        history_depth = int(section.get("history_depth", 6))
    except (TypeError, ValueError):
        history_depth = 6
    history_depth = max(0, history_depth)

    return LLMConfig(host=host, model=model, timeout=timeout, provider=provider, history_depth=history_depth)


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

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"model did not return valid JSON: {exc.msg}") from exc

        if not isinstance(parsed, dict):
            raise LLMError("model JSON must be an object at the top level")

        return parsed


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
    "object mapping each affected column name to {\"round\": N} where N is "
    "the number of decimal places. Only apply formatting to numeric columns."
)


def _format_schema(schema: Dict[str, str]) -> str:
    return "\n".join(f"  - {col} ({ctype})" for col, ctype in schema.items())


def build_user_prompt(
    nl: str,
    schema: Dict[str, str],
    selected_columns: Optional[List[str]] = None,
    history: Optional[List[tuple]] = None,
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
        "You are given a single table named `data` with these columns:\n"
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
        '  "column_formats": {"price": {"round": 2}}\n'
        "}\n"
        "\n"
        f"Allowed operators: {ops}\n"
        f"Allowed aggregation functions: {list(AGGREGATION_FUNCTIONS)}\n"
        f"Allowed order directions: {list(ORDER_DIRECTIONS)}\n"
        'Allowed logical joiners: ["AND", "OR"]\n'
        "For BETWEEN, value must be a 2-element array [low, high].\n"
        "For IS NULL / IS NOT NULL, omit value.\n"
        f"column_formats keys must be numeric columns from the schema above; "
        f"\"round\" must be an integer between 0 and {MAX_FORMAT_DECIMALS}.\n"
        "When aggregations are present, every non-aggregated column in "
        "selected_columns MUST appear in group_by.\n"
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


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise LLMError(f"{field!r} must be a non-empty string (got {value!r})")
    return value


def _require_column(value: Any, schema: Dict[str, str], field: str) -> str:
    col = _require_str(value, field)
    if col not in schema:
        raise LLMError(f"{field}: column {col!r} is not in the dataset schema")
    return col


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


def _parse_column_formats(
    raw: Any, schema: Dict[str, str]
) -> Dict[str, ColumnFormat]:
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
        raise LLMError(
            f"column_formats must be an object (got {type(raw).__name__})"
        )
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


def parse_query_plan(payload: Any, schema: Dict[str, str]) -> QueryModel:
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

    # selected_columns
    sel_raw = payload.get("selected_columns", [])
    if sel_raw is None:
        sel_raw = []
    if not isinstance(sel_raw, list):
        raise LLMError("selected_columns must be an array")
    selected_columns: List[str] = []
    for i, col in enumerate(sel_raw):
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

    # aggregations
    agg_raw = payload.get("aggregations", [])
    if agg_raw is None:
        agg_raw = []
    if not isinstance(agg_raw, list):
        raise LLMError("aggregations must be an array")
    aggregations: List[Aggregation] = [
        _parse_aggregation(entry, schema, index=i) for i, entry in enumerate(agg_raw)
    ]
    agg_names = {a.display_name for a in aggregations}

    # having — validated against group cols + aggregation display names
    having_raw = payload.get("having", [])
    if having_raw is None:
        having_raw = []
    if not isinstance(having_raw, list):
        raise LLMError("having must be an array")
    if having_raw and not group_by:
        raise LLMError("having requires at least one group_by column")
    having_schema: Dict[str, str] = {c: schema.get(c, "text") for c in group_by}
    for name in agg_names:
        having_schema[name] = "numeric"
    having: List[Filter] = [
        _parse_filter(entry, having_schema, scope=f"having[{i}]")
        for i, entry in enumerate(having_raw)
    ]

    # order_by — accepts schema columns OR aggregation display names
    order_by = _parse_order_by(payload.get("order_by"), schema, agg_names)

    # limit
    limit = _parse_limit(payload.get("limit"))

    # column_formats — optional per-column numeric formatting
    column_formats = _parse_column_formats(payload.get("column_formats"), schema)

    return QueryModel(
        table="data",
        selected_columns=selected_columns,
        filters=filters,
        group_by=group_by,
        aggregations=aggregations,
        having=having,
        order_by=order_by,
        limit=limit,
        reply=reply,
        column_formats=column_formats,
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

    if client is None:
        cfg = config or load_llm_config({})
        client = OllamaClient(host=cfg.host, model=cfg.model, timeout=cfg.timeout)

    user_prompt = build_user_prompt(
        nl.strip(), schema, selected_columns=selected_columns, history=history
    )
    payload = client.generate_json(SYSTEM_PROMPT, user_prompt)
    return parse_query_plan(payload, schema)


__all__ = [
    "LLMError",
    "LLMConfig",
    "OllamaClient",
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "load_llm_config",
    "nl_to_query_model",
    "parse_query_plan",
]
