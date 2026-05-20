"""Lightweight Ollama health probe for the Streamlit app.

Hits ``GET {host}/api/tags`` with a short timeout and reports whether
the local LLM is reachable and the configured model is pulled.  Used by
the header status pill so the user can see at a glance whether NL
features will work.

Result is cached in ``st.session_state`` with a 30-second TTL so stale
green status clears quickly when Ollama goes offline or a model is
removed; pass ``force=True`` to bypass the TTL.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

from src.config import load_config
from src.llm.natural_language import LLMConfig, load_llm_config


_PROBE_TIMEOUT = 2.0
_SESSION_KEY = "_llm_probe_result"
_CACHE_TTL = 30.0  # seconds before a re-probe is triggered


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a single Ollama health probe."""

    status: str  # "ok" | "offline"
    host: str
    model: str
    detail: str = ""
    available_models: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _do_probe(cfg: LLMConfig) -> ProbeResult:
    provider = (cfg.provider or "ollama").lower()
    if provider in ("groq", "openai_compatible"):
        return _probe_openai_compatible(cfg)
    return _probe_ollama(cfg)


def _probe_ollama(cfg: LLMConfig) -> ProbeResult:
    """Probe an Ollama server via GET /api/tags."""
    url = cfg.host.rstrip("/") + "/api/tags"
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=_PROBE_TIMEOUT) as response:  # nosec
            body = response.read()
    except HTTPError as exc:
        return ProbeResult(
            status="offline", host=cfg.host, model=cfg.model,
            detail=f"HTTP {exc.code}: {exc.reason}",
        )
    except URLError as exc:
        return ProbeResult(
            status="offline", host=cfg.host, model=cfg.model,
            detail=f"unreachable ({exc.reason})",
        )
    except (TimeoutError, OSError) as exc:
        return ProbeResult(
            status="offline", host=cfg.host, model=cfg.model,
            detail=str(exc),
        )

    try:
        envelope = json.loads(body)
    except json.JSONDecodeError:
        return ProbeResult(
            status="offline", host=cfg.host, model=cfg.model,
            detail="non-JSON response from /api/tags",
        )

    available = []
    if isinstance(envelope, dict):
        for entry in envelope.get("models", []) or []:
            name = entry.get("name") if isinstance(entry, dict) else None
            if isinstance(name, str):
                available.append(name)

    models_tuple = tuple(available)

    if available and not any(
        name == cfg.model or name.startswith(cfg.model + ":") for name in available
    ):
        detail = (
            f"server up, but '{cfg.model}' is not pulled "
            f"(available: {', '.join(available[:4])}"
            + ("…" if len(available) > 4 else "")
            + ")"
        )
        return ProbeResult(
            status="offline", host=cfg.host, model=cfg.model,
            detail=detail, available_models=models_tuple,
        )

    return ProbeResult(
        status="ok", host=cfg.host, model=cfg.model,
        available_models=models_tuple,
    )


def _probe_openai_compatible(cfg: LLMConfig) -> ProbeResult:
    """Probe Groq / OpenAI-compatible endpoints via GET /v1/models."""
    from src.llm.natural_language import GROQ_HOST, GROQ_MODELS

    if not cfg.api_key:
        return ProbeResult(
            status="offline", host=cfg.host, model=cfg.model,
            detail="API key not set",
        )

    host = GROQ_HOST if cfg.provider == "groq" else cfg.host.rstrip("/")
    url = host + "/v1/models"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
    )
    try:
        with urlopen(request, timeout=_PROBE_TIMEOUT) as response:  # nosec
            body = response.read()
    except HTTPError as exc:
        detail = "invalid API key" if exc.code == 401 else f"HTTP {exc.code}"
        return ProbeResult(
            status="offline", host=host, model=cfg.model, detail=detail,
        )
    except (URLError, TimeoutError, OSError) as exc:
        return ProbeResult(
            status="offline", host=host, model=cfg.model, detail=str(exc),
        )

    try:
        envelope = json.loads(body)
    except json.JSONDecodeError:
        return ProbeResult(
            status="offline", host=host, model=cfg.model,
            detail="non-JSON response",
        )

    # Pull model list from the /v1/models response
    available = []
    for entry in (envelope.get("data") or []):
        mid = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(mid, str):
            available.append(mid)

    # Fallback: for Groq, we know the model list
    if not available and cfg.provider == "groq":
        available = list(GROQ_MODELS)

    provider_label = "Groq" if cfg.provider == "groq" else host
    return ProbeResult(
        status="ok",
        host=host,
        model=cfg.model,
        detail=f"{provider_label} · {len(available)} models",
        available_models=tuple(available),
    )


def probe_ollama(*, force: bool = False) -> ProbeResult:
    """Return the cached probe result, re-probing when the TTL has expired."""
    cached: Optional[Tuple[ProbeResult, float]] = st.session_state.get(_SESSION_KEY)
    if cached is not None and not force:
        result, ts = cached
        if time.monotonic() - ts < _CACHE_TTL:
            return result
    cfg = load_llm_config(load_config() or {})
    result = _do_probe(cfg)
    st.session_state[_SESSION_KEY] = (result, time.monotonic())
    return result


def clear_cache() -> None:
    """Drop any cached probe so the next call re-checks immediately."""
    st.session_state.pop(_SESSION_KEY, None)


__all__ = ["ProbeResult", "probe_ollama", "clear_cache"]
