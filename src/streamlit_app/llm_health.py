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
from typing import Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

from src.config import load_config
from src.llm.natural_language import LLMConfig, load_llm_config


_PROBE_TIMEOUT = 5.0  # generous enough for remote Ollama over the internet
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


_GROQ_PROBE_MODEL = "llama-3.1-8b-instant"  # fast + cheap for health checks


def _groq_http_error_detail(exc: HTTPError) -> str:
    """Extract Groq's error message from an HTTPError response body."""
    try:
        msg = json.loads(exc.read()).get("error", {}).get("message", exc.reason)
    except Exception:
        msg = exc.reason
    return f"HTTP {exc.code}: {msg}"


def _probe_groq(cfg: LLMConfig) -> ProbeResult:
    """Probe Groq with a minimal chat completion (max_tokens=1).

    We deliberately avoid GET /openai/v1/models because some API key
    types return 403 on that endpoint even though chat completions work
    fine.  A tiny completion directly tests what the app actually uses.
    """
    if not cfg.api_key:
        return ProbeResult(
            status="offline", host="api.groq.com", model=cfg.model,
            detail="no API key — paste it in ⚙ LLM model",
        )
    model = cfg.model or _GROQ_PROBE_MODEL
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "stream": False,
    }).encode("utf-8")
    request = Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
    )
    try:
        with urlopen(request, timeout=_PROBE_TIMEOUT) as response:  # nosec
            body = response.read()
    except HTTPError as exc:
        return ProbeResult(
            status="offline", host="api.groq.com", model=model,
            detail=_groq_http_error_detail(exc),
        )
    except URLError as exc:
        return ProbeResult(
            status="offline", host="api.groq.com", model=model,
            detail=f"network error: {exc.reason}",
        )
    except (TimeoutError, OSError) as exc:
        return ProbeResult(status="offline", host="api.groq.com", model=model, detail=str(exc))

    try:
        envelope = json.loads(body)
    except json.JSONDecodeError:
        return ProbeResult(status="offline", host="api.groq.com", model=model, detail="non-JSON response from Groq")

    if not envelope.get("choices"):
        return ProbeResult(
            status="offline", host="api.groq.com", model=model,
            detail=f"unexpected response shape: {str(envelope)[:120]}",
        )
    return ProbeResult(
        status="ok",
        host="api.groq.com",
        model=model,
        available_models=tuple(_GROQ_MODELS_FALLBACK),
    )


# Static list used when probe can't enumerate models from Groq
_GROQ_MODELS_FALLBACK = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]


_GEMINI_MODELS_FALLBACK = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]


def _probe_openai_compatible(cfg: LLMConfig, base_url: str, provider_name: str, fallback_models: list) -> ProbeResult:
    """Generic probe for any OpenAI-compatible endpoint using a max_tokens=1 completion."""
    if not cfg.api_key:
        return ProbeResult(
            status="offline", host=base_url, model=cfg.model,
            detail=f"no API key — paste it in ⚙ LLM model",
        )
    payload = json.dumps({
        "model": cfg.model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "stream": False,
    }).encode("utf-8")
    request = Request(
        base_url + "/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
    )
    try:
        with urlopen(request, timeout=_PROBE_TIMEOUT) as response:  # nosec
            body = response.read()
    except HTTPError as exc:
        try:
            err_msg = json.loads(exc.read()).get("error", {}).get("message", exc.reason)
        except Exception:
            err_msg = exc.reason
        return ProbeResult(
            status="offline", host=base_url, model=cfg.model,
            detail=f"HTTP {exc.code}: {err_msg}",
        )
    except URLError as exc:
        return ProbeResult(status="offline", host=base_url, model=cfg.model, detail=f"network error: {exc.reason}")
    except (TimeoutError, OSError) as exc:
        return ProbeResult(status="offline", host=base_url, model=cfg.model, detail=str(exc))

    try:
        envelope = json.loads(body)
    except json.JSONDecodeError:
        return ProbeResult(status="offline", host=base_url, model=cfg.model, detail="non-JSON response")

    if not envelope.get("choices"):
        return ProbeResult(
            status="offline", host=base_url, model=cfg.model,
            detail=f"unexpected response: {str(envelope)[:120]}",
        )
    return ProbeResult(
        status="ok",
        host=base_url,
        model=cfg.model,
        available_models=tuple(fallback_models),
    )


def _do_probe(cfg: LLMConfig) -> ProbeResult:
    if cfg.provider == "groq":
        return _probe_groq(cfg)
    if cfg.provider == "gemini":
        return _probe_openai_compatible(
            cfg,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            provider_name="Gemini",
            fallback_models=_GEMINI_MODELS_FALLBACK,
        )
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
