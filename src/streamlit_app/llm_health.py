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

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _do_probe(cfg: LLMConfig) -> ProbeResult:
    url = cfg.host.rstrip("/") + "/api/tags"
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=_PROBE_TIMEOUT) as response:  # nosec
            body = response.read()
    except HTTPError as exc:
        return ProbeResult(
            status="offline",
            host=cfg.host,
            model=cfg.model,
            detail=f"HTTP {exc.code}: {exc.reason}",
        )
    except URLError as exc:
        return ProbeResult(
            status="offline",
            host=cfg.host,
            model=cfg.model,
            detail=f"unreachable ({exc.reason})",
        )
    except TimeoutError:
        return ProbeResult(
            status="offline",
            host=cfg.host,
            model=cfg.model,
            detail=f"no response within {_PROBE_TIMEOUT:.0f}s",
        )
    except OSError as exc:
        return ProbeResult(
            status="offline",
            host=cfg.host,
            model=cfg.model,
            detail=f"transport error: {exc}",
        )

    try:
        envelope = json.loads(body)
    except json.JSONDecodeError:
        return ProbeResult(
            status="offline",
            host=cfg.host,
            model=cfg.model,
            detail="non-JSON response from /api/tags",
        )

    # Best-effort: confirm the configured model is actually pulled.
    available = []
    if isinstance(envelope, dict):
        for entry in envelope.get("models", []) or []:
            name = entry.get("name") if isinstance(entry, dict) else None
            if isinstance(name, str):
                available.append(name)

    if available and not any(
        name == cfg.model or name.startswith(cfg.model + ":") for name in available
    ):
        detail = (
            f"server up, but model '{cfg.model}' is not pulled "
            f"(available: {', '.join(available[:4])}"
            + ("…" if len(available) > 4 else "")
            + ")"
        )
        return ProbeResult(status="offline", host=cfg.host, model=cfg.model, detail=detail)

    return ProbeResult(status="ok", host=cfg.host, model=cfg.model)


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
