"""Quick cloud LLM connectivity test — run outside Streamlit to isolate issues.

Usage:
    python scripts/test_groq.py <api_key> [provider] [model]

    provider: gemini (default) | groq
    model:    gemini-2.0-flash (default for gemini) | llama-3.1-8b-instant (groq)

Examples:
    python scripts/test_groq.py AIzaSy_KEY                        # Gemini
    python scripts/test_groq.py gsk_KEY groq                       # Groq
    python scripts/test_groq.py AIzaSy_KEY gemini gemini-1.5-flash # specific model

Tests:
    1. Minimal chat completion  (the real health check)
    2. JSON-mode completion     (what NL Ask uses)
"""

from __future__ import annotations

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TIMEOUT = 15.0

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.0-flash",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-8b-instant",
    },
}


def _read_error(exc: HTTPError) -> str:
    try:
        return json.loads(exc.read()).get("error", {}).get("message", exc.reason)
    except Exception:
        return exc.reason


def _post(base_url: str, endpoint: str, payload: dict, api_key: str) -> dict:
    data = json.dumps(payload).encode()
    req = Request(
        base_url + endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:  # nosec
            return json.loads(resp.read())
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {_read_error(exc)}") from exc
    except URLError as exc:
        raise RuntimeError(f"network error: {exc.reason}") from exc


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_groq.py <api_key> [provider] [model]")
        print("       provider: gemini (default) | groq")
        sys.exit(1)

    api_key  = sys.argv[1]
    provider = sys.argv[2] if len(sys.argv) > 2 else "gemini"
    if provider not in PROVIDERS:
        print(f"Unknown provider '{provider}'. Choose: {', '.join(PROVIDERS)}")
        sys.exit(1)

    cfg      = PROVIDERS[provider]
    base_url = cfg["base_url"]
    model    = sys.argv[3] if len(sys.argv) > 3 else cfg["default_model"]

    print(f"Provider : {provider}  ({base_url})")
    print(f"Model    : {model}")
    print()

    all_ok = True

    # ── Test 1: minimal completion ────────────────────────────────────────
    print(f"1. Minimal chat completion (max_tokens=1) …")
    try:
        resp = _post(base_url, "/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "stream": False,
        }, api_key)
        if resp.get("choices"):
            print("   ✓ Got a response — API key and model are working")
        else:
            print(f"   ⚠ Unexpected shape: {str(resp)[:200]}")
            all_ok = False
    except RuntimeError as exc:
        print(f"   ✗ {exc}")
        all_ok = False

    # ── Test 2: JSON-mode completion ──────────────────────────────────────
    print(f"2. JSON-mode completion (response_format=json_object) …")
    try:
        resp = _post(base_url, "/chat/completions", {
            "model": model,
            "messages": [
                {"role": "system", "content": 'Reply with exactly this JSON object: {"ok": true}'},
                {"role": "user", "content": "ping"},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "stream": False,
        }, api_key)
        content = resp["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if parsed.get("ok") is True:
            print(f"   ✓ Got {content!r}")
        else:
            print(f"   ⚠ Unexpected JSON: {content!r}")
            all_ok = False
    except RuntimeError as exc:
        print(f"   ✗ {exc}")
        all_ok = False
    except Exception as exc:
        print(f"   ✗ {exc}")
        all_ok = False

    print()
    if all_ok:
        print(f"✓ {provider.capitalize()} is fully functional for Query Studio.")
    else:
        print("✗ One or more checks failed — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
