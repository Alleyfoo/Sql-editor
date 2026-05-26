"""Quick Groq connectivity test — run outside Streamlit to isolate issues.

Usage:
    python scripts/test_groq.py gsk_YOUR_KEY_HERE
    python scripts/test_groq.py gsk_YOUR_KEY_HERE llama-3.1-8b-instant

Tests:
    1. Minimal chat completion with max_tokens=1  (the real health check)
    2. JSON-mode completion                        (what NL Ask uses)
    3. List available models                       (bonus — may 403, that's OK)
"""

from __future__ import annotations

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GROQ_BASE = "https://api.groq.com/openai/v1"
TIMEOUT = 15.0
DEFAULT_MODEL = "llama-3.1-8b-instant"


def _read_error(exc: HTTPError) -> str:
    try:
        return json.loads(exc.read()).get("error", {}).get("message", exc.reason)
    except Exception:
        return exc.reason


def _post(endpoint: str, payload: dict, api_key: str) -> dict:
    data = json.dumps(payload).encode()
    req = Request(
        GROQ_BASE + endpoint,
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
        print("Usage: python scripts/test_groq.py <api_key> [model]")
        sys.exit(1)

    api_key = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    all_ok = True

    # ── Test 1: minimal chat completion ───────────────────────────────────
    print(f"1. Minimal chat completion (model={model}, max_tokens=1) …")
    try:
        resp = _post("/chat/completions", {
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
        resp = _post("/chat/completions", {
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

    # ── Test 3: list models (may 403 on restricted keys — not critical) ───
    print("3. List available models (GET /models — may 403 on some keys) …")
    try:
        from urllib.request import Request as Req
        req = Req(
            f"{GROQ_BASE}/models",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        )
        with urlopen(req, timeout=TIMEOUT) as resp:  # nosec
            data = json.loads(resp.read())
        models = [e["id"] for e in data.get("data", []) if isinstance(e, dict)]
        print(f"   ✓ {len(models)} models: {', '.join(models[:6])}{'…' if len(models) > 6 else ''}")
    except HTTPError as exc:
        print(f"   ⚠ HTTP {exc.code} (key works for completions — this is fine)")
    except Exception as exc:
        print(f"   ⚠ {exc}")

    print()
    if all_ok:
        print("✓ Groq is fully functional for Query Studio.")
    else:
        print("✗ One or more checks failed — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
