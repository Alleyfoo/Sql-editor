"""Quick Groq connectivity test — run outside Streamlit to isolate issues.

Usage:
    python scripts/test_groq.py gsk_YOUR_KEY_HERE
    python scripts/test_groq.py gsk_YOUR_KEY_HERE llama-3.1-8b-instant

What it tests:
    1. List available models  (GET /openai/v1/models)
    2. Single chat completion  (POST /openai/v1/chat/completions)
       — asks the model to return {"ok": true} so we validate JSON mode too.
"""

from __future__ import annotations

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GROQ_BASE = "https://api.groq.com/openai/v1"
TIMEOUT = 15.0
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _get(url: str, api_key: str) -> dict:
    req = Request(url, headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"})
    with urlopen(req, timeout=TIMEOUT) as resp:  # nosec
        return json.loads(resp.read())


def _post(endpoint: str, payload: dict, api_key: str) -> dict:
    data = json.dumps(payload).encode()
    req = Request(
        GROQ_BASE + endpoint,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:  # nosec
            return json.loads(resp.read())
    except HTTPError as exc:
        body = exc.read()
        try:
            msg = json.loads(body).get("error", {}).get("message", exc.reason)
        except Exception:
            msg = exc.reason
        raise RuntimeError(f"HTTP {exc.code}: {msg}") from exc


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_groq.py <api_key> [model]")
        sys.exit(1)

    api_key = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    # ── Step 1: list models ────────────────────────────────────────────────
    print("1. Listing Groq models …")
    try:
        data = _get(f"{GROQ_BASE}/models", api_key)
        models = [e["id"] for e in data.get("data", []) if isinstance(e, dict)]
        print(f"   ✓ {len(models)} models: {', '.join(models[:6])}{'…' if len(models) > 6 else ''}")
        if model not in models:
            print(f"   ⚠ '{model}' not in the list — pick one from above")
    except HTTPError as exc:
        print(f"   ✗ HTTP {exc.code}: {exc.reason}")
        sys.exit(1)
    except URLError as exc:
        print(f"   ✗ network error: {exc.reason}")
        sys.exit(1)

    # ── Step 2: chat completion (JSON mode) ────────────────────────────────
    print(f"2. Chat completion with '{model}' (JSON mode) …")
    try:
        resp = _post(
            "/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Reply with exactly this JSON: {\"ok\": true}"},
                    {"role": "user", "content": "ping"},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
                "stream": False,
            },
            api_key,
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if parsed.get("ok") is True:
            print(f"   ✓ Got {content!r} — Groq is working correctly")
        else:
            print(f"   ⚠ Unexpected response: {content!r}")
    except RuntimeError as exc:
        print(f"   ✗ {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"   ✗ unexpected: {exc}")
        sys.exit(1)

    print("\nAll checks passed — Groq connection is healthy.")


if __name__ == "__main__":
    main()
