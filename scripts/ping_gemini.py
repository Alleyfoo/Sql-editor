"""Single minimal Gemini call.
Usage: python scripts/ping_gemini.py <api_key> [model]
Default model: gemini-2.0-flash
"""
import json, sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

key   = sys.argv[1] if len(sys.argv) > 1 else input("API key: ").strip()
model = sys.argv[2] if len(sys.argv) > 2 else "gemini-2.5-flash"

print(f"Pinging {model} …")
req = Request(
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    data=json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    },
)
try:
    with urlopen(req, timeout=15) as r:  # nosec
        body = json.loads(r.read())
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "(empty)")
        print(f"✓ OK — model replied: {content!r}")
except HTTPError as e:
    try:
        msg = json.loads(e.read()).get("error", {}).get("message", e.reason)
    except Exception:
        msg = e.reason
    print(f"✗ HTTP {e.code}: {msg}")
except URLError as e:
    print(f"✗ Network error: {e.reason}")
