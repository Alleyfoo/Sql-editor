# Vendored Code

This project intentionally reuses small, targeted pieces of code from earlier
Alleyfoo projects. Each vendored file is kept verbatim (modulo a header
comment with source + SHA + license) under `vendor/<source_repo>/...` and is
**not** on the Python import path by default. Active code lives under `src/`;
vendored files are references that document where patterns came from.

All three upstream projects are MIT-licensed (Copyright (c) 2025 Alleyfoo),
which permits redistribution with attribution.

## vendor/data-tool-demo/config_loader.py

- Source repo: <https://github.com/Alleyfoo/Data-tool-demo>
- Commit SHA:  `dae9eeb14c8780ff86449895425fdc9e6c22ab61`
- Source path: `src/core/config_loader.py`
- License:     MIT
- Used by:     Pattern reference for `src/config.py` (empty-on-missing,
  `yaml.safe_load` only, dict-or-empty coercion). The project uses a slimmed
  generic version rather than the synonyms-specific original, since this tool
  has different config needs.

## Patterns referenced but not vendored

### Ollama client + env-var configuration — Support-triage-llm

- Source repo: <https://github.com/Alleyfoo/Support-triage-llm>
- Commit SHA:  `4ef43070ae40b9dffcccc11cd5756091dae46438`
- Source paths:
  - `app/slm_ollama.py` — stdlib-only `urllib.request` POST pattern to
    `/api/chat` with `stream: false`.
  - `app/config.py` (lines 56–60) — env-var layering
    (`OLLAMA_HOST` / `OLLAMA_MODEL` / `OLLAMA_TIMEOUT`).

Used as pattern references in Phase 3 `src/llm/natural_language.py`
(`OllamaClient`, `load_llm_config`). No source file is copied — the
Phase 3 module is rewritten to request Ollama's native `format: "json"`
mode and to raise a typed `LLMError` rather than falling back to a stub.

### LLM output JSON validation — slm-cleanroom-demo

- Source repo: <https://github.com/Alleyfoo/slm-cleanroom-demo>
- Commit SHA:  `4fbcf845293d066a070326fc8ccdb8290e1fdda8`

The principle of validating LLM output against a known schema before
executing is applied in `src/llm/natural_language.py::parse_query_plan`.
We deliberately avoid adding a `pydantic` dependency — validation is
~60 lines of hand-rolled checks against the active dataset schema and
the same operator / function / direction allow-lists used by the visual
composer. No code is vendored from this repo.
