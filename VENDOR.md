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

### Ollama env-var configuration — Support-triage-llm

- Source repo: <https://github.com/Alleyfoo/Support-triage-llm>
- Commit SHA:  `4ef43070ae40b9dffcccc11cd5756091dae46438`
- Source path: `app/config.py` (lines 56–60)

The `OLLAMA_HOST` / `OLLAMA_MODEL` / `OLLAMA_TIMEOUT` env-var pattern is
noted in `config.yaml` under the `llm:` section for Phase 3. No code is
imported in Phase 1 because the LLM hook does not yet exist.

### LLM output JSON validation — slm-cleanroom-demo

- Source repo: <https://github.com/Alleyfoo/slm-cleanroom-demo>
- Commit SHA:  `4fbcf845293d066a070326fc8ccdb8290e1fdda8`

The idea of validating LLM output against a schema before executing will be
reused in Phase 3 when the LLM natural-language hook is implemented. No code
is vendored in Phase 1.
