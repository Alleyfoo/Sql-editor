---
name: local-data-agent
description: Local data-handling skill for deterministic CSV analytics benchmarking. Use for intake/EDA framing, query-validation checkpoints, and operation-plan execution without external services.
---

# Local Data Agent

This skill is designed for local benchmarking workflows.

## Principles

- Keep execution deterministic and reproducible.
- Validate operation plans before execution.
- Prefer pandas-native computations for Python-fit analytics tasks.
- Keep outputs schema-stable for validator-based evaluation.

## Workflow

1. Data intake and schema sanity check.
2. Intent classification to an operation ID.
3. Plan validation against supported operations.
4. Deterministic execution.
5. Post-execution shape/quality sanity checks.

## Profile

Default profile: `profiles/local_v1.json`

