# Planner-First Mixed Execution Architecture

## Goal
Replace SQL-first orchestration with:

`NL -> typed LogicalPlan -> deterministic router -> pushdown/hybrid/python execution -> validated result`

SQL is a pushdown backend, not the canonical abstraction.

## Core Contracts

1. `LogicalPlan` is the canonical internal contract.
2. `PlanValidator` rejects unknown columns/functions/operators before execution.
3. `Router` returns one of:
   - `pushdown`
   - `hybrid`
   - `python`
   - `cleaning_first`
4. `PushdownBackend` interface isolates backend-specific execution.
5. `PythonAnalyticsExecutor` is the single consolidated path for rolling/percentile/repair logic.
6. `ResultValidator` enforces output schema and type sanity before result acceptance.

## Layering

1. Planning layer (`src/mixed_execution/planner.py`)
2. Router layer (`src/mixed_execution/router.py`)
3. Pushdown layer (`src/mixed_execution/pushdown.py`)
4. Python analytics layer (`src/mixed_execution/python_analytics.py`)
5. Final validation layer (`src/mixed_execution/result_validator.py`)

## Safety Rules

- Executor never receives raw NL text.
- Plan validation is mandatory.
- Unknown columns are hard-rejected.
- Schema correctness is measured independently from execution correctness.

## Benchmarking

`eval/open_data_mixed_execution_eval.py` reports:

- `plan_valid`
- `route_correct`
- `execution_correct`
- `schema_correct`
- `safety_pass`
- `rows_scanned`
- `rows_materialized`
- `bytes_fetched`
- `peak_memory_mb`
- `fallback_used`
- `fallback_reason`

This makes correctness and payload cost visible in the same report.

