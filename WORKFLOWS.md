# Agentic Workflows

Query Studio includes a layered agentic execution system used for evaluation and
future feature work. This document explains the architecture, the data contracts
between layers, and includes a worked example showing how a question flows through
the full pipeline.

---

## Architecture overview

```
User question
      │
      ▼
┌─────────────────────┐
│  CentralCoordinator │  ← classifies task, builds & executes plan
│  (orchestration)    │
└──────────┬──────────┘
           │  dispatches to workers via opaque DataHandles
           ▼
┌─────────────────────┐
│ MixedExecutionEngine│  ← routes SQL/Python/hybrid, runs query
│ (mixed_execution)   │
└──────────┬──────────┘
           │  returns result DataFrame in a DataHandle
           ▼
┌─────────────────────┐
│ AnalysisCoordinator │  ← profiles data, plans analysis, generates insights
│ (analysis_lane)     │
└─────────────────────┘
```

Each layer is independently testable. The streamlit UI only uses
`MixedExecutionEngine` indirectly (via `QueryModel.to_sql()` + `executor.execute()`).
The orchestration and analysis layers are wired together in evaluation harnesses
and can be composed for deeper agentic workflows.

---

## Layer 1 — CentralCoordinator

**File:** `src/orchestration/runtime.py`

Classifies an incoming question into one of six task classes, builds a typed
`OrchestrationPlan` (a list of worker steps), then executes it step-by-step.
Data moves between workers as opaque `DataHandle` IDs stored in a `HandleStore`;
workers never share raw DataFrames directly.

### Task classes

| Class | Trigger | Workers used |
|---|---|---|
| `pushdown` | Simple filter / aggregation / sort | `mixed_executor_worker` → `validator_worker` |
| `hybrid` | Rolling window or moving average with dates | `mixed_executor_worker` → `validator_worker` |
| `python_first` | Percentile / stddev / outlier detection | `mixed_executor_worker` → `validator_worker` |
| `cleaning_first` | Low header confidence or messy source | `cleaning_worker` → `mixed_executor_worker` → `validator_worker` |
| `follow_up` | References "that result" / "previous result" | `followup_worker` → `validator_worker` |
| `adversarial` | SQL-injection patterns or unsafe keywords | `reject_worker` (immediate stop) |

### Workers

- **`cleaning_worker`** — emits a lightweight `cleaned_source` artifact (drops empty columns, detects row offset); does not materialise a full copy
- **`mixed_executor_worker`** — delegates to `MixedExecutionEngine`, stores result DataFrame as a `result_table` handle
- **`followup_worker`** — applies a simple follow-up transform (count / top-N / first-N / average) to a prior result handle
- **`validator_worker`** — runs a named validator function against the result; validators live in `eval/open_data_sql_vs_python_eval.py`
- **`reject_worker`** — raises a `safety_violation` error, terminates the plan

### Worked example

```python
from src.orchestration.runtime import (
    CentralCoordinator,
    SourceManifest,
    HandleStore,
    default_capability_manifest,
)
from src.ingestion import infer_schema
import pandas as pd

df = pd.read_csv("data/demo/sample_sales.csv")
schema = infer_schema(df)

handle_store = HandleStore()
source_handle = handle_store.put(
    df,
    handle_type="source",
    source_id="demo",
    metadata={"header_confidence": 1.0},
).handle_id

source_manifest = SourceManifest(
    source_id="demo",
    name="sample_sales.csv",
    kind="csv",
    path="data/demo/sample_sales.csv",
    schema=schema,
    rows_estimate=len(df),
    header_confidence=1.0,
)

coordinator = CentralCoordinator()
plan = coordinator.build_plan(
    question="top 10 customers by total revenue",
    source_handle=source_handle,
    validator="top_n_revenue",
    header_confidence=1.0,
    prior_result_handle=None,
)
# plan.task_class == "pushdown"
# plan.steps == [mixed_executor_worker, validator_worker]

run = coordinator.execute_plan(
    plan=plan,
    question="top 10 customers by total revenue",
    source_manifest=source_manifest,
    capability_manifest=default_capability_manifest(),
    handle_store=handle_store,
    source_df=df,
    source_handle=source_handle,
    payload_budget={"max_bytes_materialized": 50_000_000},
    expected_task_class="pushdown",
    expected_workers=["mixed_executor_worker", "validator_worker"],
    validator="top_n_revenue",
)

print(f"task_class_correct: {run.task_classification_correct}")
print(f"final_output_correct: {run.final_output_correct}")
print(f"hops: {[h['chosen_worker'] for h in run.hops]}")
```

---

## Layer 2 — MixedExecutionEngine

**File:** `src/mixed_execution/engine.py`

Translates a natural-language question directly into a query result without going
through the UI. Internally it:

1. Plans a `LogicalPlan` (filters, aggregates, post-processing steps)
2. Scores four execution routes and picks the best one
3. Executes via the chosen backend

### Execution routes

| Route | When | Backend |
|---|---|---|
| `pushdown` | No analytics post-processing; large or remote source | `SQLPushdownBackend` or `DataFramePushdownBackend` |
| `hybrid` | Analytics needed but also selective filters | SQL pre-filter → `PythonAnalyticsExecutor` |
| `python` | Pure analytics; small local source | `PythonAnalyticsExecutor` |
| `cleaning_first` | Header confidence < 0.9 | Clean → re-route |

### Worked example

```python
from src.mixed_execution.engine import MixedExecutionEngine

engine = MixedExecutionEngine()
run = engine.run(
    question="monthly revenue trend 2024",
    dataset_path="data/demo/sample_sales.csv",
)

print(f"route: {run.route}")          # e.g. "pushdown"
print(f"backend: {run.backend}")      # e.g. "sql"
print(f"rows: {run.rows_materialized}")
print(run.result.dataframe.head())
```

The `EngineRun` result includes `routing_artifact` — a structured dict explaining
why the route was chosen — useful for debugging and evaluation.

---

## Layer 3 — AnalysisCoordinator

**File:** `src/analysis_lane/engine.py`

Takes a distilled DataFrame (the query result) and a question, then orchestrates
a multi-worker analysis pipeline.

### Workers (in order)

1. **`profiling_worker`** — computes `AnalysisProfile`: dimensions, metrics, null rates, date coverage, top categories, trend hints, anomaly hints
2. **`analysis_worker`** — infers analysis family (trend / segment_comparison / kpi_summary / dashboard_design), builds `AnalysisPlan`, generates `ChartSpec`s and `InsightReport` via LLM with heuristic fallback
3. **`chart_render_worker`** — validates each `ChartSpec` against the data and emits render artifacts
4. **`dashboard_render_worker`** — assembles a `DashboardSpec` from tiles when requested

### Analysis families

| Family | Trigger phrase examples | Outputs |
|---|---|---|
| `trend` | "over time", "monthly", "time series" | `InsightReport` + line `ChartSpec` |
| `segment_comparison` | "by region", "compare segments" | `InsightReport` + bar `ChartSpec` |
| `kpi_summary` | "kpi", "snapshot", default | `InsightReport` |
| `dashboard_design` | "dashboard" | `InsightReport` + multiple `ChartSpec`s + `DashboardSpec` |
| `follow_up_analysis` | "that", "same", prior handle present | `InsightReport` linked to prior |
| `guardrail` | "why", "cause", "driver" | `InsightReport` with blocked causal claims |

### Guardrails

The `AnalysisRun` result includes `guardrails_clean` and a list of `guardrail_errors`.
Guardrails fire when:
- A claim uses causal language ("X causes Y") in a descriptive-only context
- A timeseries chart is requested but the time dimension is missing
- Dashboard tiles reference metrics not selected in the plan
- Insights cite weak evidence (no numeric support)

### Worked example

```python
from src.analysis_lane.engine import AnalysisCoordinator
from src.ingestion import load_csv, infer_schema
import pandas as pd

# Typically you'd run MixedExecutionEngine first and pass its result here.
# For a standalone example, load a CSV directly.
df = pd.read_csv("data/demo/sample_sales.csv")
_, schema = load_csv("data/demo/sample_sales.csv")

coordinator = AnalysisCoordinator()  # picks up config.yaml automatically
run = coordinator.run(
    question="monthly revenue trend across 2024",
    distilled_df=df,
    schema=schema,
    prior_analysis_handle=None,
    expected_followup_from=None,
)

print(f"family: {run.plan.family}")           # "trend"
print(f"guardrails clean: {run.guardrails_clean}")
print(f"insight summary: {run.report.summary}")
for chart in run.charts:
    print(f"  chart: {chart.chart_type} — {chart.title}")
```

---

## Chaining all three layers

The typical eval harness wires the layers end-to-end:

```
question + CSV path
    │
    └─► CentralCoordinator.build_plan()
              │
              └─► execute_plan()
                        │
                        └─► MixedExecutionEngine.run()  [inside mixed_executor_worker]
                                  │
                                  └─► result DataFrame
                                            │
                                            └─► AnalysisCoordinator.run()
                                                      │
                                                      └─► AnalysisRun (insights, charts, guardrails)
```

The `CentralCoordinator` only calls `MixedExecutionEngine` today. Wiring in
`AnalysisCoordinator` as a final step after `validator_worker` is the natural
next extension.

---

## LLM provider configuration

All three layers respect the same `config.yaml` / `LLMConfig` system:

```yaml
# config.yaml
llm:
  provider: ollama          # ollama | ollama_remote | groq | openai_compatible
  host: http://localhost:11434
  model: qwen2.5-coder:7b
  timeout: 120
```

For cloud providers, set the API key as an environment variable:

```bash
export GROQ_API_KEY=gsk_...       # for provider: groq
export LLM_API_KEY=sk-...         # for provider: openai_compatible
```

The `AnalysisCoordinator` and the streamlit UI's "Ask + Analyze" button both use
`make_llm_client(cfg)` — switching the provider in `config.yaml` or the UI popover
propagates to all layers automatically.
