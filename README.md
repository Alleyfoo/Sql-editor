# Planner-First Mixed Execution

This project is a local analytics system built around a mixed execution
pipeline:

1. natural language input is converted to a typed logical plan
2. a deterministic router chooses `pushdown`, `hybrid`, `python`, or `cleaning_first`
3. execution runs through pushdown backends and/or Python analytics
4. output schema validation and safety checks gate the final result

SQL is a backend, not the primary abstraction. The desktop GUI (Tkinter)
and SQL preview are interface layers for inspectability and learning, not
the core execution model.

Downstream of the cleaning pipeline, not part of it: the tool never writes
to your CSV and the SQLite database is in-memory and locked read-only.

## Status

**Planner-first mixed execution active** with local benchmark coverage for:

- typed logical-plan validation before execution
- deterministic route decisions with route-oracle evaluation
- payload-aware pass/fail budgets (materialization and bytes fetched)
- schema validation and deterministic repair before output

The legacy visual query builder and SQL preview remain supported as one
interaction path. Multi-table JOIN composition (Phase 4 UI roadmap) is
still pending.
## Install

```bash
pip install -r requirements.txt
```

`tkinter` and `sqlite3` ship with Python. On most Linux distros Tk may be
a separate package (e.g. `sudo apt install python3-tk`).

## Run

```bash
python main.py
```

Then **File â†’ Open CSVâ€¦** to load a file. As you click columns in the left
panel and add filters in the center panel, the generated SQL updates live in
the right panel. Press **Run** to execute and see results in the table at
the bottom, then **Export CSVâ€¦** to save them elsewhere.

### Natural-language input

The bar at the top of the window (**Ask in natural language**) sends
your request to a local Ollama model and populates the visual composer
with the result. The SQL preview updates so you can inspect it; clicking
**Run** then executes the query as normal. The NL flow never auto-runs.

Use **Ask + Analyze** for the agent flow:

1. NL -> JSON plan -> SQL
2. Execute SQL
3. Return a concise analysis block (summary, key insights, follow-up questions)

This keeps SQL visible in the chat and preview so query-learning remains
first-class.

Configure the Ollama endpoint in `config.yaml` or via environment
variables (env vars win):

| Variable          | Default                     |
| ----------------- | --------------------------- |
| `OLLAMA_HOST`     | `http://localhost:11434`    |
| `OLLAMA_MODEL`    | `gemma3`                    |
| `OLLAMA_TIMEOUT`  | `60` (seconds)              |

Only the stdlib `urllib` is used â€” no extra dependency is added for the
Ollama client.

## Safety

This tool will never modify your data. The guarantees:

1. **Connection is read-only.** After the CSV is loaded into an in-memory
   SQLite DB, the connection is switched to read-only mode via
   `PRAGMA query_only = ON` and a SQLite authorizer that denies every
   non-SELECT action code. Any `INSERT` / `UPDATE` / `DELETE` / `DROP` /
   `ALTER` raises immediately.
2. **Only `SELECT` statements are generated.** The query model's
   `to_sql()` method self-validates: it refuses to emit anything that
   doesn't start with `SELECT` or that contains any DDL/DML keyword
   outside a quoted literal.
3. **Executor blocklist as second layer.** Every SQL string is re-validated
   by the executor before it touches SQLite. The blocklist covers `DROP`,
   `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `CREATE`, `ATTACH`, `DETACH`,
   `PRAGMA`, `REPLACE`, `TRUNCATE`, `EXEC`, `EXECUTE`, `GRANT`, `REVOKE`.
4. **No raw-SQL input field.** There is no place for the user â€” or the
   LLM â€” to type SQL. Every query is built from widgets or parsed from
   a validated JSON query plan.
5. **No database file written to disk.** The DB lives only in memory.
6. **LLM output is treated as untrusted input.** The natural-language
   flow asks Ollama to return a JSON query plan, not SQL. The plan is
   validated against the active dataset schema and the same operator /
   aggregation / order allow-lists the visual composer uses, converted
   to a `QueryModel`, and then passes through `to_sql()` and the
   executor blocklist exactly like a hand-built query. Column
   hallucinations, bad operators, and injection attempts inside filter
   values are all blocked before SQL is emitted.

## Testing

```bash
pytest tests/ -v
```

The suite covers query-model SQL generation (every operator, AND/OR logic,
LIMIT, SQL-injection attempts in filter values), the executor blocklist, and
ingestion including a live check that the returned connection rejects writes.

### Capability Spike Eval (Model Feasibility)

Before building larger agent orchestration, run a capability spike to measure
NL->JSON plan behavior against the current trust boundary:

```bash
python eval/capability_eval.py --provider mock
python eval/capability_eval.py --provider ollama --model gemma4
python eval/capability_eval.py --provider ollama --model gemma4 --cases eval/golden/capability/security_cases.json
```

The report includes JSON object rate, valid plan rate, hallucination rate,
latency, and token usage. Starter cases live in
`eval/golden/capability/starter_cases.json` and adversarial/security cases
live in `eval/golden/capability/security_cases.json`.

### Dirty Excel Header Capability Spike

To evaluate whether the model can detect displaced headers in messy Excel/CSV
exports:

```bash
python eval/cleaning_capability_eval.py --provider mock
python eval/cleaning_capability_eval.py --provider ollama --model gemma4
```

Cases and fixtures:

- `eval/golden/cleaning/dirty_excel_cases.json`
- `eval/golden/cleaning/fixtures/*.xlsx`
- `eval/golden/cleaning/generate_fixtures.py` (regenerates all fixtures)

The dirty-cleaning set now includes 19 harder displaced-header scenarios
(long preambles, blank separators, column offsets, sparse header cells,
symbols/units, multi-table previews, and mixed-language header text).

### Open-Data SQL vs Python-Fit Benchmark

To benchmark where NL->SQL is reliable versus where requests drift into
Python-style analytics (percentiles, rolling windows), run:

```bash
python eval/open_data_sql_vs_python_eval.py --provider ollama --model gemma4
```

This benchmark uses two open datasets:

- `data/open_data/usgs_all_month.csv`
- `data/open_data/seattle_weather.csv`

and evaluates two tracks:

- `sql_fit`: should pass with accurate SQL outputs.
- `python_fit`: intentionally harder analytics that often need Python/pandas.

The report is written to `eval/reports/open_data_sql_vs_python_*.json`.

NL routing is now enabled for Python-fit intents. Prompts containing
`percentile`, `quantile`, `rolling/moving average`, `stdev/standard deviation`,
or `outlier/anomaly` are routed away from SQL generation to a Python
analytics path.

To add your own dataset (for example HSY open data):

1. Place the CSV under `data/open_data/`.
2. Add cases to `eval/golden/open_data/sql_vs_python_cases.json`.
3. Use validator `non_empty_result` or `single_numeric_scalar` for quick
   smoke checks, or add a stricter validator in
   `eval/open_data_sql_vs_python_eval.py`.

HSY April 2021 eval pack is included at:

- `eval/golden/open_data/hsy_2021_04_eval_pack.json`

Run it with:

```bash
python eval/open_data_sql_vs_python_eval.py --provider ollama --model gemma4 --cases eval/golden/open_data/hsy_2021_04_eval_pack.json
```

### Open-Data Tri-Arm Benchmark (SQL vs Python vs Agent+Skills)

To compare all three execution styles on the same case set, run:

```bash
python eval/open_data_tri_arm_eval.py --provider ollama --model gemma4 --skill-profile local_v1
```

For a local/offline smoke run (no model call), use:

```bash
python eval/open_data_tri_arm_eval.py --provider mock --model mock --skill-profile local_v1
```

The tri-arm report is written to `eval/reports/open_data_tri_arm_*.json` and includes:

- `summary.by_arm.sql|python|skills`
- `summary.by_track.sql_fit|python_fit`
- SQL routing/confusion counters (`summary.sql_routing.*`)

Local skill profile files live in:

- `skills/local_data_agent/SKILL.md`
- `skills/local_data_agent/profiles/local_v1.json`

### Planner-First Mixed Execution Benchmark

The planner-first architecture benchmark runs:

1. `NL -> typed LogicalPlan`
2. deterministic routing (`pushdown|hybrid|python|cleaning_first`)
3. routed execution
4. mandatory output-schema validation/repair

Run:

```bash
python eval/open_data_mixed_execution_eval.py
python eval/open_data_mixed_execution_eval.py --cases eval/golden/open_data/mixed_execution_cases.json --route-oracle eval/golden/open_data/mixed_execution_route_oracle.json
```

The report is written to `eval/reports/open_data_mixed_execution_*.json` and
includes payload-aware metrics:

- `plan_valid`
- `route_correct`
- `execution_correct`
- `schema_correct`
- `safety_pass`
- `payload_pass`
- `overall_pass`
- `rows_scanned`
- `rows_materialized`
- `bytes_fetched`
- `peak_memory_mb`
- `fallback_used` / `fallback_reason`
- `backend_counts` (shows SQL vs non-SQL pushdown usage)

Architecture files:

- `docs/mixed_execution_architecture.md`
- `src/mixed_execution/logical_plan.schema.json`
- `src/mixed_execution/*`
- `eval/golden/open_data/mixed_execution_cases.json`
- `eval/golden/open_data/mixed_execution_route_oracle.json`

### Central-LLM Orchestration Benchmark

This benchmark evaluates coordinator behavior as orchestration only:

- central plan output is typed JSON (no raw SQL, no raw Python)
- worker calls use opaque data handles only
- each hop is logged with worker, input/output handles, bytes, validation, fallback
- hop validation is split into `validation_scope` (contract/validator/policy) and optional `schema_validation_result`

Run:

```bash
python eval/open_data_orchestration_eval.py
python eval/open_data_orchestration_eval.py --cases eval/golden/open_data/orchestration_cases.json
```

The suite currently includes 39 cases across:

- `pushdown`
- `hybrid`
- `python_first`
- `cleaning_first`
- `follow_up`
- `adversarial`

Output report: `eval/reports/open_data_orchestration_*.json`

Coordinator metrics:

- `task_classification_correct`
- `worker_selection_correct`
- `sequence_correct`
- `handle_valid`
- `payload_pass`
- `safety_pass`
- `final_output_correct`

Typed orchestration schemas:

- `src/orchestration/schemas/source_manifest.schema.json`
- `src/orchestration/schemas/capability_manifest.schema.json`
- `src/orchestration/schemas/data_handle.schema.json`
- `src/orchestration/schemas/orchestration_plan.schema.json`
- `src/orchestration/schemas/worker_result.schema.json`
- `src/orchestration/schemas/typed_error_result.schema.json`

### Phase 0.5 Vertical Slice (PDF -> Query -> Network)

Place real PDFs in:

- `data/pdf_inputs/`

Run ingest on one PDF:

```bash
python phase05_slice.py ingest --pdf data/pdf_inputs/easy.pdf --run-id phase05-easy
```

Run a query against the produced `clean.csv`:

```bash
python phase05_slice.py query --run-id phase05-easy --ask "total revenue by region"
```

Start a local network endpoint:

```bash
python phase05_slice.py serve --host 127.0.0.1 --port 8787
```

Then call:

```bash
curl -X POST http://127.0.0.1:8787/query -H "Content-Type: application/json" -d "{\"run_id\":\"phase05-easy\",\"ask\":\"total revenue by region\"}"
```

Artifacts are written to:

- `artifacts/phase05/<run-id>/`

## Artifacts

Successful queries are appended as JSON lines to
`artifacts/query_history.jsonl`:

```json
{"ts": "2026-04-10T12:34:56+00:00", "sql": "SELECT ...", "rows": 42}
```

## Repository Layout

```
Sql-editor/
â”œâ”€â”€ main.py                   # Tkinter entry point
â”œâ”€â”€ config.yaml               # App + LLM (Phase 3) settings
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ VENDOR.md                 # Attribution for vendored legacy code
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ ingestion.py          # CSV â†’ pandas â†’ in-memory SQLite (read-only)
â”‚   â”œâ”€â”€ query_model.py        # Pure data model; emits SELECT-only SQL
â”‚   â”œâ”€â”€ executor.py           # Read-only executor + keyword blocklist
â”‚   â”œâ”€â”€ history.py            # query_history.jsonl logger
â”‚   â”œâ”€â”€ config.py             # YAML config loader
â”‚   â”œâ”€â”€ llm/
â”‚   â”‚   â””â”€â”€ natural_language.py  # Ollama client + JSON â†’ QueryModel parser
â”‚   â””â”€â”€ ui/
â”‚       â”œâ”€â”€ query_builder.py  # Main window
â”‚       â”œâ”€â”€ filter_rows.py    # Dynamic filter composer (WHERE and HAVING)
â”‚       â”œâ”€â”€ aggregation.py    # GROUP BY + aggregation rows
â”‚       â”œâ”€â”€ order_by.py       # Multi-column ORDER BY composer
â”‚       â”œâ”€â”€ sql_preview.py    # Read-only Text widget w/ keyword highlight
â”‚       â””â”€â”€ results_table.py  # ttk.Treeview results widget
â”œâ”€â”€ tests/                    # pytest suite
â”œâ”€â”€ vendor/                   # Vendored legacy code (MIT, attributed)
â””â”€â”€ artifacts/                # Runtime outputs (query history)
```

## Legacy Reuse

This project reuses patterns and small code snippets from three earlier
Alleyfoo projects:

- [Alleyfoo/Data-tool-demo](https://github.com/Alleyfoo/Data-tool-demo) â€”
  YAML config loader pattern (vendored in `vendor/data-tool-demo/`).
- [Alleyfoo/Support-triage-llm](https://github.com/Alleyfoo/Support-triage-llm) â€”
  Ollama env-var configuration pattern (referenced for Phase 3).
- [Alleyfoo/slm-cleanroom-demo](https://github.com/Alleyfoo/slm-cleanroom-demo) â€”
  JSON-schema validation for LLM output (referenced for Phase 3).

See `VENDOR.md` for exact file-level attribution with commit SHAs.

## Roadmap

- **Phase 1** âœ“ â€” Ingestion, field selection, filter composer, SQL preview,
  executor, results table, CSV export.
- **Phase 2** âœ“ â€” Aggregation (`SUM` / `COUNT` / `AVG` / `MIN` / `MAX` /
  `COUNT DISTINCT`), `GROUP BY`, `HAVING`, multi-column `ORDER BY`.
- **Phase 3** âœ“ â€” LLM natural-language input via Ollama (`gemma3`),
  emitting a `QueryModel` (not raw SQL) that passes through the same
  validator and executor blocklist before execution.
- **Phase 4** â€” Multi-CSV loading and a visual `JOIN` composer.

