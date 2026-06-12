# Query Studio

A local web app for querying CSV data with natural language. You can use the demo,
quick queries, visual composer, and offline heuristic examples without installing
Ollama or adding an API key. If you connect local Ollama, the app can also
generate structured query plans and narrative analysis without sending data to
an external service; cloud providers are optional for shareable datasets.

![status: alpha](https://img.shields.io/badge/status-alpha-orange)

---

## What it does

- **Open a CSV** → it's loaded into an in-memory SQLite database (read-only, nothing written to disk)
- **Ask a question** → a local Qwen model generates a structured query plan (JSON, not SQL)
- **Inspect the SQL** → the plan is converted to SQL and shown before anything runs
- **Click Run** → results appear with insight cards, charts, and a summary tab
- **Ask + Analyze** → runs a full analysis pipeline: profiles the result, infers the
  analysis type (trend / segment / KPI / dashboard), asks the LLM for a grounded
  insight report, and picks the right chart type automatically

The LLM never touches the database. It only produces a query plan that gets
validated against your schema, converted to SQL by trusted Python code, and
executed after you've seen it.

---

## Quick start

### No Ollama or API key? Start here.

You can still try the core workflow:

1. Install the Python dependencies.
2. Run the Streamlit app.
3. Click **Load demo dataset**.
4. Try the quick-query buttons or ask one of these heuristic examples:
   - `sum revenue by region`
   - `top 10 products by revenue`
   - `count rows by status`
   - `show region and country`
5. Inspect the generated SQL, then click **Run query**.

This path is fully local and model-free. The app uses deterministic Python rules
for simple natural-language questions and pre-built quick queries for richer demos.

Use Ollama or a cloud API key only when you want the LLM features:

- more flexible natural-language parsing,
- **Ask + Analyze** narrative insight reports,
- heuristic-vs-LLM comparison on the **LLM SQL Assistant** tab.

### 1. Install Python dependencies

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

Dependencies: `streamlit`, `pandas`, `altair`, `pyyaml`, `openpyxl`, `pdfplumber`.

### 2. Run

```bash
streamlit run streamlit_app.py
```

Opens at **http://localhost:8501**. Click **Load demo dataset** to try it
immediately without uploading a file.

### 3. Optional: install Ollama for LLM features

Download from **[ollama.com](https://ollama.com/download)**. Run the installer,
then verify:

```bash
ollama --version
```

### 4. Optional: pull a model

For SQL generation, Qwen models work well:

```bash
# Good balance of speed and quality (~4.5 GB)
ollama pull qwen2.5-coder:7b

# Better reasoning, needs more RAM (~8 GB)
ollama pull qwen2.5:14b
```

Any model that follows JSON-format instructions will work. Start with
`qwen2.5-coder:7b` if you're unsure.

No GPU required — Ollama runs CPU inference by default.

### 5. Optional: configure the model

Edit `config.yaml`:

```yaml
llm:
  model: qwen2.5-coder:7b   # must match what you pulled
  host: http://localhost:11434
  timeout: 120
```

Or set environment variables (these override `config.yaml`):

| Variable         | Default                  |
| ---------------- | ------------------------ |
| `OLLAMA_HOST`    | `http://localhost:11434` |
| `OLLAMA_MODEL`   | `gemma3`                 |
| `OLLAMA_TIMEOUT` | `60` (seconds)           |

---

## Using it

**Load data** — click **Open CSV** in the top bar and upload any CSV file,
or click **Load demo dataset** for a built-in 3 000-row sales dataset spanning
2023–2025 with revenue, margin, customer IDs, and order status.

**Ask questions** — type in the ask bar and press **Ask** (fast, offline heuristic
first) or **Ask + Analyze** (LLM + narrative insight). Examples that work well:

- `monthly revenue trend 2024`
- `top 10 customers by margin`
- `compare EMEA vs AMER year-over-year`
- `which product categories have the highest return rate`
- `show revenue by status using a breakdown`

**Quick queries** — the row of buttons below the ask bar runs pre-built templates
offline with no LLM needed: monthly trend, top customers, margin by category,
orders by status, and more.

**Query Composer** — a visual builder for filters, GROUP BY, aggregations, ORDER BY,
and HAVING. The LLM populates it; you can tweak it before running.

**SQL Preview** — shows the exact SQL that will run. Nothing executes until you
click **▶ Run query**.

**Model selector** — click **⚙ LLM model** in the top bar to switch between
local Ollama and remote Ollama, and to choose which pulled model to use.

---

## Safety guarantees

This tool will never modify your data. Six layers enforce this:

1. **Read-only connection.** After loading, the SQLite connection switches to
   `PRAGMA query_only = ON` with a deny-all authorizer. Any `INSERT`, `UPDATE`,
   `DELETE`, `DROP`, or `ALTER` raises immediately.
2. **LLM produces JSON, not SQL.** The model is asked for a query plan (column
   names, filters, aggregations). It never sees or touches the database.
3. **Schema validation.** Every column name in the plan is checked against the
   actual dataset. Hallucinated columns are rejected before SQL is emitted.
4. **Operator allowlist.** Only known-safe operators, aggregation functions, and
   ORDER BY directions are accepted from the plan.
5. **SELECT-only code path.** `QueryModel.to_sql()` will only emit `SELECT`
   statements. There is no raw SQL input field anywhere.
6. **Executor blocklist.** A second validation pass before SQLite execution
   blocks `DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `CREATE`, `ATTACH`,
   `DETACH`, `PRAGMA`, `REPLACE`, `TRUNCATE`, `EXEC`, `EXECUTE`, `GRANT`,
   `REVOKE`.

**Local mode keeps data on your machine.** With Ollama, prompts are sent only to
your local Ollama server. If you choose Gemini, Groq, or another remote provider,
the app sends the prompt/schema context to that provider; use those modes only
with data you are allowed to share.

---

## Running tests

```bash
pytest tests/ -v
```

---

## Repository layout

```
sql-editor/
├── streamlit_app.py              # Entry point
├── config.yaml                   # App + LLM settings
├── requirements.txt
├── WORKFLOWS.md                  # Agentic pipeline architecture guide
├── data/
│   └── demo/
│       ├── sample_sales.csv      # Built-in demo dataset (3 000 rows)
│       └── generate_sample_sales.py
├── src/
│   ├── ingestion.py              # CSV → in-memory SQLite (read-only)
│   ├── query_model.py            # Data model; emits SELECT-only SQL
│   ├── executor.py               # Read-only executor + blocklist
│   ├── heuristic_nl.py           # Offline NL fast-path (no LLM)
│   ├── history.py                # Query history logger
│   ├── config.py                 # YAML config loader
│   ├── llm/
│   │   └── natural_language.py   # Multi-provider LLM client + JSON → QueryModel
│   ├── analysis_lane/            # Phase 4 analysis pipeline
│   │   ├── engine.py             # AnalysisCoordinator — profiling → insights → charts
│   │   ├── models.py             # AnalysisProfile, InsightReport, ChartSpec, etc.
│   │   ├── profiler.py           # Deterministic data profiler
│   │   └── validation.py         # Guardrail checks for claims and chart specs
│   ├── mixed_execution/          # Query routing engine (SQL/Python/hybrid)
│   ├── orchestration/            # Multi-worker coordination runtime
│   └── streamlit_app/
│       ├── app.py                # Page layout
│       ├── styles.css            # Design tokens + component styles
│       ├── insight_engine.py     # Deterministic insight cards (always-on)
│       ├── insight_enrichment.py # Lightweight LLM narrative (Ask path)
│       ├── quick_queries.py      # Schema-aware offline query templates
│       ├── demo_dataset.py       # Demo CSV loader
│       └── components/
│           ├── ask.py            # NL input, quick queries, analysis wiring
│           ├── assistant.py      # Conversation + insight card panel
│           ├── composer.py       # Visual query builder
│           ├── header.py         # Top bar, file upload, multi-provider selector
│           ├── results.py        # Results table + coordinator chart + summary tabs
│           ├── sidebar.py        # Schema explorer + recent runs
│           └── sql_preview.py    # SQL display + Run button
└── tests/
```

---

## Roadmap

| Phase | Status | What shipped |
|---|---|---|
| 1 — Core query engine | ✅ done | CSV ingestion, QueryModel, read-only SQLite, blocklist |
| 2 — Visual composer | ✅ done | Filter / GROUP BY / ORDER BY builder, SQL preview |
| 3 — NL + heuristic | ✅ done | Ollama integration, offline heuristic fast-path, quick queries |
| 4 — Deep analysis | ✅ done | AnalysisCoordinator pipeline, multi-provider LLM, smart chart selection |
| 5 — Multi-source | 🔜 next | Load multiple CSVs, visual JOIN composer, cross-dataset queries |
