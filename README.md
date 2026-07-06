# Query Studio

A local web app for querying CSV data with natural language. You can use the demo,
quick queries, visual composer, and offline heuristic examples without installing
Ollama or adding an API key. If you connect local Ollama, the app can also
generate structured query plans and narrative analysis without sending data to
an external service; cloud providers are optional for shareable datasets.

![status: alpha](https://img.shields.io/badge/status-alpha-orange)

> **Try it live (no install):** <https://sql-editor-lehuxzh2q7mnmnm49an5hm.streamlit.app/>
> The hosted demo ships with the bundled sales dataset and the full UI. The
> model-free path (heuristic parsing, quick queries, composer) works instantly.
> To use the **LLM features on the hosted demo**, bring a free cloud key — see
> [Try it with a real LLM](#try-it-with-a-real-llm--no-install) below. No Ollama
> install or server required.

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

### Try it with a real LLM — no install

The hosted demo and any local run can use a real LLM **without installing
Ollama**, via a free cloud API key. Bring-your-own-key keeps your data in your
hands — the key is never stored or sent anywhere except the provider you pick.

1. Open the demo: <https://sql-editor-lehuxzh2q7mnmnm49an5hm.streamlit.app/>
2. Click **Load demo dataset**.
3. Open the **⚙ LLM model** popover in the top bar and pick a cloud provider:
   - **Groq (cloud)** — free key at <https://console.groq.com>.
     Recommended model: `llama-3.1-8b-instant` (fast) or
     `llama-3.3-70b-versatile` (higher quality).
   - **Gemini (cloud)** — free key at <https://aistudio.google.com/apikey>.
     Recommended model: `gemini-2.5-flash`.
4. Paste your key into the API key field and choose the model.
5. On the **LLM SQL Assistant** tab, ask a question (e.g.
   `monthly revenue trend 2024`) and compare the heuristic plan against the
   LLM plan, or run **Ask + Analyze** for a narrative insight report.

The key lives only in your browser session for that tab. Nothing is written to
disk, the repo, or the deploy. **Data note:** with a cloud provider the prompt
and a trimmed schema snapshot are sent to that provider — use it only with data
you're allowed to share. For fully offline use, run locally with Ollama (below).

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
  provider: ollama            # ollama | ollama_remote | groq | gemini
  model: qwen2.5-coder:7b      # must match what you pulled (Ollama) or the provider's catalog
  host: http://localhost:11434 # Ollama host (ollama / ollama_remote only)
  timeout: 120
  # api_key: gsk_...           # Groq / Gemini only — prefer env var or ⚙ popover
```

For a cloud provider, set `provider: groq` (or `gemini`) and supply the key via
the **⚙ LLM model** popover, an environment variable, or Streamlit secrets —
not in a committed `config.yaml`.

Or set environment variables (these override `config.yaml`):

| Variable         | Default                  | Used by                |
| ---------------- | ------------------------ | ---------------------- |
| `OLLAMA_HOST`    | `http://localhost:11434` | ollama, ollama_remote  |
| `OLLAMA_MODEL`   | `gemma3`                | ollama, ollama_remote  |
| `OLLAMA_TIMEOUT` | `60` (seconds)          | ollama, ollama_remote  |
| `GROQ_API_KEY`   | _none_                  | groq                   |
| `LLM_API_KEY`    | _none_                  | groq, gemini (fallback) |

On **Streamlit Community Cloud**, put secrets in `.streamlit/secrets.toml`
instead of env vars — the app reads `st.secrets["llm"]` (e.g.
`provider = "groq"`, `api_key = "gsk_..."`, `model = "llama-3.1-8b-instant"`).

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

**Model selector** — click **⚙ LLM model** in the top bar to switch providers and
pick a model. Four providers are supported:

- **Local Ollama** — a model running on this machine (default; data stays local).
- **Remote Ollama** — an Ollama server reachable over the network (`OLLAMA_HOST`).
- **Groq (cloud)** — bring a free key from <https://console.groq.com>.
- **Gemini (cloud)** — bring a free key from <https://aistudio.google.com/apikey>.

For the cloud providers, the popover shows an API key field (paste a free key)
and a model dropdown. The key is held only in the browser session for that tab —
it is not written to disk, the repo, or the deploy.

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
