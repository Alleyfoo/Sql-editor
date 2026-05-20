# Query Studio

A local web app for querying CSV data with natural language. Ask questions in plain
English, inspect the generated SQL before it runs, and get structured insight cards
plus LLM-written narrative — all running on your machine with no data leaving it.

![status: alpha](https://img.shields.io/badge/status-alpha-orange)

---

## What it does

- **Open a CSV** → it's loaded into an in-memory SQLite database (read-only, nothing written to disk)
- **Ask a question** → a local Qwen model generates a structured query plan (JSON, not SQL)
- **Inspect the SQL** → the plan is converted to SQL and shown before anything runs
- **Click Run** → results appear with insight cards, charts, and a summary tab
- **Ask + Analyze** → after running, the LLM writes a 2–3 sentence interpretation and suggests follow-up questions

The LLM never touches the database. It only produces a query plan that gets
validated against your schema, converted to SQL by trusted Python code, and
executed after you've seen it.

---

## Quick start

### 1. Install Ollama

Download from **[ollama.com](https://ollama.com/download)** — available for
Windows, Mac, and Linux. Run the installer, then verify:

```bash
ollama --version
```

### 2. Pull a model

For SQL generation, Qwen models work well:

```bash
# Good balance of speed and quality (~4.5 GB)
ollama pull qwen2.5-coder:7b

# Better reasoning, needs more RAM (~8 GB)
ollama pull qwen2.5:14b
```

Any model that follows JSON-format instructions will work. Start with
`qwen2.5-coder:7b` if you're unsure.

### 3. Install Python dependencies

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

Dependencies: `streamlit`, `pandas`, `altair`, `pyyaml`.
No GPU required — Ollama runs CPU inference by default.

### 4. Configure the model

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

### 5. Run

```bash
streamlit run streamlit_app.py
```

Opens at **http://localhost:8501**. Click **Load demo dataset** to try it
immediately without uploading a file.

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
locally pulled Ollama models without editing `config.yaml`.

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

**No data leaves your machine.** Ollama runs locally, SQLite is in-memory,
and the app makes no external network calls.

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
│   │   └── natural_language.py   # Ollama client + JSON → QueryModel
│   └── streamlit_app/
│       ├── app.py                # Page layout
│       ├── styles.css            # Design tokens + component styles
│       ├── insight_engine.py     # Deterministic insight cards
│       ├── insight_enrichment.py # Phase 4c LLM narrative layer
│       ├── quick_queries.py      # Schema-aware offline query templates
│       ├── demo_dataset.py       # Demo CSV loader
│       └── components/
│           ├── ask.py            # NL input + quick queries
│           ├── assistant.py      # Conversation panel
│           ├── composer.py       # Visual query builder
│           ├── header.py         # Top bar + file upload + model selector
│           ├── results.py        # Results table + chart + summary tabs
│           ├── sidebar.py        # Schema explorer + recent runs
│           └── sql_preview.py    # SQL display + Run button
└── tests/
```
