# Visual Query Builder

A desktop tool for loading CSV data, composing queries visually, and
understanding the resulting SQL. Built on Python + Tkinter + SQLite
(in-memory) + pandas.

Downstream of the cleaning pipeline, not part of it: the tool never writes
to your CSV and the SQLite database is in-memory and locked read-only.

## Status

**Phase 2** — Phase 1 plus aggregation controls (`SUM` / `COUNT` / `AVG` /
`MIN` / `MAX` / `COUNT DISTINCT`), `GROUP BY`, `HAVING` (reuses the filter
composer, dropdown scoped to group columns and aggregation aliases),
multi-column `ORDER BY` with `ASC` / `DESC`. `to_sql()` enforces the
standard SQL rule that every non-aggregated column in the SELECT list
must appear in GROUP BY when aggregations are present. LLM natural-language
input (Phase 3) and multi-table JOINs (Phase 4) are not yet implemented.

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

Then **File → Open CSV…** to load a file. As you click columns in the left
panel and add filters in the center panel, the generated SQL updates live in
the right panel. Press **Run** to execute and see results in the table at
the bottom, then **Export CSV…** to save them elsewhere.

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
4. **No raw-SQL input field.** Phase 1 has no place for the user to type
   SQL — every query is built from widgets.
5. **No database file written to disk.** The DB lives only in memory.

## Testing

```bash
pytest tests/ -v
```

The suite covers query-model SQL generation (every operator, AND/OR logic,
LIMIT, SQL-injection attempts in filter values), the executor blocklist, and
ingestion including a live check that the returned connection rejects writes.

## Artifacts

Successful queries are appended as JSON lines to
`artifacts/query_history.jsonl`:

```json
{"ts": "2026-04-10T12:34:56+00:00", "sql": "SELECT ...", "rows": 42}
```

## Repository Layout

```
Sql-editor/
├── main.py                   # Tkinter entry point
├── config.yaml               # App + LLM (Phase 3) settings
├── requirements.txt
├── VENDOR.md                 # Attribution for vendored legacy code
├── src/
│   ├── ingestion.py          # CSV → pandas → in-memory SQLite (read-only)
│   ├── query_model.py        # Pure data model; emits SELECT-only SQL
│   ├── executor.py           # Read-only executor + keyword blocklist
│   ├── history.py            # query_history.jsonl logger
│   ├── config.py             # YAML config loader
│   └── ui/
│       ├── query_builder.py  # Main window
│       ├── filter_rows.py    # Dynamic filter composer (WHERE and HAVING)
│       ├── aggregation.py    # GROUP BY + aggregation rows
│       ├── order_by.py       # Multi-column ORDER BY composer
│       ├── sql_preview.py    # Read-only Text widget w/ keyword highlight
│       └── results_table.py  # ttk.Treeview results widget
├── tests/                    # pytest suite
├── vendor/                   # Vendored legacy code (MIT, attributed)
└── artifacts/                # Runtime outputs (query history)
```

## Legacy Reuse

This project reuses patterns and small code snippets from three earlier
Alleyfoo projects:

- [Alleyfoo/Data-tool-demo](https://github.com/Alleyfoo/Data-tool-demo) —
  YAML config loader pattern (vendored in `vendor/data-tool-demo/`).
- [Alleyfoo/Support-triage-llm](https://github.com/Alleyfoo/Support-triage-llm) —
  Ollama env-var configuration pattern (referenced for Phase 3).
- [Alleyfoo/slm-cleanroom-demo](https://github.com/Alleyfoo/slm-cleanroom-demo) —
  JSON-schema validation for LLM output (referenced for Phase 3).

See `VENDOR.md` for exact file-level attribution with commit SHAs.

## Roadmap

- **Phase 1** ✓ — Ingestion, field selection, filter composer, SQL preview,
  executor, results table, CSV export.
- **Phase 2** ✓ — Aggregation (`SUM` / `COUNT` / `AVG` / `MIN` / `MAX` /
  `COUNT DISTINCT`), `GROUP BY`, `HAVING`, multi-column `ORDER BY`.
- **Phase 3** — LLM natural-language input via Ollama (`gemma3`), emitting
  a `QueryModel` (not raw SQL) that passes through the same validator
  before execution.
- **Phase 4** — Multi-CSV loading and a visual `JOIN` composer.
