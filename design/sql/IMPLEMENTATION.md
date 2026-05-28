# Streamlit port of the Visual Query Builder — implementation brief

You (Claude Code) are porting `src/ui/query_builder.py` (Tkinter) to a Streamlit web app. The new design is in `Visual Query Builder.html` — read it before writing code. This brief tells you **what to build, how to lay it out, what to reuse, and what the safety boundary is**.

The current architecture is intentionally split so that the UI is a thin presentation layer over pure-Python query primitives. **Reuse all non-UI modules verbatim** — they were designed for this. Only the Tk window gets replaced.

---

## 1. What to reuse without modification

These modules already enforce the safety boundary and have full test coverage. Import them; do not rewrite.

| Module | Purpose | Use in Streamlit |
|---|---|---|
| `src/ingestion.py` | CSV → pandas → in-memory read-only SQLite, returns `(conn, schema)` | Call on file upload |
| `src/query_model.py` | `QueryModel`, `Filter`, `Aggregation`, `OrderBy`, `to_sql()` | Build from UI widgets exactly as Tk does |
| `src/executor.py` | `execute(conn, sql)` → `pd.DataFrame`, with keyword blocklist | Call on Run; surface `ExecutionError` |
| `src/history.py` | Appends to `artifacts/query_history.jsonl` | Call after every successful run |
| `src/config.py` | `load_config()` → YAML | Read once at startup |
| `src/llm/natural_language.py` | `nl_to_query_model(...)` returning a validated `QueryModel` | Call from the "Ask" button |
| `src/llm/result_analysis.py` | `analyze_result_with_llm(...)` → summary/insights/follow-ups | Call from "Ask + Analyze" |

**Do not** generate SQL strings anywhere in the Streamlit layer. Every SQL string must come from `QueryModel.to_sql()`. Every LLM response must come back as a `QueryModel`, never as raw SQL. This is non-negotiable — the README spells out the multi-layer rationale.

---

## 2. File structure to add

```
src/ui/
  ...existing Tkinter modules stay (don't delete; tests reference them)
src/streamlit_app/
  __init__.py
  app.py                # main entry — `streamlit run src/streamlit_app/app.py`
  state.py              # session_state helpers + dataclass-ish accessors
  components/
    __init__.py
    header.py           # top bar: brand, breadcrumb, upload, history
    sidebar.py          # dataset card + schema profile + recent runs
    ask.py              # NL ask bar + Ask / Ask+Analyze buttons + suggestion chips
    assistant.py        # transcript: user msg, bot reply, insight cards, follow-ups
    composer.py         # 5 stacked composer sections (SELECT/WHERE/GROUP/HAVING/ORDER)
    sql_preview.py      # syntax-highlighted SQL + Run button + safety badge
    results.py          # tabs (Table/Chart/Summary/JSON) + data table + footer
  styles.py             # `inject_css()` — returns the theme block from styles.css
  styles.css            # the cosmetic CSS from the mockup
  profile.py            # column profile: nulls, unique count, min/max, mini-histogram
streamlit_app.py        # thin shim at repo root → `from src.streamlit_app.app import run; run()`
.streamlit/
  config.toml           # theme overrides (see §6)
```

`main.py` (the Tkinter entry) stays put. The new entry is `streamlit run streamlit_app.py`. Update `README.md` Run section to mention both.

---

## 3. Layout (mirrors `Visual Query Builder.html`)

Use `st.set_page_config(page_title="Query Studio", layout="wide", initial_sidebar_state="expanded")`.

```
┌───────────────────────────────────────────────────────────────────┐
│ TOP BAR (custom HTML via st.markdown — Streamlit has no native bar)│
├───────────┬───────────────────────────────────────────────────────┤
│           │  ASK BAR (st.container)                               │
│ SIDEBAR   │   ├ st.text_input (large)                             │
│ (st.side  │   └ two st.buttons: Ask / Ask + Analyze (primary)     │
│  bar)     │                                                       │
│           │  ASSISTANT TRANSCRIPT (st.container)                  │
│  Dataset  │   └ for each turn: st.chat_message + insight cards    │
│  card     │                                                       │
│           │  WORK ROW: st.columns([1.15, 1], gap="medium")        │
│  Schema   │   ├ COMPOSER (5 collapsible sections)                 │
│  profile  │   └ SQL PREVIEW (st.code w/ language="sql" + Run)     │
│           │                                                       │
│  Recent   │  RESULTS (st.tabs)                                    │
│  runs     │   ├ Table  · st.dataframe with column_config          │
│           │   ├ Chart  · st.altair_chart on chosen x/y            │
│           │   ├ Summary · profile of result df                    │
│           │   └ JSON   · st.json of df.to_dict("records")         │
└───────────┴───────────────────────────────────────────────────────┘
```

Streamlit has no native top bar. Render it as a single `st.markdown(..., unsafe_allow_html=True)` block fixed to the top, and add CSS padding-top to the main body to clear it. Same for the bottom status bar.

---

## 4. Session state

Define a single namespaced dict so state is easy to inspect:

```python
# src/streamlit_app/state.py
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import streamlit as st
import pandas as pd
import sqlite3
from src.query_model import QueryModel
from src.ingestion import TABLE_NAME

def init() -> None:
    ss = st.session_state
    ss.setdefault("conn", None)             # sqlite3.Connection (read-only)
    ss.setdefault("schema", {})             # Dict[str, str]
    ss.setdefault("dataset_name", None)     # str
    ss.setdefault("dataset_meta", {})       # rows, size_bytes, loaded_at
    ss.setdefault("model", QueryModel(table=TABLE_NAME))
    ss.setdefault("results_df", None)       # pd.DataFrame | None
    ss.setdefault("last_sql", "")           # str
    ss.setdefault("last_exec_ms", None)     # float
    ss.setdefault("nl_history", [])         # List[Tuple[str,str]]
    ss.setdefault("transcript", [])         # List[dict] — user/bot turns
    ss.setdefault("nl_status", "")          # "Thinking…" etc
    ss.setdefault("composer_open", {"select": True, "where": True, "group": True, "having": False, "order": True})
```

Mutations must go through small helper functions so the trigger-render flow stays predictable. Streamlit reruns top-to-bottom on every interaction — embrace that, don't fight it.

---

## 5. Component contracts

### 5.1 Header (`components/header.py`)

Render a sticky HTML bar with: brand, breadcrumb (workspace › dataset.csv › "Untitled query"), upload button (triggers an `st.file_uploader` in a popover), History button (opens an expander or modal listing recent runs from `query_history.jsonl`), Share placeholder, avatar.

**Implementation note.** Use `st.popover("Open CSV")` to host the file picker — it matches the mockup's inline behavior. On submit, call `load_csv()` from `ingestion.py`, store conn + schema in session state, reset the `QueryModel`, and `st.rerun()`.

### 5.2 Sidebar (`components/sidebar.py`)

Use the real `st.sidebar`. Sections, top to bottom:

1. **Dataset card.** Show filename, rows, columns, size, loaded-at. Two badges: "Connected" (green dot) and "read-only" (amber). Inactive state when no CSV is loaded — gray, "Open a CSV to begin".
2. **Schema profile.** For each column, render a row with: checkbox (toggles `model.selected_columns`), name, type chip, then a stats sub-row. Stats depend on type:
   - **text** → unique count, null count, completeness bar (`100% - null%`)
   - **numeric** → min/max, mean (small), micro-histogram (12 bins, mini sparkline)
   - **date** → min/max formatted as `MMM dd`, completeness bar
   Profile data comes from a `compute_profile(df, col, dtype)` helper in `profile.py` that runs **once per CSV load**, cached via `@st.cache_data` keyed on the file hash.
3. **Recent runs.** Read `artifacts/query_history.jsonl` (last 8 entries), reverse, show relative time + truncated SQL + row count. Click loads the SQL back into the model — easiest way is to keep the parsed `QueryModel` in the history file too (add to `history.log_query` payload, backward-compatible).

Schema rows are clickable: clicking a row toggles selection. The checkbox is decorative — Streamlit can't render a real checkbox inside a custom-HTML row cleanly, so use one HTML row per column and intercept clicks via an invisible `st.button` per row (label = column name, `use_container_width=True`, styled to disappear) OR use `streamlit-extras`'s `clickable_container`. The first approach is dependency-free; the second is prettier.

### 5.3 Ask (`components/ask.py`)

```python
col_input, col_ask, col_analyze = st.columns([1, 0.13, 0.18])
text = col_input.text_input("ask", label_visibility="collapsed", placeholder="Ask in plain English…")
ask = col_ask.button("Ask", use_container_width=True)
analyze = col_analyze.button("Ask + Analyze", type="primary", use_container_width=True)
```

Below the row, show 3–4 suggestion chips. Hardcode three generic ones plus one schema-aware suggestion (e.g. `f"top {n} {first_text_col} by {first_numeric_col}"`).

Both buttons disabled (`disabled=not st.session_state.conn`) until a CSV is loaded.

On click: call `nl_to_query_model(...)` (synchronously — Streamlit has no real background threads that play well with reruns; show a spinner with `with st.spinner("Thinking…"):`). On success, set `st.session_state.model = result`, append to `transcript`, `st.rerun()`. On `RouteToPythonError`, append a transcript entry explaining the route and do **not** mutate the model. On any other `LLMError`, append as an error turn — never raise into the page.

For "Ask + Analyze" do the same plus immediately execute and run `analyze_result_with_llm`. The transcript entry then carries `summary`, `insights[]`, `next_questions[]`, `warnings[]` — render those as the three insight cards + follow-up chips shown in the mockup.

### 5.4 Assistant transcript (`components/assistant.py`)

Iterate `st.session_state.transcript`. Each entry is one of:

```python
{"role": "user", "text": str, "ts": datetime, "ds": str}
{"role": "assistant", "reply": str, "sql": str, "analysis": ResultAnalysis|None, "error": str|None}
```

Render user turns with `st.chat_message("user")`. Render assistant turns with `st.chat_message("assistant")`, then below the reply text:

- **Insight cards** (only when analysis present): use `st.columns(3)` with custom-styled containers — match the mockup's `.insight` style (uppercase label, italic serif value, monospace delta line). Cap at 3.
- **Follow-up chips**: render each as a small `st.button` with a `↳` prefix. Clicking pre-fills the ask input via `st.session_state.nl_prefill = chip_text; st.rerun()`.

### 5.5 Composer (`components/composer.py`)

Five sections in order: **SELECT**, **WHERE**, **GROUP BY · Aggregate**, **HAVING**, **ORDER BY · LIMIT**. Each is an `st.expander(label, expanded=...)` whose label contains a brief summary of the current state — match the mockup's `.summary` line. Use the open-state from `session_state.composer_open[key]`.

The composer's job is to mutate `st.session_state.model` and call `st.rerun()` when something changes. Concretely:

| Section | Widgets | Mutates |
|---|---|---|
| SELECT | `st.multiselect` of `schema.keys()`, OR a custom pill row (see below) | `model.selected_columns` |
| WHERE | dynamic list of filter rows | `model.filters` |
| GROUP BY · Agg | multiselect for group_by + dynamic agg rows | `model.group_by`, `model.aggregations` |
| HAVING | dynamic filter rows (disabled if `group_by` empty — show inline hint) | `model.having` |
| ORDER BY · LIMIT | dynamic sort rows + number_input for limit | `model.order_by`, `model.limit` |

**Filter row implementation.** One row = `st.columns([0.13, 0.28, 0.22, 0.32, 0.05])`. Cells:

1. `AND`/`OR` segmented control (`st.radio` horizontal, label collapsed) — hidden on the first row.
2. Column selectbox sourced from current schema.
3. Operator selectbox driven by the selected column's type (`OPERATORS_BY_TYPE[col_type]` from `query_model.py`).
4. Value input — branches:
   - `IS NULL` / `IS NOT NULL` → render nothing (use `st.empty()`).
   - `BETWEEN` → two inputs in a sub-`st.columns([1,0.05,1])` with an "and" separator.
   - other → single `st.text_input` (or `st.number_input` if column type is numeric).
5. Remove button (`st.button("×", key=...)`) — pops the row and reruns.

Persist filter rows as a list in `session_state.where_rows`, mirror into `model.filters` on each render via the `to_filters()` translator.

**Aggregation row.** `st.columns([0.2, 0.4, 0.05, 0.3, 0.05])`: function selectbox (`SUM/COUNT/AVG/MIN/MAX/COUNT DISTINCT`), column selectbox, an arrow label `→`, alias text input, remove button. Defaults match the patterns shipped in `aggregation.py`.

**LIMIT.** `st.number_input("LIMIT", min_value=0, max_value=1_000_000, step=100, value=1000)`. Note "0 = no limit" inline.

After any composer interaction, recompute SQL by calling `model.to_sql()` inside a `try/except ValueError` and cache the result in `session_state.last_sql`. Display invalid-state errors above the SQL panel as a small red banner — don't raise.

### 5.6 SQL preview (`components/sql_preview.py`)

```python
st.markdown('<div class="sql-panel-wrap">', unsafe_allow_html=True)
st.code(st.session_state.last_sql or "-- compose a query above --", language="sql", line_numbers=True)
```

Underneath, a "runbar" row with:
- A primary **Run query** button (`type="primary"`). Disabled while `model.to_sql()` raises.
- An "Explain" button (placeholder — open an expander explaining each clause; defer real `EXPLAIN QUERY PLAN` to a follow-up).
- A safety badge: `🔒 SELECT-only · read-only conn`.
- Keyboard hint `⌘↵` (informational only; Streamlit doesn't capture global hotkeys cleanly).

Run handler:

```python
sql = st.session_state.model.to_sql()
with st.spinner("Running…"):
    t0 = time.perf_counter()
    df = execute(st.session_state.conn, sql)
    elapsed_ms = (time.perf_counter() - t0) * 1000
st.session_state.results_df = df
st.session_state.last_exec_ms = elapsed_ms
history.log_query(sql, len(df))
st.rerun()
```

Wrap in `try/except ExecutionError as exc: st.error(str(exc))`. Do not call `st.stop()`.

### 5.7 Results (`components/results.py`)

`st.tabs(["Table", "Chart", "Summary", "JSON"])`. Above the tabs, render a stats row (rows, cols, exec time, scanned bytes if available) plus an Export CSV button. **Export** uses `st.download_button` with the current df's `to_csv(index=False)`.

**Table tab.** `st.dataframe(df, column_config=..., use_container_width=True, hide_index=True)`. Build `column_config` from result dtypes:

- numeric → `st.column_config.NumberColumn(format="%.2f")` and consider `ProgressColumn` when the column looks like a magnitude (e.g. ends with `revenue`, `total`, `count`, `amount`). Detection: if `df[col].min() >= 0` and `df[col].max() / df[col].min().clip(lower=1) > 5`, use ProgressColumn with `min_value=0, max_value=df[col].max()`. Match the mockup's tiny accent bar this way.
- date/datetime → `DatetimeColumn` with `format="YYYY-MM-DD"`.
- text → default; add a `help=` tooltip with unique-count if low cardinality.

Headers should display a one-line column profile underneath. Streamlit can't put rich content inside `st.dataframe` headers natively; render the profile **above** the dataframe as a separate column-strip (one `st.columns` row with type chip + null %), aligned by column ratios that mirror the auto-sized columns. Acceptable approximation; mark as "v2 polish" if time-boxed.

**Chart tab.** Pick a sensible default: if there's exactly one text column and one numeric column, render `alt.Chart(df).mark_bar()`. Two numerics → scatter. One date + one numeric → line. Otherwise show "Pick X and Y" with two selectboxes. Use Altair (`pip install altair` already available via Streamlit). Theme it to match: warm rust bars (`#C2410C`), `#1A1714` text.

**Summary tab.** For each result column, render: type, count, nulls, unique (text), or min/mean/max/std (numeric). Use `df.describe(include="all")` and reformat into the same insight-card visual language used in the assistant transcript.

**JSON tab.** `st.json(df.head(200).to_dict(orient="records"))` with a note "showing first 200 rows".

---

## 6. Theme

Add `.streamlit/config.toml`:

```toml
[theme]
base = "light"
primaryColor = "#C2410C"
backgroundColor = "#F4F1EA"
secondaryBackgroundColor = "#FBFAF6"
textColor = "#1A1714"
font = "sans serif"
```

Then inject the cosmetic CSS from `styles.css` once per page via `st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)`. Port the relevant rules from `Visual Query Builder.html`'s `<style>` block, keyed against Streamlit's actual class names (`[data-testid="stSidebar"]`, `[data-testid="stExpander"]`, `[data-testid="stDataFrame"]`, etc.). Concretely you want to override:

- Sidebar background to `--paper` (`#FBFAF6`), border-right `1px solid #E4DFD2`.
- `h1, h2, h3` font to IBM Plex Sans, tighter letter-spacing.
- Expander chrome to match the composer-section look (numbered badge + summary on the right).
- `st.code` block: dark background `#1A1714`, color tokens for keywords/strings/numbers. Streamlit uses Prism — override Prism token colors via CSS.
- Button rounding to `6px`, primary button color to `#C2410C`.

Load IBM Plex Sans + IBM Plex Mono via a `<link>` injected in the same CSS block.

**Do not** ship the entire mockup CSS verbatim — most of it targets bespoke HTML that won't exist in the Streamlit DOM. Lift the **tokens** (colors, radii, type ramp) and the **patterns** (insight card, type chip, sample-value chip) and apply them to Streamlit's components.

---

## 7. NL flow — behavioral notes

These mirror the Tkinter app and must be preserved:

1. **NL never auto-runs.** "Ask" populates the composer; the user still clicks Run. The mockup makes this explicit by leaving the Run button on the SQL panel, not the ask bar.
2. **"Ask + Analyze" does run** after planning. After execution it calls `analyze_result_with_llm`. Both the SQL and the analysis are visible in the transcript.
3. **History context.** Maintain `nl_history` as a bounded list of `(question, reply)` tuples sized by `llm_config.history_depth`. Pass it into `nl_to_query_model`.
4. **`RouteToPythonError`** is not an error to the user. Render the message in the transcript as a normal assistant turn with a small "routed to Python" pill. Do not toast.
5. **Errors from the LLM or executor** become a single transcript entry with `error=...` styled red. Use `st.toast(str(exc), icon="⚠️")` for ephemeral notification.

---

## 8. Safety boundary — what must remain true

These invariants are tested in `tests/test_executor.py`, `tests/test_query_model.py`, `tests/test_nl_parser.py`. Do not introduce any code path that breaks them.

- No raw SQL input field. There is no `st.text_area` anywhere that flows into `execute(...)`.
- The connection passed to `execute(...)` is always the one returned by `ingestion.load_csv()`, which has `PRAGMA query_only = ON` and the deny-all authorizer.
- Every SQL string passed to `execute` is the output of `QueryModel.to_sql()`. The keyword blocklist still runs inside `execute`.
- The DB is in-memory. No file on disk is written by the executor.
- LLM output is parsed into a `QueryModel` and validated against the active schema — never executed as text.

Add a tiny visible badge ("🔒 SELECT-only · read-only conn") near the Run button so this guarantee is visible to the user — not hidden in the code.

---

## 9. Pandas / display niceties

- After `execute(...)`, cast obvious date columns: any `object` column whose name ends in `_date` or `_at` and parses as ISO-8601 → `pd.to_datetime`. Keep a copy of the original for export. Do this in a `decorate_for_display(df)` helper that's only used in the Table tab.
- For text columns with cardinality ≤ 12, surface a value-count summary in the column profile area.
- Format large numbers with thousands separators in the table (`%.2f` with locale or `f"{x:,.2f}"`).
- Round float monetary-looking columns to 2 decimals; keep raw values in the underlying df.

---

## 10. Tests

Add at least:

- `tests/test_streamlit_state.py` — exercises `state.init()`, the composer mutation helpers, and the `model → SQL` reconciliation path. Use `streamlit.testing.v1.AppTest` (Streamlit 1.30+). Verify: a) opening a CSV populates schema, b) ticking columns updates `model.selected_columns`, c) adding a filter updates `model.filters`, d) the generated SQL matches what the existing `to_sql()` would produce — i.e. compare against `QueryModel(...).to_sql()` directly.
- `tests/test_streamlit_safety.py` — assert that the rendered page contains the safety badge text, that there is no `st.text_area` whose value is passed to `execute`, and that running with a malformed model surfaces an `st.error` rather than crashing.

Existing Tkinter tests stay green — none of them import the Streamlit module.

---

## 11. Requirements

Add to `requirements.txt`:

```
streamlit>=1.32
altair>=5
```

Everything else (pandas, sqlite3, urllib) is already a dependency.

---

## 12. Cutover

1. Build the Streamlit app under `src/streamlit_app/` per the structure above.
2. Add `streamlit_app.py` shim at the repo root.
3. Update `README.md` "Run" section: keep `python main.py` for the Tk app; add `streamlit run streamlit_app.py` as the new primary entry point.
4. Tag the Tk version as legacy in the README's Roadmap section. Do not delete it — `query_builder.py` and its sub-widgets stay for regression and offline use.
5. Run the full pytest suite. All existing tests must still pass.

---

## 13. Look & feel reference

`Visual Query Builder.html` is the source of truth for layout, density, and palette. When in doubt about a visual decision (corner radius, padding, color), open the file and lift the value from the relevant CSS rule. The mockup's design tokens live in the `:root` block at the top of the `<style>` element — those are the canonical color/radius values. Do not invent new ones.

The two highest-leverage style wins over default Streamlit:

1. **Type chips and column profiles in the sidebar.** This is the single biggest "data tool" upgrade vs. the Tk version — surface column health (nulls, cardinality, range) at a glance.
2. **Conversational transcript with insight cards + follow-ups.** The current Tk app drops LLM output into a `Text` widget. In Streamlit, render it as a real conversation with structured insights. The user spends most of their attention there.

Everything else can degrade gracefully toward stock Streamlit if a polish is too expensive — these two cannot.
