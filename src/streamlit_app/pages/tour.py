"""Tour — the scroll-driven presentation that is the app's front door.

The workshop (Studio + LLM SQL Assistant tabs) is a powerful but sprawling
surface: a visitor lands in three sibling tabs, a model selector, a composer,
a SQL preview and an ask bar, and has to assemble the point themselves. The
genuinely interesting thing about Query Studio gets buried:

    An LLM can query your data in plain English *without ever touching the
    database*. It writes a plan (JSON, not SQL); trusted code validates every
    column and operator against your schema and emits SELECT-only SQL; you see
    it and approve it before anything runs.

This page tells that thesis as one story across five live beats, each a real
run of the existing pipeline — not a static deck:

    1. Your data, in plain English  — the dataset + an editable question
    2. Why this is safe             — the contrast: naive NL→SQL vs plan→SQL
    3. The plan, in the open        — live: question -> JSON/structured plan
    4. You approve, it runs        — live: plan -> SELECT SQL -> Run -> table
    5. The insight, automatically   — live: result -> headline + cards + chart
                                     (+ optional LLM narrative)

The workshop stays one click away via "Open the workshop".

Decoupled dataset
-----------------
The tour always runs against the bundled demo dataset, cached in
tour-private session keys (``_tour_conn``, ``_tour_schema``, ``_tour_df``), so
the narrative is deterministic regardless of what the visitor loaded in the
workshop. The workshop's ``st.session_state["conn"]`` is untouched.

Default path is the heuristic (instant, needs no key); an optional
"Generate … with LLM" button in chapters 3 and 5 runs the real LLM path when a
provider + key are configured in the ⚙ popover, falling back to the heuristic
on any failure so the page never breaks.
"""

from __future__ import annotations

import html
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from src.config import load_config
from src.executor import execute
from src.heuristic_nl import parse_heuristic
from src.llm.natural_language import load_llm_config, nl_to_query_model
from src.streamlit_app import TAB_STUDIO, state
from src.streamlit_app.components import header as header_comp
from src.streamlit_app.components import results as results_comp
from src.streamlit_app.demo_dataset import DEMO_DESCRIPTION, DEMO_NAME, load_demo
from src.streamlit_app.insight_engine import compute_insights
from src.streamlit_app.insight_enrichment import enrich_analysis, results_to_sample_csv
from src.streamlit_app.sql_highlight import render_sql_block
from src.streamlit_app.styles import inject_css

# The canonical question the tour walks through. Editable in chapter 1; the
# downstream chapters react to whatever the visitor types. This default is
# chosen because the heuristic parses it reliably into a clean GROUP BY with
# a chart-worthy result and non-empty insight cards (so the tour never opens
# on an empty/broken beat for a key-less visitor).
_DEFAULT_QUESTION = "sum revenue by region"

# Clickable suggestions in chapter 1. Each must parse via the heuristic so the
# tour stays deterministic without an LLM. (Trend questions like "monthly
# revenue trend 2024" need the LLM path — visitors who connect a key can type
# those freely.)
_EXAMPLE_QUESTIONS = (
    "sum revenue by region",
    "top 10 products by revenue",
    "count rows by status",
)

# Design tokens (mirror styles.css :root) so inline HTML matches the app.
_INK = "#1A1714"
_INK_3 = "#8E867B"
_LINE = "#E4DFD2"
_PAPER = "#FBFAF6"
_ACCENT = "#C2410C"
_ACCENT_SOFT = "#FBE9DD"
_GOOD = "#3F6B45"
_GOOD_SOFT = "#E1F0E2"
_BAD = "#8A4A11"
_BAD_SOFT = "#FAE7D0"


# ---------------------------------------------------------------------------
# Demo dataset (tour-private, decoupled from the workshop's ss.conn)
# ---------------------------------------------------------------------------


def _ensure_tour_demo() -> None:
    """Load the bundled demo dataset into tour-private session keys once.

    Decouples the tour from the workshop: the narrative always has the
    demo schema (revenue / region / date / ...) regardless of what the visitor
    uploaded in the workshop. No ``st.rerun`` — load synchronously and
    continue; beats read the keys on the same pass.
    """
    ss = st.session_state
    if ss.get("_tour_conn") is not None:
        return
    conn, schema, df, meta = load_demo()
    ss["_tour_conn"] = conn
    ss["_tour_schema"] = schema
    ss["_tour_df"] = df
    ss["_tour_meta"] = meta


def _tour_schema() -> Dict[str, str]:
    return st.session_state.get("_tour_schema", {}) or {}


def _tour_conn() -> Any:
    return st.session_state.get("_tour_conn")


def _tour_df() -> Optional[pd.DataFrame]:
    return st.session_state.get("_tour_df")


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _llm_cfg():
    return load_llm_config(load_config())


def _cloud_llm_ready(cfg) -> bool:
    """True when a cloud provider + key are configured (the visitor scenario)."""
    return cfg.provider in ("groq", "gemini") and bool(cfg.api_key)


def _llm_plan(question: str, schema: Dict[str, str]) -> Tuple[Optional[Any], Optional[dict], Optional[str]]:
    """Call the LLM for a plan. Returns (model, raw_payload, error_msg).

    ``model`` is a validated QueryModel, ``raw_payload`` is the model's raw JSON
    plan (for display), ``error_msg`` is set when the call failed. Any of the
    three may be None.
    """
    try:
        model, raw = nl_to_query_model(question, schema, return_raw=True, table_name="data")
        return model, raw, None
    except Exception as exc:  # LLMError, RouteToPythonError, network, parse — never crash the tour
        return None, None, str(exc)


# ---------------------------------------------------------------------------
# Small render helpers
# ---------------------------------------------------------------------------


def _chapter(num: int, title: str, desc: str) -> None:
    """Numbered chapter card, reusing the workshop's ``.wf-step`` styling."""
    st.markdown(
        f'<div class="wf-step">'
        f'<div class="wf-step-num">{num}</div>'
        f'<div class="wf-step-body">'
        f'<div class="wf-step-title">{html.escape(title)}</div>'
        f'<div class="wf-step-desc">{desc}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _type_chip_cls(dtype: str) -> str:
    if dtype == "numeric":
        return "type-num"
    if dtype == "date":
        return "type-date"
    return "type-text"


def _schema_chips_html(schema: Dict[str, str]) -> str:
    chips = "".join(
        f'<span class="type-chip {_type_chip_cls(dt)}">{html.escape(col)}</span>'
        for col, dt in schema.items()
    )
    return f'<div style="display:flex;flex-wrap:wrap;gap:6px;">{chips}</div>'


def _plan_breakdown_html(model) -> str:
    """Human-readable breakdown of a QueryModel's plan clauses."""
    rows: list[str] = []

    def add(k: str, v: str) -> None:
        rows.append(
            f'<div style="display:flex;gap:12px;padding:5px 10px;'
            f'border-bottom:1px solid {_LINE};">'
            f'<span style="min-width:96px;font-size:10.5px;font-weight:700;'
            f'letter-spacing:.07em;text-transform:uppercase;color:{_ACCENT};'
            f'padding-top:1px;">{k}</span>'
            f'<span style="font-size:13px;color:{_INK};font-family:IBM Plex Mono,'
            f'monospace;">{html.escape(v)}</span></div>'
        )

    sel = model.selected_columns or []
    if sel:
        add("SELECT", ", ".join(str(c) for c in sel))

    aggs = getattr(model, "aggregations", []) or []
    if aggs:
        parts = []
        for a in aggs:
            col = getattr(a, "column", "?")
            fn = getattr(a, "function", "?")
            alias = getattr(a, "alias", None)
            expr = f"{fn}({col})" if col != "*" else f"{fn}(*)"
            if alias:
                expr += f" AS {alias}"
            parts.append(expr)
        add("AGGREGATE", ", ".join(parts))

    gbs = model.group_by or []
    if gbs:
        add("GROUP BY", ", ".join(str(c) for c in gbs))

    buckets = getattr(model, "date_buckets", {}) or {}
    if buckets:
        add("DATE BUCKET", ", ".join(f"{c} → {g}" for c, g in buckets.items()))

    flts = model.filters or []
    if flts:
        parts = []
        for f in flts:
            col = getattr(f, "column", "?")
            op = getattr(f, "operator", "?")
            val = getattr(f, "value", None)
            if op.upper() in ("IS NULL", "IS NOT NULL"):
                parts.append(f"{col} {op}")
            elif val is None:
                parts.append(f"{col} {op}")
            else:
                parts.append(f"{col} {op} {val}")
        add("WHERE", " AND ".join(parts))

    obs = model.order_by or []
    if obs:
        parts = [f"{c} {d}" for (c, d) in obs]
        add("ORDER BY", ", ".join(parts))

    if model.limit is not None:
        add("LIMIT", str(model.limit))

    if not rows:
        rows.append(
            f'<div style="display:flex;gap:12px;padding:5px 10px;">'
            f'<span style="font-size:13px;color:{_INK_3};">empty plan</span></div>'
        )

    body = "".join(rows)
    return (
        f'<div style="border:1px solid {_LINE};border-radius:8px;'
        f'background:{_PAPER};overflow:hidden;">{body}</div>'
    )


def _tour_chart(df: pd.DataFrame) -> None:
    """Minimal self-contained auto-chart for chapter 5.

    Deliberately does NOT touch ``st.session_state["last_chart_specs"]`` (which
    belongs to the workshop) — the tour is decoupled.
    """
    try:
        import altair as alt
    except ImportError:
        st.info("Install altair to enable charting: `pip install altair`")
        return

    if df is None or df.empty or len(df.columns) < 2:
        st.caption("Not enough data to chart.")
        return

    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    text_cols = [
        c for c in df.columns
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])
    ]

    x_field = None
    y_field = num_cols[0] if num_cols else None
    mark = "bar"
    if date_cols:
        x_field = date_cols[0]
        mark = "line"
    elif text_cols:
        x_field = text_cols[0]
    if y_field is None:
        st.caption("No numeric column to chart.")
        return
    if x_field == y_field or x_field is None:
        st.caption("Not enough data to chart.")
        return

    chart = alt.Chart(df.head(500))
    if mark == "line":
        chart = chart.mark_line(color=_ACCENT, strokeWidth=2, point=True)
    else:
        chart = chart.mark_bar(color=_ACCENT, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    chart = chart.encode(
        x=alt.X(x_field), y=alt.Y(y_field), tooltip=[x_field, y_field]
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)


def _render_insight_cards(det) -> None:
    """Render a DeterministicAnalysis (headline + cards + prose), reusing the
    assistant panel's HTML classes."""
    if det.headline:
        st.markdown(
            f'<div class="headline-callout">{html.escape(det.headline.text)}</div>',
            unsafe_allow_html=True,
        )
    insights = det.insights[:3] if det.insights else []
    if insights:
        cards = ""
        for ins in insights:
            delta_cls = {
                "up": "delta-up", "down": "delta-down", "neutral": "delta-neutral"
            }.get(ins.direction, "delta-neutral")
            cards += (
                f'<div class="insight-card">'
                f'<div class="insight-label">{html.escape(ins.label)}</div>'
                f'<div class="insight-value">{html.escape(ins.value)}</div>'
                f'<div class="insight-delta {delta_cls}">{html.escape(ins.delta or "")}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div class="insights-grid insights-{len(insights)}">{cards}</div>',
            unsafe_allow_html=True,
        )
    if det.prose:
        st.markdown(
            f'<div class="analysis-prose">{html.escape(det.prose)}</div>',
            unsafe_allow_html=True,
        )
    for w in det.warnings or []:
        st.warning(w)


def _render_sql_panel(sql: str) -> None:
    """Read-only styled SQL block (no Run button — chapter 4 owns the Run)."""
    st.markdown(
        f'<div class="sql-panel"><div class="sql-toolbar">'
        f'<span class="dot" style="background:#E66B5B;"></span>'
        f'<span class="dot" style="background:#E6C36B;"></span>'
        f'<span class="dot" style="background:#7FB37F;"></span>'
        f'<span class="stat">SELECT · read-only</span>'
        f'</div>{render_sql_block(sql)}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Top bar + hero
# ---------------------------------------------------------------------------


def _render_bar() -> None:
    """Trimmed bar: brand + thesis + Open workshop + the ⚙ LLM popover."""
    brand, thesis, _, pop = st.columns([0.18, 0.52, 0.06, 0.24])
    with brand:
        st.markdown(
            f'<div class="brand-strip" style="padding:0;">'
            f'◈ <strong style="color:{_INK};">Query Studio</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with thesis:
        st.markdown(
            f'<div style="font-size:12.5px;color:{_INK_3};padding-top:4px;">'
            f'Ask your CSV anything. The LLM writes a plan — never SQL.'
            f'</div>',
            unsafe_allow_html=True,
        )
    with pop:
        with st.popover("⚙ LLM", use_container_width=True):
            header_comp._render_model_selector()


def _render_hero() -> None:
    st.markdown(
        f'<div style="padding:18px 0 8px 0;">'
        f'<h1 style="font-size:30px;font-weight:700;letter-spacing:-.01em;'
        f'color:{_INK};margin:0;">Ask your CSV anything. Safely.</h1>'
        f'<p style="font-size:15px;color:{_INK_3};max-width:760px;margin:10px 0 0 0;'
        f'line-height:1.55;">'
        f'Query Studio turns a plain-English question into a query plan, '
        f'validates it against your schema, builds SELECT-only SQL, and runs it '
        f'only after you have seen it. The model never touches the database. '
        f'Scroll on — every step below is live.'
        f'</p></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chapters
# ---------------------------------------------------------------------------


def _chapter_one() -> str:
    schema = _tour_schema()
    meta = st.session_state.get("_tour_meta", {}) or {}
    n_rows = meta.get("rows", "?")
    n_cols = meta.get("cols", len(schema))
    _chapter(
        1,
        "Your data, in plain English",
        f"The bundled demo: <strong>{html.escape(DEMO_NAME)}</strong> — "
        f"{n_rows:,} rows, {n_cols} columns. Type a question; that is the whole "
        f"interface.",
    )
    st.markdown(
        f'<div style="margin:6px 0 10px 0;">{_schema_chips_html(schema)}</div>',
        unsafe_allow_html=True,
    )
    # Apply a pending question from a chip click before the text_input renders
    # (a widget's own key can't be modified after instantiation — same pattern
    # as the entry script's _pending_main_tab handling).
    _pending_q = st.session_state.pop("_pending_tour_q", None)
    if _pending_q:
        st.session_state["tour_question"] = _pending_q
    st.session_state.setdefault("tour_question", _DEFAULT_QUESTION)
    st.text_input(
        "Your question",
        key="tour_question",
        label_visibility="collapsed",
        placeholder="e.g. sum revenue by region",
    )
    # Clickable example questions (pending-flag pattern; see comment above).
    chip_cols = st.columns(len(_EXAMPLE_QUESTIONS))
    for col, q in zip(chip_cols, _EXAMPLE_QUESTIONS):
        with col:
            if st.button(q, key=f"tour_chip_{q}", width="stretch"):
                st.session_state["_pending_tour_q"] = q
                st.rerun()
    return st.session_state["tour_question"]


def _chapter_two() -> None:
    _chapter(
        2,
        "Why this is safe",
        "Most natural-language-to-SQL tools hand your schema to the model and "
        "run whatever SQL it writes back. Query Studio does not.",
    )
    left, right = st.columns(2)
    with left:
        st.markdown(
            f'<div style="border:1px solid {_LINE};border-radius:8px;padding:14px;'
            f'background:{_BAD_SOFT};height:100%;">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:.08em;'
            f'text-transform:uppercase;color:{_BAD};">What most tools do</div>'
            f'<div style="font-size:13.5px;color:{_INK};margin-top:8px;'
            f'line-height:1.5;">'
            f"Question + schema &rarr; <strong>model writes SQL</strong> "
            f"&rarr; runs it. The model can emit "
            f"<code>DROP</code>, <code>DELETE</code>, or columns that don't exist, "
            f"and you never see it first."
            f"</div></div>",
            unsafe_allow_html=True,
        )
        st.code(
            "-- a naive tool might run this, sight unseen:\n"
            "DELETE FROM orders WHERE region = 'EMEA';\n"
            "-- or hallucinate a column:\n"
            "SELECT profit_margin_pct FROM orders;  -- no such column",
            language="sql",
        )
    with right:
        st.markdown(
            f'<div style="border:1px solid {_LINE};border-radius:8px;padding:14px;'
            f'background:{_GOOD_SOFT};height:100%;">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:.08em;'
            f'text-transform:uppercase;color:{_GOOD};">What Query Studio does</div>'
            f'<div style="font-size:13.5px;color:{_INK};margin-top:8px;'
            f'line-height:1.5;">'
            f"Question &rarr; <strong>model writes a plan</strong> (JSON) &rarr; "
            f"every column/operator is checked against your schema &rarr; "
            f"trusted code emits <code>SELECT</code>-only SQL &rarr; "
            f"<strong>you approve</strong> &rarr; it runs."
            f"</div></div>",
            unsafe_allow_html=True,
        )
        st.code(
            '// the model is only allowed to output a plan:\n'
            '{"aggregations":[{"function":"SUM","column":"revenue"}],\n'
            ' "group_by":["region"], "order_by":[["revenue","DESC"]]}',
            language="json",
        )
    st.markdown(
        f'<div class="safety-badge" style="margin-top:6px;">'
        f"Six layers enforce this — read-only SQLite, JSON-not-SQL, schema "
        f"validation, operator allowlist, SELECT-only code path, executor "
        f"blocklist."
        f"</div>",
        unsafe_allow_html=True,
    )


def _chapter_three(question: str, schema: Dict[str, str]) -> Tuple[Optional[Any], Optional[str]]:
    """The plan, in the open. Returns (model, sql) for downstream chapters."""
    _chapter(
        3,
        "The plan, in the open",
        "The question becomes a structured plan. The heuristic builds one "
        "instantly from rules; or ask the LLM (free cloud key in ⚙) to write "
        "its own.",
    )

    model = None
    sql: Optional[str] = None

    # --- Heuristic path (always on) ---
    heur = parse_heuristic(question, schema)
    if heur.parsed:
        model = heur.model
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;letter-spacing:.07em;'
            f'text-transform:uppercase;color:{_INK_3};margin:4px 0 6px 0;">'
            f"Heuristic plan · confidence {heur.confidence:.0%}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(_plan_breakdown_html(model), unsafe_allow_html=True)
        try:
            sql = model.to_sql()
        except Exception as exc:
            st.warning(f"Plan did not produce valid SQL: {exc}")
    else:
        st.info(
            "The heuristic couldn't parse this question. Try another, or use "
            "the LLM button below."
        )

    # --- LLM path (optional) ---
    cfg = _llm_cfg()
    llm_ready = _cloud_llm_ready(cfg) or cfg.provider in ("ollama", "ollama_remote")
    if llm_ready:
        cache: Dict[str, Tuple] = st.session_state.setdefault("_tour_llm_plan", {})
        btn = st.button(
            "Generate plan with LLM",
            key="tour_llm_plan_btn",
            help="Calls the configured provider. Falls back to the heuristic on any error.",
        )
        if btn or question in cache:
            if question not in cache:
                with st.spinner("Asking the LLM for a plan…"):
                    cache[question] = _llm_plan(question, schema)
            lm, raw, err = cache[question]
            if lm is not None:
                st.markdown(
                    f'<div style="font-size:11px;font-weight:700;letter-spacing:.07em;'
                    f'text-transform:uppercase;color:{_ACCENT};margin:10px 0 6px 0;">'
                    f"LLM plan · {html.escape(cfg.provider)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if raw:
                    st.json(raw)
                st.markdown(
                    f'<div style="font-size:11px;color:{_INK_3};margin:4px 0 6px 0;">'
                    f"Validated into a QueryModel:"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(_plan_breakdown_html(lm), unsafe_allow_html=True)
                # Prefer the LLM plan downstream when it succeeded.
                model = lm
                try:
                    sql = lm.to_sql()
                except Exception as exc:
                    st.warning(f"LLM plan did not produce valid SQL: {exc}")
            else:
                st.warning(f"LLM plan failed: {err}. Showing the heuristic plan above.")
    else:
        st.caption(
            "Connect a free cloud key in ⚙ LLM (Groq at console.groq.com or "
            "Gemini at aistudio.google.com/apikey) to let the model write its "
            "own plan here."
        )

    return model, sql


def _chapter_four(question: str, model, sql: Optional[str]) -> None:
    _chapter(
        4,
        "You approve, it runs",
        "The plan becomes SELECT-only SQL. Nothing executes until you press "
        "Run — and it can only ever read.",
    )
    if model is None or sql is None:
        st.info("No valid plan from step ③ — try a different question.")
        return

    _render_sql_panel(sql)

    results: Dict[str, pd.DataFrame] = st.session_state.setdefault("_tour_results", {})
    has_result = question in results

    run_col, _ = st.columns([0.3, 0.7])
    with run_col:
        if st.button("▶ Run query", type="primary", key="tour_run_btn"):
            try:
                results[question] = execute(_tour_conn(), sql)
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                results.pop(question, None)
            else:
                st.rerun()

    if has_result:
        df = results[question]
        n = len(df)
        st.markdown(
            f'<div class="results-stats-bar" style="margin-top:8px;">'
            f'<span class="rs"><b>{n:,}</b> rows</span>'
            f'<span class="rs-sep">·</span>'
            f'<span class="rs">{len(df.columns)} cols</span>'
            f'<span class="rs-sep">·</span>'
            f'<span class="rs mono">SELECT · read-only</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        results_comp._render_table(df)
    else:
        st.caption("Press ▶ Run query to execute the SQL above.")


def _chapter_five(question: str, model, sql: Optional[str]) -> None:
    _chapter(
        5,
        "The insight, automatically",
        "One question, a full analysis: headline finding, fact cards, and the "
        "right chart — no extra prompting. Add an LLM for a written narrative.",
    )
    results: Dict[str, pd.DataFrame] = st.session_state.get("_tour_results", {})
    df = results.get(question)
    if df is None or model is None:
        st.info("Run the query in step ④ first to see the automatic analysis.")
        return

    source_n = len(_tour_df()) if _tour_df() is not None else None
    det = compute_insights(df, model, source_row_count=source_n)

    # Headline + deterministic cards + auto chart
    _render_insight_cards(det)
    _tour_chart(df)

    # Optional LLM narrative
    cfg = _llm_cfg()
    llm_ready = _cloud_llm_ready(cfg) or cfg.provider in ("ollama", "ollama_remote")
    if llm_ready and sql:
        narratives: Dict[str, str] = st.session_state.setdefault("_tour_narrative", {})
        if st.button("Write narrative with LLM", key="tour_llm_narr_btn"):
            try:
                enriched = enrich_analysis(
                    det,
                    sql=sql,
                    user_text=question,
                    results_sample=results_to_sample_csv(df),
                    config=cfg,
                )
                narratives[question] = enriched.prose or ""
            except Exception:
                narratives.pop(question, None)
            st.rerun()
        if question in narratives and narratives[question]:
            st.markdown(
                f'<div class="analysis-prose" style="margin-top:8px;">'
                f"{html.escape(narratives[question])}"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_escape_hatch(question: str) -> None:
    st.markdown(
        f'<div style="border-top:1px solid {_LINE};margin-top:24px;padding-top:18px;">'
        f'<div style="font-size:15px;color:{_INK};font-weight:600;margin-bottom:4px;">'
        f"Want the full workshop?"
        f"</div>"
        f'<div style="font-size:13px;color:{_INK_3};margin-bottom:12px;">'
        f"Open the Studio (visual composer, SQL editor, query history) or the "
        f"LLM SQL Assistant (heuristic-vs-LLM comparison). Your question is "
        f"pre-filled."
        f"</div></div>",
        unsafe_allow_html=True,
    )
    if st.button("Open the full Studio / LLM tools →", type="primary", key="tour_open_workshop"):
        st.session_state["nl_prefill"] = question
        st.session_state["nl_auto_submit"] = False
        st.session_state["_pending_main_tab"] = TAB_STUDIO
        st.session_state["_pending_view"] = "workshop"
        st.rerun()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render() -> None:
    inject_css()
    state.init()
    _ensure_tour_demo()

    _render_bar()
    _render_hero()

    schema = _tour_schema()
    question = _chapter_one()
    _chapter_two()
    model, sql = _chapter_three(question, schema)
    _chapter_four(question, model, sql)
    _chapter_five(question, model, sql)
    _render_escape_hatch(question)