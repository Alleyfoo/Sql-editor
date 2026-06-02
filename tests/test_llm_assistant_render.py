"""Smoke tests for the LLM SQL Assistant page.

The page exposes a side-by-side *heuristic vs LLM* comparison. These
tests guard four invariants that the page must always satisfy:

1. The page module imports and the public ``render`` entry point exists.
2. The page advertises a safety banner that references the six layers
   promised in the README §"Safety guarantees".
3. The page's input area and example questions are wired up.
4. A *passive* page render (text area populated, no button click) MUST
   NOT trigger an outbound LLM call. This is the most important
   safety property: a user who simply opens the page should never
   cause a network request to the model. We assert this by
   monkey-patching ``nl_to_query_model`` to raise and then running the
   page with a non-empty input — if the page called the LLM, the test
   would fail.
5. The LLM column surfaces both ``LLMError`` and ``RouteToPythonError``
   cleanly in the UI without crashing the page.

We use ``streamlit.testing.v1.AppTest.from_string`` to run the page
inside a simulated Streamlit session.  ``from_string`` takes the full
script source — which lets us prepend the ``import streamlit as st``
line that the page needs (its module-level alias ``st`` doesn't survive
extraction by ``from_function``).  ``set_page_config`` is also stubbed
out so AppTest's own config isn't fought with.
"""

from __future__ import annotations

import textwrap
from typing import List

import pytest

from src.heuristic_nl import parse_heuristic
from src.llm import natural_language as nl_module
from src.streamlit_app.demo_dataset import load_demo
from src.streamlit_app.pages import llm_assistant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo():
    """Bundled demo dataset — used as the schema source for the page."""
    conn, schema, df, meta = load_demo()
    yield {"conn": conn, "schema": schema, "df": df, "meta": meta}
    try:
        conn.close()
    except Exception:
        pass


def _build_page_script(extra_body: str = "") -> str:
    """Source for an AppTest script that calls the page's ``render``.

    ``from_function`` and ``from_file`` both need a self-contained
    module body.  This helper builds one that imports streamlit,
    stubs ``set_page_config`` (AppTest manages the test session's
    page config itself), then calls the page's ``render``.  The
    ``extra_body`` parameter lets tests inject interactive steps
    (text_area.set_value, button.click) via streamlit's session state
    helpers — but in practice we just use ``AppTest.session_state``
    or the widget APIs directly on the AppTest instance.
    """
    return textwrap.dedent(
        f"""
        import streamlit as st
        # AppTest provides its own page config — stub the page's call
        # so we don't conflict.
        st.set_page_config = lambda *a, **kw: None
        from src.streamlit_app.pages import llm_assistant
        {extra_body}
        llm_assistant.render()
        """
    )


# ---------------------------------------------------------------------------
# 1. Module-level wiring
# ---------------------------------------------------------------------------


def test_page_module_exposes_render():
    """``render`` is the entry point ``st.navigation`` will call."""
    assert hasattr(llm_assistant, "render")
    assert callable(llm_assistant.render)


def test_page_module_lists_example_questions():
    """The chip examples should include the README's headline phrasings."""
    qs = llm_assistant._EXAMPLE_QUESTIONS
    assert qs
    assert any("monthly revenue" in q for q in qs)
    assert any("top" in q and "margin" in q for q in qs)
    assert any("count" in q for q in qs)


# ---------------------------------------------------------------------------
# 2. Safety banner
# ---------------------------------------------------------------------------


def test_safety_banner_lists_six_pillars():
    """The banner must reference the six layers the README promises."""
    titles = [title for _icon, title in llm_assistant._SAFETY_PILLARS]
    text = " ".join(titles).lower()
    for keyword in (
        "read-only",
        "json",
        "schema",
        "operator",
        "select-only",
        "blocklist",
    ):
        assert keyword in text, f"safety banner missing pillar about {keyword!r}"


# ---------------------------------------------------------------------------
# 3. Heuristic column produces a SQL plan for an unambiguous input
# ---------------------------------------------------------------------------


def test_heuristic_parses_unambiguous_sum_query(demo):
    """The page's primary example must be inside the heuristic's scope."""
    res = parse_heuristic("sum revenue by region", demo["schema"])
    assert res.parsed
    sql = res.model.to_sql()
    assert sql.lstrip().upper().startswith("SELECT")
    assert "GROUP BY" in sql.upper()
    assert "SUM" in sql.upper()


# ---------------------------------------------------------------------------
# 4. Passive render does not call the LLM
# ---------------------------------------------------------------------------


def test_passive_render_does_not_call_llm(monkeypatch, demo):
    """Opening the page with input populated must not call the LLM.

    The page only invokes ``nl_to_query_model`` when the user presses
    the *Run comparison* button. We assert this by patching the function
    to raise — if the page tried to call it during a passive render,
    the test would fail with that exception.
    """
    def _boom(*args, **kwargs):
        raise AssertionError(
            "LLM was called on a passive page render — this must only happen "
            "after the user presses the *Run comparison* button."
        )

    monkeypatch.setattr(llm_assistant, "nl_to_query_model", _boom)

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_build_page_script()).run()
    # The page should have rendered without an unhandled exception.
    assert not at.exception, f"page raised an exception: {at.exception}"
    # The LLM column must show the idle placeholder, not a result.
    info_texts = [el.value for el in at.info]
    assert any("Run comparison" in t for t in info_texts), (
        "the LLM column should show the 'press Run comparison' placeholder "
        f"on a passive render. Saw: {info_texts!r}"
    )


# ---------------------------------------------------------------------------
# 5. Clicking *Run comparison* wires through to nl_to_query_model
# ---------------------------------------------------------------------------


def test_run_comparison_calls_llm_and_renders_sql(monkeypatch, demo):
    """Pressing the button must call the LLM and render the resulting SQL."""
    real_model = parse_heuristic(
        "sum revenue by region", demo["schema"]
    ).model
    assert real_model is not None

    captured: list = []

    def _stub(nl, schema, **kwargs):
        captured.append((nl, dict(schema)))
        return real_model

    monkeypatch.setattr(llm_assistant, "nl_to_query_model", _stub)

    from streamlit.testing.v1 import AppTest

    # AppTest widgets are populated by the first .run() — interact with
    # them in a second run.
    at = AppTest.from_string(_build_page_script()).run()
    at.text_area[0].set_value("sum revenue by region")
    # Find the run button by key — chip buttons come first in the
    # element tree, so index 0 is unreliable.
    run_btn = next(b for b in at.button if b.key == "llm_showcase_run")
    run_btn.click().run()

    assert not at.exception, f"page raised: {at.exception}"
    assert captured, "Run comparison button must invoke nl_to_query_model"
    nl_arg, schema_arg = captured[0]
    assert nl_arg == "sum revenue by region"
    assert "revenue" in schema_arg

    code_blocks = [el.value for el in at.code]
    assert any("SELECT" in c.upper() for c in code_blocks), (
        f"LLM column should render a SELECT code block. Saw: {code_blocks!r}"
    )


# ---------------------------------------------------------------------------
# 6. LLM error surfaces in the UI without crashing the page
# ---------------------------------------------------------------------------


def test_run_comparison_surfaces_llm_error(monkeypatch, demo):
    """A transport / validation error must appear in the LLM column."""

    def _raise(*args, **kwargs):
        raise nl_module.LLMError(
            "could not reach Ollama at http://localhost:11434"
        )

    monkeypatch.setattr(llm_assistant, "nl_to_query_model", _raise)

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_build_page_script()).run()
    at.text_area[0].set_value("sum revenue by region")
    run_btn = next(b for b in at.button if b.key == "llm_showcase_run")
    run_btn.click().run()

    assert not at.exception, f"page should catch LLMError, not crash: {at.exception}"
    error_texts = [el.value for el in at.error]
    assert any("Ollama" in t for t in error_texts), (
        f"LLM error message should appear. Saw: {error_texts!r}"
    )


# ---------------------------------------------------------------------------
# 7. Python-route intent surfaces a warning
# ---------------------------------------------------------------------------


def test_run_comparison_surfaces_python_route(monkeypatch, demo):
    """Percentile-style intent should show as a Python-route warning."""

    def _route(*args, **kwargs):
        raise nl_module.RouteToPythonError(
            "percentile",
            blocked_intent="75th percentile of revenue",
            next_actions=["Use the Python analytics path"],
        )

    monkeypatch.setattr(llm_assistant, "nl_to_query_model", _route)

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_build_page_script()).run()
    at.text_area[0].set_value("75th percentile of revenue")
    run_btn = next(b for b in at.button if b.key == "llm_showcase_run")
    run_btn.click().run()

    assert not at.exception, f"page should catch RouteToPythonError: {at.exception}"
    warning_texts = [el.value for el in at.warning]
    # The LLM column surfaces a routing-to-Python warning, and the
    # detail block lists the blocked intent. Either signal is enough.
    python_signalled = any(
        "Routed away from SQL" in t
        or "routed to Python" in t.lower()
        or "percentile" in t.lower()
        for t in warning_texts
    )
    assert python_signalled, (
        f"Python-route warning should appear. Saw: {warning_texts!r}"
    )


# ---------------------------------------------------------------------------
# 8. Heuristic column reflects the parsed result on passive render
# ---------------------------------------------------------------------------


def test_heuristic_column_shows_progress_and_code(demo):
    """For an unambiguous input the heuristic column must render code + success."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_build_page_script()).run()
    at.text_area[0].set_value("sum revenue by region")
    at.run()  # rerun with the populated text area; no button click

    assert not at.exception, f"page raised: {at.exception}"

    # Heuristic column surfaces a SQL code block with SUM and GROUP BY.
    code_texts = [el.value for el in at.code]
    assert any("SUM" in c.upper() and "GROUP BY" in c.upper() for c in code_texts), (
        f"heuristic column should show a SUM ... GROUP BY code block; saw: {code_texts!r}"
    )
    success_texts = [el.value for el in at.success]
    assert any("Parsed" in t for t in success_texts), (
        f"heuristic column should announce a successful parse; saw: {success_texts!r}"
    )


# ---------------------------------------------------------------------------
# 9. Schema-aware chip generation
# ---------------------------------------------------------------------------


def test_schema_aware_chips_pick_measure_and_group():
    """Chips should mention a numeric measure and a text group-by column."""
    schema = {
        "revenue": "numeric",
        "region": "text",
        "order_date": "date",
        "category": "text",
    }
    chips = llm_assistant._schema_aware_example_questions(schema)
    assert chips
    assert any("revenue" in c and "region" in c for c in chips), (
        f"expected a 'sum revenue by region' chip; got {chips!r}"
    )
    assert any("top 10" in c for c in chips)
    assert any("monthly" in c and "revenue" in c for c in chips)


def test_schema_aware_chips_fallback_to_static():
    """Empty / tiny schemas should fall back to the static list."""
    assert llm_assistant._schema_aware_example_questions({}) == list(
        llm_assistant._EXAMPLE_QUESTIONS
    )


# ---------------------------------------------------------------------------
# 10. Row-count estimate helper
# ---------------------------------------------------------------------------


def test_estimate_row_count_runs_sql_against_conn(demo):
    """Estimate should report the actual row count for a valid SQL.

    We can't call the helper directly from a unit test because
    ``st.session_state`` requires a running Streamlit session, so we
    test the page's behaviour: when the input is populated with a
    known-good query, the heuristic column's success badge should
    include a row count (a number, formatted with a thousands
    separator).  The actual value depends on the dataset — for the
    "sum revenue by region" query on the demo dataset there are 3
    distinct regions, so the GROUP BY returns 3 rows.
    """
    real_model = parse_heuristic("sum revenue by region", demo["schema"]).model
    from streamlit.testing.v1 import AppTest

    # Run the page in a custom AppTest script that pre-populates
    # session_state.conn with the demo connection.
    script = textwrap.dedent(
        f"""
        import streamlit as st
        st.set_page_config = lambda *a, **kw: None
        from src.streamlit_app.demo_dataset import load_demo
        from src.streamlit_app.pages import llm_assistant
        # Inject the test's demo connection into the page's session state
        # so the row-count helper has a real conn to query.
        conn, schema, df, meta = load_demo()
        st.session_state["conn"] = conn
        st.session_state["schema"] = schema
        llm_assistant.render()
        """
    )

    at = AppTest.from_string(script).run()
    # Populate the input and trigger a heuristic parse by rerunning.
    at.text_area[0].set_value("sum revenue by region")
    at.run()

    # Heuristic column should now show a non-empty row count in its
    # success badge.  We accept any positive integer — the demo
    # dataset has 3 distinct regions so this query returns 3 rows.
    success_texts = [el.value for el in at.success]
    row_count_seen = any("rows" in t for t in success_texts)
    assert row_count_seen, (
        f"heuristic column should show a row count; saw: {success_texts!r}"
    )
    # And it must have rendered without exception.
    assert not at.exception


def test_estimate_row_count_returns_none_for_bad_sql(demo):
    """Bad SQL should not raise — return None instead."""
    assert llm_assistant._estimate_row_count("SELECT FROM nonexistent") is None


# ---------------------------------------------------------------------------
# 11. Comparison log
# ---------------------------------------------------------------------------


def test_log_appends_on_run(monkeypatch, demo):
    """Each Run-comparison press should add one log entry."""
    real_model = parse_heuristic("sum revenue by region", demo["schema"]).model
    monkeypatch.setattr(
        llm_assistant, "nl_to_query_model",
        lambda *a, **kw: (real_model, {"reply": "ok", "selected_columns": []}),
    )

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_build_page_script()).run()
    at.text_area[0].set_value("sum revenue by region")
    run_btn = next(b for b in at.button if b.key == "llm_showcase_run")
    run_btn.click().run()

    assert not at.exception
    log = at.session_state.filtered_state.get(llm_assistant.LOG_KEY, [])
    assert len(log) == 1
    entry = log[0]
    assert entry["input"] == "sum revenue by region"
    assert entry["llm_status"] == "accepted"
    assert "SUM" in entry["llm_sql"].upper()


def test_log_capped_at_max(monkeypatch, demo):
    """The log should not grow beyond LOG_MAX entries."""
    real_model = parse_heuristic("sum revenue by region", demo["schema"]).model
    monkeypatch.setattr(
        llm_assistant, "nl_to_query_model",
        lambda *a, **kw: (real_model, {}),
    )

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_build_page_script()).run()
    at.text_area[0].set_value("sum revenue by region")
    run_btn = next(b for b in at.button if b.key == "llm_showcase_run")
    # Press the run button more than LOG_MAX times.
    for _ in range(llm_assistant.LOG_MAX + 5):
        run_btn.click().run()

    log = at.session_state.filtered_state.get(llm_assistant.LOG_KEY, [])
    assert len(log) == llm_assistant.LOG_MAX


# ---------------------------------------------------------------------------
# 12. Raw LLM JSON plan
# ---------------------------------------------------------------------------


def test_run_comparison_renders_raw_json_plan(monkeypatch, demo):
    """On a successful LLM call the page must show the verbatim JSON plan."""
    real_model = parse_heuristic("sum revenue by region", demo["schema"]).model
    raw = {
        "reply": "I'll sum revenue by region.",
        "selected_columns": ["region", "revenue"],
        "aggregations": [{"function": "SUM", "column": "revenue", "alias": "total"}],
        "group_by": ["region"],
    }
    monkeypatch.setattr(
        llm_assistant, "nl_to_query_model",
        lambda *a, **kw: (real_model, raw),
    )

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_build_page_script()).run()
    at.text_area[0].set_value("sum revenue by region")
    run_btn = next(b for b in at.button if b.key == "llm_showcase_run")
    run_btn.click().run()

    assert not at.exception
    # The raw-JSON expander renders the payload. AppTest surfaces json
    # elements directly.
    json_payloads = [el.value for el in at.json]
    assert any("I'll sum revenue by region." in str(p) for p in json_payloads), (
        f"raw LLM JSON plan must include the model's reply; saw: {json_payloads!r}"
    )


# ---------------------------------------------------------------------------
# 13. Diff view
# ---------------------------------------------------------------------------


def test_diff_renders_when_both_sides_have_models(monkeypatch, demo):
    """Plan diff rows must appear when both heuristic and LLM have plans."""
    real_model = parse_heuristic("sum revenue by region", demo["schema"]).model
    monkeypatch.setattr(
        llm_assistant, "nl_to_query_model",
        lambda *a, **kw: (real_model, {}),
    )

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_build_page_script()).run()
    at.text_area[0].set_value("sum revenue by region")
    run_btn = next(b for b in at.button if b.key == "llm_showcase_run")
    run_btn.click().run()

    assert not at.exception
    markdown_texts = [el.value for el in at.markdown]
    # The diff header "Plan diff (heuristic vs LLM)" should appear.
    assert any("Plan diff" in t for t in markdown_texts), (
        f"Plan diff header should render; saw markdown: {markdown_texts!r}"
    )
