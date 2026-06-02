"""Smoke tests for the LLM SQL Assistant page (under the tab model).

The page is one of three panels in the top-level tabbed app
(``streamlit_app.py``).  These tests guard the invariants that the
page must always satisfy:

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
6. The handoff path writes the right ``st.session_state`` keys to
   switch tabs and pre-fill the Studio's ask bar — the heart of the
   cross-tab workflow.

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
    """``render`` is the entry point the tabbed app calls.

    Under the tab model the entry script imports the page module and
    calls ``render()`` explicitly inside a ``with`` block — there is
    no longer any ``__main__`` exec magic.  The function just has to
    exist and be callable.
    """
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


# ---------------------------------------------------------------------------
# 14. Tab model — entry script, handoff, and Workflow tour wiring
# ---------------------------------------------------------------------------


def test_default_tab_is_studio():
    """A fresh session must default to the Studio tab.

    The entry script sets ``st.session_state["main_tabs"] = TAB_STUDIO``
    when the key is missing.  AppTest starts with a clean session, so
    the default-tab test exercises that path.
    """
    from streamlit.testing.v1 import AppTest
    from src.streamlit_app import TAB_STUDIO

    at = AppTest.from_file("streamlit_app.py", default_timeout=60).run()
    assert not at.exception, f"entry script raised: {at.exception}"
    assert at.session_state.filtered_state.get("main_tabs") == TAB_STUDIO


def test_tabs_render_in_visible_order():
    """The three top-level tabs must appear in left-to-right order.

    Visible order is Studio, LLM, Workflow — the body execution order
    is decoupled (Studio → Workflow → LLM) so the prefill trick works,
    but the ``st.tabs`` widget arg order is what the user sees.

    Note: AppTest's ``.tabs`` property returns *every* ``st.tabs``
    widget on the page, including the Studio's inner Schema/Compose/
    History tabs.  We slice the first three (the top-level widget)
    and assert their labels match.
    """
    from streamlit.testing.v1 import AppTest
    from src.streamlit_app import TAB_LLM, TAB_STUDIO, TAB_WORKFLOW

    at = AppTest.from_file("streamlit_app.py", default_timeout=60).run()
    assert not at.exception, f"entry script raised: {at.exception}"
    # The top-level tabs widget is the first one rendered.  Inner
    # Studio tabs (Schema / Compose / History) come after.
    top_level = [t.label for t in at.tabs[:3]]
    assert top_level == [TAB_STUDIO, TAB_LLM, TAB_WORKFLOW], (
        f"visible tab order should be Studio → LLM → Workflow; "
        f"saw {top_level!r}"
    )


def test_handoff_writes_main_tabs_and_nl_prefill(demo):
    """``_handoff(model)`` must copy the plan into Studio and switch tabs.

    The four keys written are:
    - ``model``  — the deep-copied QueryModel (Studio's last_sql depends on it)
    - ``last_sql`` — the generated SQL (Studio's preview reads this)
    - ``nl_prefill`` — the LLM's plain-English reply (Studio's ask bar)
    - ``main_tabs`` — the bound tabs key, set to TAB_STUDIO

    Implementation note: we patch ``st.rerun`` to a no-op inside the
    script body and restore it at the end.  Without the patch, AppTest
    sees the snapshot taken *before* the rerun chain settles, and the
    handoff writes appear to be missing.  The patch keeps the writes
    in the same script run's session_state so AppTest captures them.

    The restore is critical: AppTest runs the script in the same
    Python process as the test, so a permanent ``st.rerun = lambda``
    patch would silently break every subsequent test that uses a
    Streamlit button.
    """
    from streamlit.testing.v1 import AppTest
    from src.streamlit_app import TAB_STUDIO

    real_model = parse_heuristic(
        "sum revenue by region", demo["schema"]
    ).model
    # Give the model a ``reply`` so ``_handoff``'s nl_prefill write
    # exercises the LLM-summary branch (not the empty-string fallback).
    real_model.reply = "Sum revenue grouped by region."

    script = textwrap.dedent(
        """
        import streamlit as st
        st.set_page_config = lambda *a, **kw: None
        # Suppress the rerun so the handoff writes settle in this run.
        # CRITICAL: restore at the end — AppTest shares the streamlit
        # module with the test process, so an unrestored patch would
        # silently break every subsequent test.
        _original_rerun = st.rerun
        st.rerun = lambda *a, **kw: None
        try:
            from src.heuristic_nl import parse_heuristic
            from src.streamlit_app.demo_dataset import load_demo
            from src.streamlit_app.pages import llm_assistant

            conn, schema, df, meta = load_demo()
            real_model = parse_heuristic("sum revenue by region", schema).model
            real_model.reply = "Sum revenue grouped by region."
            llm_assistant._handoff(real_model)
        finally:
            st.rerun = _original_rerun
        """
    )
    at = AppTest.from_string(script).run()
    ss = at.session_state.filtered_state
    assert ss.get("main_tabs") == TAB_STUDIO, (
        f"_handoff should set main_tabs to {TAB_STUDIO!r}; "
        f"got {ss.get('main_tabs')!r}"
    )
    assert ss.get("nl_prefill") == "Sum revenue grouped by region.", (
        f"nl_prefill should be the LLM's reply; "
        f"got {ss.get('nl_prefill')!r}"
    )
    assert ss.get("nl_auto_submit") is False, (
        f"nl_auto_submit should be False (no auto-submit on handoff); "
        f"got {ss.get('nl_auto_submit')!r}"
    )
    last_sql = ss.get("last_sql", "")
    assert "SUM" in last_sql.upper() and "GROUP BY" in last_sql.upper(), (
        f"last_sql should contain SUM ... GROUP BY; got {last_sql!r}"
    )
    model = ss.get("model")
    assert model is not None and model.reply == "Sum revenue grouped by region.", (
        f"model should be the deep-copied LLM plan; "
        f"got {getattr(model, 'reply', 'MISSING')!r}"
    )


def test_handoff_does_not_call_switch_page(monkeypatch, demo):
    """``_handoff`` must not invoke ``st.switch_page``.

    Under the tab model the bound ``main_tabs`` key handles tab
    switching — calling ``st.switch_page`` would blow up (no
    navigation router exists under ``st.tabs``).  We patch
    ``streamlit.switch_page`` to raise and assert the call doesn't
    fire.

    The script also patches ``st.rerun`` to a no-op and restores it.
    See ``test_handoff_writes_main_tabs_and_nl_prefill`` for why the
    restore is critical (AppTest shares the streamlit module with
    the test process).
    """
    from src.streamlit_app.pages import llm_assistant
    import streamlit as st_real

    def _boom(*args, **kwargs):
        raise AssertionError(
            "st.switch_page was called from _handoff — under the tab "
            "model this is wrong; write main_tabs instead."
        )

    monkeypatch.setattr(st_real, "switch_page", _boom)

    real_model = parse_heuristic(
        "sum revenue by region", demo["schema"]
    ).model
    real_model.reply = "Sum revenue by region."

    from streamlit.testing.v1 import AppTest
    script = textwrap.dedent(
        """
        import streamlit as st
        st.set_page_config = lambda *a, **kw: None
        _original_rerun = st.rerun
        st.rerun = lambda *a, **kw: None
        try:
            from src.heuristic_nl import parse_heuristic
            from src.streamlit_app.demo_dataset import load_demo
            from src.streamlit_app.pages import llm_assistant

            conn, schema, df, meta = load_demo()
            real_model = parse_heuristic("sum revenue by region", schema).model
            real_model.reply = "Sum revenue by region."
            llm_assistant._handoff(real_model)
        finally:
            st.rerun = _original_rerun
        """
    )
    at = AppTest.from_string(script).run()
    # If _handoff had called switch_page, the patched _boom would have
    # raised and AppTest would have surfaced it as an exception.
    assert not at.exception, f"_handoff raised: {at.exception}"


def test_workflow_step3_writes_chip_prefill(demo):
    """Clicking the Workflow Step 3 button must pre-fill the LLM tab.

    Step 3 of the guided tour writes
    ``_llm_showcase_chip_prefill`` and ``main_tabs = TAB_LLM`` so the
    LLM tab's text area picks up the seed question on its next
    render.  We assert both keys after the click.
    """
    from streamlit.testing.v1 import AppTest
    from src.streamlit_app import TAB_LLM
    from src.streamlit_app.pages import workflow

    # A small script that loads the demo and renders the workflow tab.
    # We don't need the other tabs to test this — the Workflow tab is
    # self-contained.
    script = textwrap.dedent(
        f"""
        import streamlit as st
        st.set_page_config = lambda *a, **kw: None
        from src.streamlit_app.demo_dataset import load_demo
        from src.streamlit_app.pages import workflow
        conn, schema, df, meta = load_demo()
        st.session_state["conn"] = conn
        st.session_state["schema"] = schema
        workflow.render()
        """
    )
    at = AppTest.from_string(script).run()
    assert not at.exception, f"workflow raised on first render: {at.exception}"

    # Click the Step 3 button.  The key is stable: ``wf_step3``.
    step3 = next(b for b in at.button if b.key == "wf_step3")
    step3.click().run()

    ss = at.session_state.filtered_state
    assert ss.get("main_tabs") == TAB_LLM, (
        f"Step 3 must switch to {TAB_LLM!r}; got {ss.get('main_tabs')!r}"
    )
    assert ss.get("_llm_showcase_chip_prefill") == "monthly revenue trend 2024", (
        f"Step 3 must seed the LLM text area; got "
        f"{ss.get('_llm_showcase_chip_prefill')!r}"
    )


def test_studio_ask_picks_up_workflow_prefill(demo):
    """End-to-end: Workflow Step 2 pre-fill must reach the Studio ask bar.

    This is the most important integration test.  It renders the full
    three-tab app, clicks Workflow Step 2, then asserts:
    - ``main_tabs`` switched to "Studio"
    - The Studio's ask-bar widget (``nl_text_input``) is populated
      with the seed question

    Proves the cross-tab prefill works end-to-end.

    Note: we don't assert on ``nl_prefill``'s presence/absence — that
    key is *consumed* by ``ask.render()`` via ``st.session_state.pop``
    which still leaves the key in session_state with an empty-string
    value.  The visible end-state is the ``nl_text_input`` widget value.
    """
    from streamlit.testing.v1 import AppTest

    # Render the full app so all session_state is initialized.
    at = AppTest.from_file("streamlit_app.py", default_timeout=60).run()
    assert not at.exception, f"first render raised: {at.exception}"

    # Click the Workflow Step 2 button.  The key is stable: ``wf_step2``.
    step2 = next(b for b in at.button if b.key == "wf_step2")
    step2.click().run()

    assert not at.exception, f"post-click render raised: {at.exception}"

    ss = at.session_state.filtered_state
    assert ss.get("main_tabs") == "Studio", (
        f"Step 2 must switch to Studio; got {ss.get('main_tabs')!r}"
    )
    # The ask bar's text_input widget has key="nl_text_input"; the
    # prefill should be sitting in its session_state slot.
    assert ss.get("nl_text_input") == "sum revenue by region", (
        f"Studio's ask bar should be pre-populated with the seed "
        f"question; got {ss.get('nl_text_input')!r}"
    )


def test_workflow_load_demo_button_disabled_when_loaded(demo):
    """The Load demo button must be disabled once a connection is present.

    The first click loads the demo and sets ``st.session_state.conn``.
    On the next render the button is rendered with ``disabled=True``
    and the caption shows the connected dataset's name.  This test
    catches the easy-to-miss regression where someone reverts the
    disabled-branch logic.

    Implementation: a single AppTest instance is used for both the
    pre-click and post-click renders so they share ``st.session_state``.
    The script does NOT pre-seed ``ss.conn = None`` — that would
    overwrite the conn the click handler sets in the rerun.  An
    AppTest session starts fresh, so ``ss.get("conn")`` is ``None``
    by default, which is exactly the pre-click state we want.
    """
    from streamlit.testing.v1 import AppTest
    from src.streamlit_app.demo_dataset import DEMO_NAME

    script = textwrap.dedent(
        f"""
        import streamlit as st
        st.set_page_config = lambda *a, **kw: None
        from src.streamlit_app.pages import workflow
        workflow.render()
        """
    )
    at = AppTest.from_string(script).run()
    load_btn = next(b for b in at.button if b.key == "wf_load_demo")
    assert not load_btn.disabled, (
        "Load demo button should be enabled when no connection is loaded"
    )

    # Click it — the callback calls _load_demo_into_session() and
    # triggers st.rerun().  On the rerun, conn_loaded is True, so the
    # disabled branch renders.
    load_btn.click().run()
    assert not at.exception, f"load demo raised: {at.exception}"

    # Inspect the post-click button + caption.
    load_btn2 = next(b for b in at.button if b.key == "wf_load_demo")
    assert load_btn2.disabled, (
        "Load demo button should be disabled after a connection is loaded"
    )
    captions = [c.value for c in at.caption]
    assert any(DEMO_NAME in c for c in captions), (
        f"caption should mention the loaded dataset ({DEMO_NAME!r}); "
        f"saw {captions!r}"
    )


def test_invalid_main_tabs_value_falls_back(demo):
    """A garbage ``main_tabs`` value must not crash the entry script.

    Streamlit's ``st.tabs`` silently falls back to the first tab when
    the bound key's value doesn't match any of the labels.  We assert
    the entry script renders cleanly and at least one of the three
    tabs is present.
    """
    from streamlit.testing.v1 import AppTest

    # Pre-seed session_state via a tiny wrapper script — AppTest
    # doesn't let us write to session_state from outside, but we can
    # pre-populate it inside the test script.
    script = textwrap.dedent(
        f"""
        import streamlit as st
        st.set_page_config = lambda *a, **kw: None
        # Garbage value — Streamlit should silently fall back to tab 0.
        st.session_state["main_tabs"] = "BogusTab"
        import streamlit_app
        """
    )
    at = AppTest.from_string(script).run()
    assert not at.exception, (
        f"garbage main_tabs should not crash: {at.exception}"
    )
    # All three tab labels should still be present in the element tree.
    labels = [t.label for t in at.tabs]
    assert "Studio" in labels and "LLM SQL Assistant" in labels and "Workflow" in labels
