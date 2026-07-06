"""Smoke tests for the Tour page — the scroll-driven presentation front door.

The Tour (``src/streamlit_app/pages/tour.py``) is the default landing view.
It tells the core thesis as five live beats (ask → safe plan → approve → run
→ auto-insight) against a tour-private demo connection, decoupled from the
workshop's ``st.session_state["conn"]``.

These tests guard the invariants the tour must always satisfy:

1. The module exposes ``render`` and renders without exception, loading the
   demo into tour-private session keys.
2. All five chapter cards render.
3. The default question parses via the heuristic into a SELECT with SUM and
   GROUP BY (so the tour never opens on an empty/broken beat for a key-less
   visitor).
4. The chapter-4 SQL panel renders the SELECT on a passive render.
5. Clicking an example chip changes the question (pending-flag pattern).
6. Clicking ▶ Run query executes and caches the result.
7. The escape-hatch button writes the right keys to switch to the workshop
   and pre-fill Studio.
8. A passive render never calls the LLM (the "Generate plan with LLM" button
   is the only call site).

We use ``streamlit.testing.v1.AppTest`` to run the page inside a simulated
Streamlit session.
"""

from __future__ import annotations

import textwrap

import pytest

from src.heuristic_nl import parse_heuristic
from src.streamlit_app.demo_dataset import load_demo
from src.streamlit_app.pages import tour


# ---------------------------------------------------------------------------
# Script helpers
# ---------------------------------------------------------------------------


def _tour_script(extra_body: str = "") -> str:
    """AppTest source that renders the tour page in isolation."""
    return textwrap.dedent(
        f"""
        import streamlit as st
        st.set_page_config = lambda *a, **kw: None
        from src.streamlit_app.pages import tour
        {extra_body}
        tour.render()
        """
    )


# ---------------------------------------------------------------------------
# 1. Module wiring + clean render
# ---------------------------------------------------------------------------


def test_tour_module_exposes_render():
    assert hasattr(tour, "render") and callable(tour.render)


def test_tour_renders_and_loads_demo():
    """A passive render must not raise and must seed the tour-private demo."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_tour_script()).run()
    assert not at.exception, f"tour raised: {at.exception}"
    ss = at.session_state.filtered_state
    assert ss.get("_tour_conn") is not None, "tour must load the demo connection"
    assert ss.get("_tour_schema"), "tour must seed the schema"
    assert ss.get("tour_question") == tour._DEFAULT_QUESTION


def test_tour_renders_five_chapters():
    """The five numbered chapter cards must be present."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_tour_script()).run()
    assert not at.exception
    # Filter on the rendered card div, not the class name: inject_css() emits a
    # <style> block that defines the .wf-step-title rule, which would otherwise
    # be counted as a chapter card.
    chapter_md = [m.value for m in at.markdown if '<div class="wf-step">' in m.value]
    assert len(chapter_md) == 5, (
        f"expected five chapter cards; saw {len(chapter_md)}: {chapter_md!r}"
    )


# ---------------------------------------------------------------------------
# 2. The default question is heuristic-friendly
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_schema():
    _, schema, _, _ = load_demo()
    return schema


def test_default_question_parses_via_heuristic(demo_schema):
    """The default question must produce a clean SELECT plan without an LLM."""
    res = parse_heuristic(tour._DEFAULT_QUESTION, demo_schema)
    assert res.parsed, (
        f"the tour default {tour._DEFAULT_QUESTION!r} must parse via the "
        f"heuristic so the tour works for a key-less visitor"
    )
    sql = res.model.to_sql().upper()
    assert sql.startswith("SELECT")
    assert "SUM" in sql and "GROUP BY" in sql


def test_tour_example_questions_all_parse(demo_schema):
    """Every clickable example chip must parse via the heuristic."""
    for q in tour._EXAMPLE_QUESTIONS:
        res = parse_heuristic(q, demo_schema)
        assert res.parsed, f"example chip {q!r} must parse via the heuristic"


# ---------------------------------------------------------------------------
# 3. Chapter 4 SQL panel renders on a passive render
# ---------------------------------------------------------------------------


def test_tour_renders_sql_panel_passively():
    """The SELECT SQL must be visible on a passive render (no Run click)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_tour_script()).run()
    assert not at.exception
    md = [m.value for m in at.markdown]
    # render_sql_block wraps keywords in <span class="kw">…</span>; SELECT is
    # always the first keyword of a plan query.
    assert any("SELECT" in m for m in md), (
        f"chapter 4 should render the SELECT SQL panel; saw markdown: {md!r}"
    )


# ---------------------------------------------------------------------------
# 4. Example chip changes the question
# ---------------------------------------------------------------------------


def test_tour_chip_changes_question():
    """Clicking an example chip must update ``tour_question`` via the
    pending-flag pattern (it can't write the text_input's key directly)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_tour_script()).run()
    assert not at.exception
    target = "count rows by status"
    chip = next(b for b in at.button if b.key == f"tour_chip_{target}")
    chip.click().run()
    assert not at.exception
    assert at.session_state.filtered_state.get("tour_question") == target, (
        f"chip click must set tour_question to {target!r}; got "
        f"{at.session_state.filtered_state.get('tour_question')!r}"
    )


# ---------------------------------------------------------------------------
# 5. Run query executes and caches the result
# ---------------------------------------------------------------------------


def test_tour_run_executes_and_caches():
    """Clicking ▶ Run query must execute the SQL and cache the result frame."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_tour_script()).run()
    assert not at.exception
    run_btn = next(b for b in at.button if b.key == "tour_run_btn")
    run_btn.click().run()
    assert not at.exception, f"run raised: {at.exception}"
    results = at.session_state.filtered_state.get("_tour_results", {})
    assert tour._DEFAULT_QUESTION in results, (
        f"run must cache the result under the question key; saw {list(results)!r}"
    )
    df = results[tour._DEFAULT_QUESTION]
    assert df is not None and not df.empty, "run must produce a non-empty frame"


# ---------------------------------------------------------------------------
# 6. Escape hatch writes the workshop-switch + prefill keys
# ---------------------------------------------------------------------------


def test_tour_escape_hatch_writes_keys():
    """The escape-hatch button must switch to the workshop and pre-fill Studio."""
    from streamlit.testing.v1 import AppTest
    from src.streamlit_app import TAB_STUDIO

    at = AppTest.from_string(_tour_script()).run()
    assert not at.exception
    esc = next(b for b in at.button if b.key == "tour_open_workshop")
    esc.click().run()
    assert not at.exception
    ss = at.session_state.filtered_state
    assert ss.get("_pending_view") == "workshop", (
        f"escape hatch must set _pending_view=workshop; got "
        f"{ss.get('_pending_view')!r}"
    )
    assert ss.get("_pending_main_tab") == TAB_STUDIO, (
        f"escape hatch must set _pending_main_tab=Studio; got "
        f"{ss.get('_pending_main_tab')!r}"
    )
    assert ss.get("nl_prefill") == tour._DEFAULT_QUESTION, (
        f"escape hatch must pre-fill nl_prefill with the tour question; got "
        f"{ss.get('nl_prefill')!r}"
    )


# ---------------------------------------------------------------------------
# 7. Passive render does not call the LLM
# ---------------------------------------------------------------------------


def test_tour_passive_render_does_not_call_llm(monkeypatch):
    """A passive render must never trigger an outbound LLM call.

    The "Generate plan with LLM" button is the only LLM call site.  We patch
    ``nl_to_query_model`` to raise; a passive render (no button click) must
    not hit it.
    """
    def _boom(*args, **kwargs):
        raise AssertionError(
            "LLM was called on a passive tour render — it must only fire "
            "when the visitor presses 'Generate plan with LLM'."
        )

    monkeypatch.setattr(tour, "nl_to_query_model", _boom)

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_tour_script()).run()
    assert not at.exception, f"tour raised: {at.exception}"