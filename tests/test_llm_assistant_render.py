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
