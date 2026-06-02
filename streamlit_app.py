"""Query Studio — single-script app with three top-of-page tabs.

Replaces the previous ``st.navigation`` setup.  All three panels
(Studio, LLM SQL Assistant, Workflow) live in the same Streamlit
script and share ``st.session_state``.  Cross-tab handoff is a
``st.session_state["main_tabs"]`` write (the bound key on the
``st.tabs`` widget) — no ``st.switch_page`` is needed.

Visible tab order (left → right):
  Studio · LLM SQL Assistant · Workflow

Body-execution order:
  Studio → Workflow → LLM SQL Assistant

The body order is deliberately **not** the visible order: the
Workflow tab's buttons write private prefill keys
(``_llm_showcase_chip_prefill``, ``nl_prefill``), and the LLM tab's
``render()`` pops those keys *before* instantiating its widgets.  By
running the LLM tab last in the script, the prefill is visible in
the same top-to-bottom pass — no extra rerun needed.

Tab labels live in :mod:`src.streamlit_app` as module-level
constants so the entry script and the page modules can never get
out of sync.
"""

from __future__ import annotations

import streamlit as st

from src.streamlit_app import TAB_LLM, TAB_STUDIO, TAB_WORKFLOW

# Page config is set ONCE here.  The page modules no longer call it.
st.set_page_config(
    page_title="Query Studio",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Default to Studio on first visit (and after any session reset).
if "main_tabs" not in st.session_state:
    st.session_state["main_tabs"] = TAB_STUDIO

# Visible order: Studio, LLM, Workflow.
tab_studio, tab_llm, tab_workflow = st.tabs(
    [TAB_STUDIO, TAB_LLM, TAB_WORKFLOW]
)

# Body order: Studio (1st), Workflow (2nd — writes prefill keys),
# LLM (3rd — pops prefill keys).  Labels are decoupled from the
# order of the ``with`` blocks; only the script order matters for
# the prefill trick to work.
with tab_studio:
    from src.streamlit_app.pages import studio
    studio.render()

with tab_workflow:
    from src.streamlit_app.pages import workflow
    workflow.render()

with tab_llm:
    from src.streamlit_app.pages import llm_assistant
    llm_assistant.render()
