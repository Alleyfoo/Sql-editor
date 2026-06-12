"""Workflow — interactive guided tour tab.

One of three panels in the top-level tabbed app
(``streamlit_app.py``).  A four-step walkthrough that takes the user
from an empty session to a query plan produced by the LLM and run
against the demo dataset.  Every step is one click; the user can
leave the tour at any time.

Steps
-----
1. Load the demo dataset.  Seeds ``st.session_state`` the same way
   the Studio's load path does, so both tabs have a connection.
2. Try the heuristic on Studio.  Switches to the Studio tab with a
   pre-filled ask bar (``nl_prefill``).  Reuses the existing prefill
   pattern in :mod:`src.streamlit_app.components.ask`.
3. Compare with the LLM.  Switches to the LLM tab with a pre-filled
   text area (``_llm_showcase_chip_prefill``).  Reuses the existing
   prefill pattern in :mod:`src.streamlit_app.pages.llm_assistant`.
4. Hand the LLM plan back to the Studio.  Static instruction — the
   user clicks **Use LLM plan** on the LLM tab, which writes
   ``main_tabs = TAB_STUDIO`` and pre-fills the ask bar with the
   LLM's ``model.reply``.

State sharing
-------------
All cross-tab state lives in ``st.session_state`` — see
:mod:`src.streamlit_app` for the tab-label constants.  The
``st.tabs`` widget in ``streamlit_app.py`` is bound to the
``main_tabs`` key; writing it switches the active tab.
"""

from __future__ import annotations

import streamlit as st

from src.streamlit_app import TAB_LLM, TAB_STUDIO, state
from src.streamlit_app.demo_dataset import DEMO_NAME, load_demo


# Step 2 / Step 3 NL examples — used to pre-fill the other tabs.
_STEP2_QUESTION = "sum revenue by region"
_STEP3_QUESTION = "monthly revenue trend 2024"


def _load_demo_into_session() -> None:
    """Seed session_state with the bundled demo dataset.

    Mirrors the keys set by the Studio's load path so the rest of
    the app (sidebar, schema strip, SQL preview, LLM tab) sees a
    fully loaded connection.  Idempotent — caller should disable
    the button when ``st.session_state.get("conn")`` is not None.
    """
    conn, schema, df, meta = load_demo()
    ss = st.session_state
    # If a previous conn is hanging around, close it cleanly.
    if ss.get("conn") is not None:
        try:
            ss.conn.close()
        except Exception:
            pass
    ss.conn = conn
    ss.schema = schema
    ss.tables = {}
    ss.dataset_name = DEMO_NAME
    ss.dataset_meta = meta
    ss.dataset_df = df
    state.reset_query()


def render() -> None:
    state.init()

    st.markdown(
        "<h2 style='margin:8px 0 4px 0;'>Workflow — guided tour</h2>"
        "<p style='color:#57514A;font-size:13px;margin:0 0 18px 0;'>"
        "Start with the model-free path: load the demo, try an offline "
        "example, inspect the generated SQL, and run it.  If Ollama or "
        "an API key is available, the LLM comparison step shows what a "
        "model adds; otherwise the first two steps are still a complete "
        "working demo."
        "</p>",
        unsafe_allow_html=True,
    )

    conn_loaded = st.session_state.get("conn") is not None
    schema = st.session_state.get("schema", {})

    # ---- Step 1 ---------------------------------------------------------
    st.markdown("### 1. Load the demo dataset")
    st.markdown(
        "The Studio and the LLM tab both need a connection.  The "
        "bundled demo dataset is 3 000 synthetic B2B orders from "
        "2023–2025.  Click below to load it; the same connection is "
        "shared across tabs."
    )
    if conn_loaded:
        st.button(
            "Demo dataset loaded ✓",
            disabled=True,
            key="wf_load_demo",
            width="stretch",
        )
        st.caption(
            f"Connected to **{DEMO_NAME}** "
            f"({len(schema)} column{'s' if len(schema) != 1 else ''})."
        )
    else:
        if st.button(
            "Load demo dataset",
            type="primary",
            key="wf_load_demo",
            width="stretch",
        ):
            try:
                _load_demo_into_session()
            except Exception as exc:  # pragma: no cover - dataset missing
                st.error(f"Could not load demo: {exc}")
            else:
                st.rerun()

    st.markdown("---")

    # ---- Step 2 ---------------------------------------------------------
    st.markdown("### 2. Try a model-free example")
    st.markdown(
        "The Studio's ask bar uses the offline heuristic for "
        "unambiguous phrasings.  Click below to seed the ask bar with "
        f"“{_STEP2_QUESTION}” and switch to the Studio tab.  Press "
        "**Ask**, inspect the SQL preview, then press **Run query**.  "
        "This does not require Ollama or an API key."
    )
    if st.button(
        f"Open Studio with “{_STEP2_QUESTION}”",
        disabled=not conn_loaded,
        key="wf_step2",
        width="stretch",
    ):
        st.session_state["nl_prefill"] = _STEP2_QUESTION
        st.session_state["nl_auto_submit"] = False
        st.session_state["main_tabs"] = TAB_STUDIO
        st.rerun()

    st.markdown("---")

    # ---- Step 3 ---------------------------------------------------------
    st.markdown("### 3. Optional: compare with an LLM")
    st.markdown(
        "If a model is connected, run the same kind of question through "
        "the LLM tab and compare it with the heuristic.  If the LLM is "
        "offline, the tab still shows the heuristic result and explains "
        "what to connect before pressing **Run comparison**."
    )
    if st.button(
        f"Open LLM tab with “{_STEP3_QUESTION}”",
        disabled=not conn_loaded,
        key="wf_step3",
        width="stretch",
    ):
        st.session_state["_llm_showcase_chip_prefill"] = _STEP3_QUESTION
        st.session_state["main_tabs"] = TAB_LLM
        st.rerun()

    st.markdown("---")

    # ---- Step 4 ---------------------------------------------------------
    st.markdown("### 4. Hand the LLM plan back to the Studio")
    st.markdown(
        "On the LLM tab, run the comparison and then click **Use LLM "
        "plan** (or **Use heuristic plan** — try both).  The chosen "
        "QueryModel is copied into the Studio's session state and the "
        "tab switches back to Studio with the SQL preview already "
        "populated.  Press **▶ Run query** to execute."
    )
    st.caption(
        "Nothing happens on this step until you visit the LLM tab "
        "yourself — there is no button here on purpose."
    )

    st.markdown("---")
    st.caption(
        "Tip: leaving this tab is fine.  The session state is shared "
        "— if you already have a connection loaded, every step is one "
        "click."
    )
