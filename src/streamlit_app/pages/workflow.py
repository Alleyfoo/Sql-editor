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
   ``_pending_main_tab = TAB_STUDIO`` and pre-fills the ask bar with the
   LLM's ``model.reply``.

State sharing
-------------
All cross-tab state lives in ``st.session_state`` — see
:mod:`src.streamlit_app` for the tab-label constants.  The
``st.tabs`` widget in ``streamlit_app.py`` is bound to the
``main_tabs`` key (with ``on_change="rerun"``).  Switching tabs from a
button can't write that widget key directly — the widget is already
instantiated by the time the button runs — so the buttons write a
``_pending_main_tab`` flag that the entry script applies to ``main_tabs``
before the widget re-renders.
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


def _step_card(
    num: int,
    title: str,
    desc: str,
    done: bool = False,
    optional: bool = False,
) -> None:
    done_cls = " done" if done else ""
    optional_badge = (
        '<span class="wf-step-optional">optional</span>' if optional else ""
    )
    num_content = "✓" if done else str(num)
    st.markdown(
        f'<div class="wf-step{done_cls}">'
        f'<div class="wf-step-num">{num_content}</div>'
        f'<div class="wf-step-body">'
        f'<div class="wf-step-title">{title}</div>'
        f'{optional_badge}'
        f'<div class="wf-step-desc">{desc}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def render() -> None:
    state.init()

    st.markdown(
        "<h2 style='margin:8px 0 2px 0;font-size:20px;'>Guided tour</h2>"
        "<p style='color:#8E867B;font-size:12.5px;margin:0 0 16px 0;'>"
        "Four steps from zero to a running query — the first two work "
        "with no model, API key, or Ollama required."
        "</p>",
        unsafe_allow_html=True,
    )

    conn_loaded = st.session_state.get("conn") is not None
    schema = st.session_state.get("schema", {})

    # ---- Step 1 ---------------------------------------------------------
    _step_card(
        1,
        "Load the demo dataset",
        "3 000 synthetic B2B orders (2023–2025). One click — the connection "
        "is shared across all tabs so you only need to do this once.",
        done=conn_loaded,
    )
    if conn_loaded:
        st.button(
            "✓ Demo dataset loaded",
            disabled=True,
            key="wf_load_demo",
            width="stretch",
        )
        st.caption(
            f"Connected to **{DEMO_NAME}** — "
            f"{len(schema)} column{'s' if len(schema) != 1 else ''}."
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

    # ---- Step 2 ---------------------------------------------------------
    _step_card(
        2,
        "Try a model-free example",
        f'Prefills the Studio\'s ask bar with &ldquo;<em>{_STEP2_QUESTION}</em>&rdquo;. '
        "Press <strong>Ask</strong>, inspect the SQL preview, then "
        "<strong>▶ Run query</strong>. No Ollama or API key needed.",
        done=False,
    )
    if st.button(
        f'Open Studio → "{_STEP2_QUESTION}"',
        disabled=not conn_loaded,
        key="wf_step2",
        width="stretch",
    ):
        st.session_state["nl_prefill"] = _STEP2_QUESTION
        st.session_state["nl_auto_submit"] = False
        # Don't write the widget key directly — it can't be modified after
        # the st.tabs widget is instantiated. Write a pending flag that the
        # entry script applies before the widget re-renders. See streamlit_app.py.
        st.session_state["_pending_main_tab"] = TAB_STUDIO
        st.rerun()

    # ---- Step 3 ---------------------------------------------------------
    _step_card(
        3,
        "Compare with an LLM",
        f'Opens the LLM tab with &ldquo;<em>{_STEP3_QUESTION}</em>&rdquo; prefilled. '
        "Run the comparison to see heuristic vs model side-by-side. "
        "If no model is connected the heuristic column still works.",
        done=False,
        optional=True,
    )
    if st.button(
        f'Open LLM tab → "{_STEP3_QUESTION}"',
        disabled=not conn_loaded,
        key="wf_step3",
        width="stretch",
    ):
        st.session_state["_llm_showcase_chip_prefill"] = _STEP3_QUESTION
        # Pending flag, not the widget key — see streamlit_app.py.
        st.session_state["_pending_main_tab"] = TAB_LLM
        st.rerun()

    # ---- Step 4 ---------------------------------------------------------
    _step_card(
        4,
        "Hand the plan back to Studio",
        "On the LLM tab, press <strong>Use LLM plan</strong> or "
        "<strong>Use heuristic plan</strong>. The chosen plan is copied "
        "into the Studio and the tab switches back automatically. "
        "Press <strong>▶ Run query</strong> to execute.",
        done=False,
        optional=True,
    )
    st.caption(
        "This step has no button here — visit the LLM tab and use "
        "one of the handoff buttons there."
    )

