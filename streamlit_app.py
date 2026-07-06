"""Query Studio — single-script app with two views: Tour and Workshop.

The **Tour** (``pages/tour.py``) is the default landing: a scroll-driven
presentation that tells the core thesis as five live beats (ask → safe plan →
approve → run → auto-insight). The **Workshop** is the full three-panel
experience — the Studio (visual composer + SQL editor + results) and the LLM
SQL Assistant — reachable behind an "Open the workshop" escape hatch on the
tour, and returned to via a "← Back to tour" button.

Which view renders is driven by ``st.session_state["view"]`` (default
``"tour"``). View switches use a ``_pending_view`` pending flag, mirroring the
``_pending_main_tab`` mechanism used for tab switches inside the workshop:
buttons that run after a widget is instantiated can't write the widget's own
key, so they write a pending flag the entry script applies before the next
render.

Inside the workshop, the active tab is bound to ``st.session_state["main_tabs"]``
via ``st.tabs(..., key="main_tabs", on_change="rerun")``. Tab-switch buttons
(Studio↔LLM handoffs) write ``_pending_main_tab``; the entry script applies it
before the widget re-renders. The standalone Workflow tab was superseded by the
Tour and is no longer dispatched (its page module is kept on disk so existing
tests still pass).

Tab labels live in :mod:`src.streamlit_app` as module-level constants so the
entry script and the page modules can never get out of sync.
"""

from __future__ import annotations

# Ensure the project root is on sys.path so ``import src.streamlit_app``
# works under Streamlit Cloud, where the runtime's CWD and ``sys.path``
# don't always match the script's directory.  Local ``streamlit run``
# already does this; the explicit insert is a no-op there.
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

try:
    from src.streamlit_app import TAB_LLM, TAB_STUDIO, TAB_WORKFLOW
except ImportError as _exc:
    # Streamlit Cloud's error overlay redacts the actual exception
    # message ("to prevent data leaks").  Re-raise with the *real*
    # error prepended so it shows up in the Streamlit Cloud logs
    # (which are unredacted).  Format: DIAG: <what we tried> ...
    # failed: <original error> at <path>.
    raise ImportError(
        f"DIAG: from src.streamlit_app import TAB_LLM, TAB_STUDIO, "
        f"TAB_WORKFLOW failed with: {_exc!r}. "
        f"sys.path[0:3]={sys.path[0:3]!r}, "
        f"_PROJECT_ROOT={str(_PROJECT_ROOT)!r}, "
        f"src exists={(_PROJECT_ROOT / 'src').is_dir()!r}, "
        f"src/streamlit_app/__init__.py exists="
        f"{(_PROJECT_ROOT / 'src' / 'streamlit_app' / '__init__.py').is_file()!r}"
    ) from _exc

# Page config is set ONCE here.  The page modules no longer call it.
st.set_page_config(
    page_title="Query Studio",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── View selection: Tour (default) or Workshop ──────────────────────────────
# Pending-flag pattern (see module docstring): view-switch buttons run after
# widgets are instantiated, so they write ``_pending_view`` and we apply it
# here, before anything renders.
_pending_view = st.session_state.pop("_pending_view", None)
if _pending_view in ("tour", "workshop"):
    st.session_state["view"] = _pending_view
if st.session_state.get("view") not in ("tour", "workshop"):
    st.session_state["view"] = "tour"

if st.session_state["view"] == "tour":
    from src.streamlit_app.pages import tour
    tour.render()
else:
    # ── Workshop: Studio + LLM SQL Assistant tabs ─────────────────────────
    _VALID_TABS = (TAB_STUDIO, TAB_LLM)

    # Apply a pending tab-switch request, then validate.  Tab-switch buttons
    # (the tour's escape hatch, the LLM tab's "Use … plan") write
    # ``_pending_main_tab`` because they can't write the ``main_tabs`` widget
    # key after it is instantiated.  A garbage/stale value resets to Studio.
    _pending_tab = st.session_state.pop("_pending_main_tab", None)
    if _pending_tab in _VALID_TABS:
        st.session_state["main_tabs"] = _pending_tab
    if st.session_state.get("main_tabs") not in _VALID_TABS:
        st.session_state["main_tabs"] = TAB_STUDIO

    if st.button("← Back to tour", key="wk_back_to_tour"):
        st.session_state["_pending_view"] = "tour"
        st.rerun()

    # key="main_tabs" + on_change="rerun" binds the active tab to session
    # state.  on_change="rerun" is REQUIRED: without it, st.tabs defaults to
    # on_change="ignore" and ignores writes to session_state["main_tabs"].
    tab_studio, tab_llm = st.tabs(
        [TAB_STUDIO, TAB_LLM],
        key="main_tabs",
        on_change="rerun",
    )

    # Body order: Studio first, LLM second.  Prefill keys (nl_prefill,
    # _llm_showcase_chip_prefill) are written by the tour before the rerun
    # into the workshop, so they are already in session_state and order does
    # not matter for the prefill trick.
    with tab_studio:
        from src.streamlit_app.pages import studio
        studio.render()

    with tab_llm:
        from src.streamlit_app.pages import llm_assistant
        llm_assistant.render()