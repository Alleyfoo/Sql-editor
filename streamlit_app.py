"""Query Studio — single-script app with three top-of-page tabs.

Replaces the previous ``st.navigation`` setup.  All three panels
(Studio, LLM SQL Assistant, Workflow) live in the same Streamlit
script and share ``st.session_state``.  Cross-tab handoff is a
``st.session_state["main_tabs"]`` write (the bound key on the
``st.tabs`` widget) — no ``st.switch_page`` is needed.

Visible tab order (left → right):
  Workflow · Studio · LLM SQL Assistant

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

# Default to Workflow on first visit — it's the guided entry point.
_VALID_TABS = (TAB_WORKFLOW, TAB_STUDIO, TAB_LLM)

# Apply a pending tab-switch request, then validate.  The tab-switch
# buttons (Workflow's "Open Studio/LLM tab", LLM's "Use … plan") run
# inside the ``with tab_*:`` blocks BELOW — i.e. AFTER this ``st.tabs``
# widget is instantiated.  Streamlit forbids modifying a widget's own
# session_state key after the widget is created, so those buttons can't
# write ``main_tabs`` directly.  They write a separate ``_pending_main_tab``
# flag instead, and we apply it here, before the widget renders (which is
# allowed).  A garbage value (e.g. a stale key from an old deploy) is
# reset to Workflow so ``st.tabs`` never sees an invalid label.
_pending_tab = st.session_state.pop("_pending_main_tab", None)
if _pending_tab in _VALID_TABS:
    st.session_state["main_tabs"] = _pending_tab
if st.session_state.get("main_tabs") not in _VALID_TABS:
    st.session_state["main_tabs"] = TAB_WORKFLOW

# Visible order: Workflow, Studio, LLM SQL Assistant.
# key="main_tabs" + on_change="rerun" binds the active tab to session state.
# on_change="rerun" is REQUIRED: without it, st.tabs defaults to
# on_change="ignore", does not track state, and ignores writes to
# session_state["main_tabs"] — so the buttons above can't switch tabs.
tab_workflow, tab_studio, tab_llm = st.tabs(
    [TAB_WORKFLOW, TAB_STUDIO, TAB_LLM],
    key="main_tabs",
    on_change="rerun",
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
