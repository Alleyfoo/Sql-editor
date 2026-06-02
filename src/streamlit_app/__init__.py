"""Tab label constants — single source of truth for cross-tab handoff.

The entry script (``streamlit_app.py``) and the page modules
(``pages.studio``, ``pages.llm_assistant``, ``pages.workflow``) all
import these constants so the tabs widget, the handoff buttons, and
the Workflow tour never get out of sync over a typo.
"""

from __future__ import annotations

TAB_STUDIO = "Studio"
TAB_LLM = "LLM SQL Assistant"
TAB_WORKFLOW = "Workflow"

__all__ = ["TAB_STUDIO", "TAB_LLM", "TAB_WORKFLOW"]
