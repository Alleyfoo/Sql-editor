"""Streamlit SQL editor — application package.

Tab-label constants are defined in :mod:`src.streamlit_app._tab_labels`
and re-exported here for backward compatibility with imports like
``from src.streamlit_app import TAB_STUDIO``.  See that module for the
rationale.
"""

from __future__ import annotations

from src.streamlit_app._tab_labels import TAB_LLM, TAB_STUDIO, TAB_WORKFLOW

__all__ = ["TAB_LLM", "TAB_STUDIO", "TAB_WORKFLOW"]
