"""Backwards-compatible re-export for the old single-page app entry point.

The body moved to :mod:`src.streamlit_app.pages.studio` so the multipage
app can register it under ``st.navigation``.  This shim keeps the
historical ``from src.streamlit_app.app import run`` import path working.
"""

from .studio import render as run

__all__ = ["run"]
