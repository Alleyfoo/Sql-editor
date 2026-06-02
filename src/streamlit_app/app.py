"""Backwards-compatible shim for the old single-page app entry point.

The page body now lives in :mod:`src.streamlit_app.pages.studio` so the
multipage app can register it under ``st.navigation``.  This shim keeps
the historical ``from src.streamlit_app.app import run`` import path
working (used by ``streamlit_app.py``).
"""

from src.streamlit_app.pages.studio import render as run

__all__ = ["run"]
