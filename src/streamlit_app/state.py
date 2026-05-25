from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from src.ingestion import TABLE_NAME
from src.query_model import QueryModel


def init() -> None:
    ss = st.session_state
    ss.setdefault("conn", None)
    ss.setdefault("schema", {})
    ss.setdefault("dataset_name", None)
    ss.setdefault("dataset_meta", {})
    ss.setdefault("model", QueryModel(table=TABLE_NAME))
    ss.setdefault("results_df", None)
    ss.setdefault("last_sql", "")
    ss.setdefault("last_exec_ms", None)
    ss.setdefault("nl_history", [])
    ss.setdefault("transcript", [])
    ss.setdefault("nl_status", "")
    ss.setdefault("nl_prefill", "")
    ss.setdefault("composer_open", {
        "select": True,
        "where": True,
        "group": False,
        "having": False,
        "order": True,
    })
    # Filter/sort row state for the composer (list of dicts)
    ss.setdefault("where_rows", [])
    ss.setdefault("having_rows", [])
    ss.setdefault("order_rows", [])
    ss.setdefault("agg_rows", [])


def reset_query() -> None:
    ss = st.session_state
    ss.model = QueryModel(table=TABLE_NAME)
    ss.results_df = None
    ss.last_sql = ""
    ss.last_exec_ms = None
    ss.where_rows = []
    ss.having_rows = []
    ss.order_rows = []
    ss.agg_rows = []


def append_transcript(entry: Dict[str, Any]) -> None:
    st.session_state.transcript.append(entry)


def push_nl_history(question: str, reply: str) -> None:
    hist = st.session_state.nl_history
    hist.append((question, reply))
    # Trim to configured depth (default 6)
    st.session_state.nl_history = hist[-6:]
