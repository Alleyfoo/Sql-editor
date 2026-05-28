from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from src.ingestion import TABLE_NAME
from src.query_model import QueryModel


def init() -> None:
    ss = st.session_state
    ss.setdefault("conn", None)
    ss.setdefault("schema", {})
    ss.setdefault("tables", {})   # {table_name: schema} for multi-table datasets
    ss.setdefault("relationships", [])  # detected FK relationships for multi-table
    ss.setdefault("dataset_name", None)
    ss.setdefault("dataset_meta", {})
    ss.setdefault("model", QueryModel(table=TABLE_NAME))
    ss.setdefault("results_df", None)
    ss.setdefault("last_sql", "")
    ss.setdefault("last_exec_ms", None)
    ss.setdefault("nl_history", [])
    ss.setdefault("nl_history_mt", [])  # multi-table: [(question, sql), ...]
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
    ss.setdefault("join_rows", [])  # Phase 5b: JOIN configuration


def reset_query() -> None:
    ss = st.session_state
    # For multi-table datasets keep the primary table name; fall back to TABLE_NAME
    primary = next(iter(ss.get("tables", {})), TABLE_NAME)
    ss.model = QueryModel(table=primary)
    ss.results_df = None
    ss.last_sql = ""
    ss.last_exec_ms = None
    ss.where_rows = []
    ss.having_rows = []
    ss.order_rows = []
    ss.agg_rows = []
    ss.join_rows = []  # Phase 5b: clear JOIN configurations
    ss.nl_history_mt = []
    ss.pop("_raw_sql_lock", None)


def append_transcript(entry: Dict[str, Any]) -> None:
    st.session_state.transcript.append(entry)


def push_nl_history(question: str, reply: str) -> None:
    hist = st.session_state.nl_history
    hist.append((question, reply))
    # Trim to configured depth (default 6)
    st.session_state.nl_history = hist[-6:]


def push_nl_history_mt(question: str, sql: str) -> None:
    """Store a (question, sql) pair for multi-table conversation context."""
    hist = st.session_state.nl_history_mt
    hist.append((question, sql))
    st.session_state.nl_history_mt = hist[-6:]
