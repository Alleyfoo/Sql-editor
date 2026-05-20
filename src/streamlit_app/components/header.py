from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st

from src.ingestion import load_csv
from src.streamlit_app import state
from src.streamlit_app.demo_dataset import (
    DEMO_DESCRIPTION,
    DEMO_NAME,
    load_demo,
)
from src.streamlit_app.llm_health import probe_ollama


def _llm_status_pill_html() -> str:
    """Return inline HTML for the LLM health pill in the topbar."""
    try:
        probe = probe_ollama()
    except Exception as exc:  # pragma: no cover — never crash the topbar
        tooltip = f"probe error: {exc}"
        return (
            '<span title="' + tooltip + '" '
            'style="display:inline-flex;align-items:center;gap:6px;'
            "padding:2px 9px;border-radius:999px;font-size:11.5px;"
            'background:#FAE7D0;color:#8A4A11;border:1px solid #E9C79A;">'
            '<span style="width:7px;height:7px;border-radius:50%;background:#C2410C;"></span>'
            "LLM: probe error</span>"
        )

    if probe.ok:
        tooltip = f"{probe.host} \u00b7 model {probe.model}"
        if probe.detail:
            tooltip += f" \u2014 {probe.detail}"
        bg, fg, border, dot = "#E1F0E2", "#3F6B45", "#C8DDC9", "#3F8A4F"
        label = f"LLM: connected \u00b7 {probe.model}"
    else:
        tooltip = f"{probe.host} \u00b7 {probe.detail or 'offline'}"
        bg, fg, border, dot = "#FAE7D0", "#8A4A11", "#E9C79A", "#C2410C"
        label = "LLM: offline"

    # Escape double quotes in tooltip just in case
    tooltip = tooltip.replace('"', "&quot;")
    return (
        f'<span title="{tooltip}" '
        f'style="display:inline-flex;align-items:center;gap:6px;'
        f"padding:2px 9px;border-radius:999px;font-size:11.5px;"
        f'background:{bg};color:{fg};border:1px solid {border};">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{dot};"></span>'
        f"{label}</span>"
    )


def render() -> None:
    dataset_name: Optional[str] = st.session_state.get("dataset_name")
    crumb_csv = f'<span class="chip-csv">{dataset_name}</span>' if dataset_name else ""
    crumb_query = '<span class="active">Untitled query</span>' if dataset_name else ""
    sep = '<span class="sep">/</span>' if dataset_name else ""

    topbar_html = f"""
    <div style="
        height:52px;display:flex;align-items:center;gap:14px;padding:0 22px;
        border-bottom:1px solid #E4DFD2;background:#FBFAF6;margin-bottom:12px;
    ">
      <div style="display:flex;align-items:center;gap:9px;font-weight:600;font-size:13px;">
        <div style="
            width:22px;height:22px;background:#1A1714;border-radius:5px;
            display:grid;place-items:center;color:#FBFAF6;position:relative;flex-shrink:0;
        ">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <rect x="2" y="2" width="7" height="7" rx="1" stroke="#FBFAF6" stroke-width="1.2" fill="none"/>
            <circle cx="9" cy="9" r="1.5" fill="#C2410C"/>
          </svg>
        </div>
        Query Studio
        <span style="color:#8E867B;font-weight:400;margin-left:2px;font-size:12px;">alpha</span>
      </div>
      <div style="
          display:flex;align-items:center;gap:8px;color:#57514A;font-size:12.5px;
          padding-left:12px;border-left:1px solid #E4DFD2;margin-left:4px;
      ">
        workspace {sep} {crumb_csv} {sep} {crumb_query}
      </div>
      <div style="flex:1;"></div>
      {_llm_status_pill_html()}
      <span style="font-size:12px;color:#8E867B;margin-left:10px;">
        {datetime.now(timezone.utc).strftime("%H:%M UTC")}
      </span>
    </div>
    """
    st.markdown(topbar_html, unsafe_allow_html=True)

    csv_col, llm_col = st.columns([5, 3], gap="small")

    # File upload in a popover
    with csv_col:
        with st.popover("Open CSV", use_container_width=True):
            uploaded = st.file_uploader(
                "Choose a CSV file",
                type=["csv"],
                key="csv_uploader",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                _handle_upload(uploaded)

            st.markdown(
                '<div style="margin-top:10px;padding-top:10px;'
                "border-top:1px solid #E4DFD2;font-size:11px;"
                "font-weight:600;letter-spacing:.07em;text-transform:uppercase;"
                'color:#8E867B;">Or try the demo</div>',
                unsafe_allow_html=True,
            )
            st.caption(DEMO_DESCRIPTION)
            if st.button(
                "Load demo dataset",
                key="load_demo_btn",
                use_container_width=True,
            ):
                _handle_demo_load()

    # LLM model selector
    with llm_col:
        with st.popover("⚙ LLM model", use_container_width=True):
            _render_model_selector()


def _handle_demo_load() -> None:
    try:
        conn, schema, df, meta = load_demo()
    except Exception as exc:
        st.error(f"Failed to load demo dataset: {exc}")
        return

    ss = st.session_state
    if ss.get("conn") is not None:
        try:
            ss.conn.close()
        except Exception:
            pass

    ss.conn = conn
    ss.schema = schema
    ss.dataset_name = DEMO_NAME
    ss.dataset_meta = meta
    ss.dataset_df = df
    ss.pop("_col_profiles", None)
    state.reset_query()
    st.rerun()


def _handle_upload(uploaded) -> None:
    import hashlib
    import tempfile

    raw = uploaded.read()
    file_hash = hashlib.md5(raw).hexdigest()

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        conn, schema = load_csv(tmp_path)
    except Exception as exc:
        st.error(f"Failed to load CSV: {exc}")
        return

    import pandas as pd

    df = pd.read_csv(tmp_path)

    ss = st.session_state
    if ss.get("conn") is not None:
        try:
            ss.conn.close()
        except Exception:
            pass

    ss.conn = conn
    ss.schema = schema
    ss.dataset_name = uploaded.name
    ss.dataset_meta = {
        "rows": len(df),
        "cols": len(df.columns),
        "size_bytes": len(raw),
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_hash": file_hash,
    }
    ss.dataset_df = df
    state.reset_query()
    st.rerun()


def _render_model_selector() -> None:
    from src.streamlit_app.llm_health import clear_cache, probe_ollama

    probe = probe_ollama()
    configured = probe.model
    models = list(probe.available_models)

    st.markdown(
        '<div style="font-size:10.5px;font-weight:600;letter-spacing:.07em;'
        'text-transform:uppercase;color:#8E867B;margin-bottom:8px;">Ollama model</div>',
        unsafe_allow_html=True,
    )

    if not models:
        st.markdown(
            f'<div style="font-size:12px;color:#8E867B;margin-bottom:8px;">'
            f'Ollama offline.<br>Configured: <code>{configured}</code></div>',
            unsafe_allow_html=True,
        )
        if st.button("↺ Retry connection", key="llm_retry", use_container_width=True):
            clear_cache()
            st.rerun()
        return

    if configured not in models:
        models = [configured] + models

    selected = st.selectbox(
        "Model",
        options=models,
        index=models.index(configured),
        key="llm_model_select",
        label_visibility="collapsed",
    )

    if selected != configured:
        _save_model_to_config(selected)
        clear_cache()
        st.rerun()

    st.markdown(
        f'<div style="font-size:11px;color:#8E867B;margin-top:6px;">'
        f'{len(models)} model{"s" if len(models) != 1 else ""} available</div>',
        unsafe_allow_html=True,
    )
    if st.button("↺ Refresh list", key="llm_refresh", use_container_width=True):
        clear_cache()
        st.rerun()


def _save_model_to_config(model: str) -> None:
    from pathlib import Path
    import yaml

    path = Path("config.yaml")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            cfg = {}
        if "llm" not in cfg:
            cfg["llm"] = {}
        cfg["llm"]["model"] = model
        path.write_text(
            yaml.dump(cfg, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
    except Exception:
        pass
