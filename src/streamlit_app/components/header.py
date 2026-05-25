from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st

from src.config import load_config
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
        with st.popover("Open CSV", width='stretch'):
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
                width='stretch',
            ):
                _handle_demo_load()

    # LLM model selector
    with llm_col:
        with st.popover("⚙ LLM model", width='stretch'):
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


_PROVIDERS = {
    "ollama":        "Local Ollama",
    "ollama_remote": "Remote Ollama",
}


def _render_model_selector() -> None:
    from src.llm.natural_language import load_llm_config
    from src.streamlit_app.llm_health import clear_cache

    cfg = load_llm_config(load_config())

    st.markdown(
        '<div style="font-size:10.5px;font-weight:600;letter-spacing:.07em;'
        'text-transform:uppercase;color:#8E867B;margin-bottom:6px;">Provider</div>',
        unsafe_allow_html=True,
    )
    provider_keys = list(_PROVIDERS.keys())
    current = cfg.provider if cfg.provider in _PROVIDERS else "ollama"

    selected_provider = st.selectbox(
        "Provider",
        options=provider_keys,
        index=provider_keys.index(current),
        format_func=lambda k: _PROVIDERS[k],
        key="llm_provider_select",
        label_visibility="collapsed",
    )
    if selected_provider != current:
        _save_field_to_config("provider", selected_provider)
        st.session_state.pop("_cached_provider", None)  # invalidate ask-bar cache
        clear_cache()
        st.rerun()

    st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)

    _render_ollama_section(cfg, selected_provider)


# ── Per-provider sections ──────────────────────────────────────────────────

def _render_ollama_section(cfg, provider: str) -> None:
    from src.streamlit_app.llm_health import clear_cache, probe_ollama

    if provider == "ollama_remote":
        _label("Host URL")
        new_host = st.text_input(
            "Host", value=cfg.host, key="llm_remote_host",
            label_visibility="collapsed", placeholder="http://server-ip:11434",
        )
        if new_host and new_host != cfg.host:
            _save_field_to_config("host", new_host)
            clear_cache()
            st.rerun()

    probe = probe_ollama()
    models = list(probe.available_models)
    _label("Model")

    if not models:
        st.markdown(
            f'<div style="font-size:12px;color:#8E867B;margin-bottom:8px;">'
            f'Ollama not reachable at <code>{cfg.host}</code>.</div>',
            unsafe_allow_html=True,
        )
        if st.button("↺ Retry", key="llm_retry", width='stretch'):
            clear_cache()
            st.rerun()
        return

    if cfg.model not in models:
        models = [cfg.model] + models

    selected = st.selectbox(
        "Model", options=models, index=models.index(cfg.model),
        key="llm_model_select", label_visibility="collapsed",
    )
    if selected != cfg.model:
        _save_field_to_config("model", selected)
        clear_cache()
        st.rerun()

    st.markdown(
        f'<div style="font-size:11px;color:#8E867B;margin-top:4px;">'
        f'{len(models)} model{"s" if len(models) != 1 else ""} pulled</div>',
        unsafe_allow_html=True,
    )
    if st.button("↺ Refresh", key="llm_refresh", width='stretch'):
        clear_cache()
        st.rerun()


# ── Config helpers ─────────────────────────────────────────────────────────

def _label(text: str, *, top_margin: bool = False) -> None:
    margin = "margin-top:8px;" if top_margin else ""
    st.markdown(
        f'<div style="font-size:10.5px;font-weight:600;letter-spacing:.07em;'
        f'text-transform:uppercase;color:#8E867B;margin-bottom:4px;{margin}">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


def _save_field_to_config(field: str, value: str) -> None:
    """Write a single llm.* field to config.yaml."""
    from pathlib import Path
    import yaml
    path = Path("config.yaml")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            cfg = {}
        if "llm" not in cfg:
            cfg["llm"] = {}
        cfg["llm"][field] = value
        path.write_text(
            yaml.dump(cfg, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
    except Exception:
        pass


def _save_model_to_config(model: str) -> None:
    _save_field_to_config("model", model)
