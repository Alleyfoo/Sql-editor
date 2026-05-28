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
    SUPPLY_CHAIN_DESCRIPTION,
    SUPPLY_CHAIN_NAME,
    SUPPLY_CHAIN_SHOWCASE_SQL,
    load_demo,
    load_supply_chain_demo,
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

    # Compact header: brand + crumbs + popovers in one 44px row
    head_cols = st.columns([0.22, 0.5, 0.14, 0.14])
    
    with head_cols[0]:
        st.markdown(
            '<div class="brand-strip">'
            '<span class="brand-mark">▣</span>'
            '<span class="brand-name">Query Studio</span>'
            '<span class="brand-tag">alpha</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    
    with head_cols[1]:
        ds_label = dataset_name or "no dataset"
        st.markdown(
            f'<div class="crumbs">'
            f'workspace <span class="sep">›</span> '
            f'<span>{ds_label}</span> '
            f'<span class="sep">›</span> Untitled query'
            f'</div>',
            unsafe_allow_html=True,
        )
    
    with head_cols[2]:
        with st.popover("📂 Open CSV", use_container_width=True):
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

            st.markdown(
                '<div style="margin-top:10px;padding-top:10px;'
                "border-top:1px solid #E4DFD2;font-size:11px;"
                "font-weight:600;letter-spacing:.07em;text-transform:uppercase;"
                'color:#8E867B;">Or try the JOIN demo</div>',
                unsafe_allow_html=True,
            )
            st.caption(SUPPLY_CHAIN_DESCRIPTION)
            if st.button(
                "Load supply chain demo",
                key="load_supply_chain_btn",
                width='stretch',
            ):
                _handle_supply_chain_load()
    
    with head_cols[3]:
        with st.popover("⚙ LLM", use_container_width=True):
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
    ss.tables = {}
    ss.dataset_name = DEMO_NAME
    ss.dataset_meta = meta
    ss.dataset_df = df
    ss.pop("_col_profiles", None)
    state.reset_query()
    st.rerun()


def _handle_supply_chain_load() -> None:
    try:
        conn, tables_schema, meta = load_supply_chain_demo()
    except Exception as exc:
        st.error(f"Failed to load supply chain demo: {exc}")
        return

    ss = st.session_state
    if ss.get("conn") is not None:
        try:
            ss.conn.close()
        except Exception:
            pass

    # Detect relationships between tables
    from src.relationships import detect_relationships
    relationships = detect_relationships(tables_schema)

    # Flatten tables_schema for widgets that expect a single schema dict
    flat_schema = {col: dtype for s in tables_schema.values() for col, dtype in s.items()}

    ss.conn = conn
    ss.schema = flat_schema
    ss.tables = tables_schema          # full per-table schema for sidebar + JOIN composer
    ss.relationships = relationships   # detected FK relationships
    ss.dataset_name = SUPPLY_CHAIN_NAME
    ss.dataset_meta = meta
    ss.dataset_df = None               # no single dataframe for multi-table
    ss.pop("_col_profiles", None)
    ss.last_sql = SUPPLY_CHAIN_SHOWCASE_SQL   # pre-load the showcase JOIN query
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
    "gemini":        "Gemini (cloud)",
    "groq":          "Groq (cloud)",
}

_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]


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

    if selected_provider == "gemini":
        _render_cloud_llm_section(cfg, "gemini", _GEMINI_MODELS)
    elif selected_provider == "groq":
        _render_cloud_llm_section(cfg, "groq", _GROQ_MODELS)
    else:
        _render_ollama_section(cfg, selected_provider)


# ── Per-provider sections ──────────────────────────────────────────────────

_CLOUD_KEY_PLACEHOLDER = {
    "gemini": "AIza…",
    "groq":   "gsk_…",
}

_CLOUD_KEY_HELP = {
    "gemini": "Get a free key at aistudio.google.com/apikey",
    "groq":   "Get a free key at console.groq.com",
}


def _render_cloud_llm_section(cfg, provider: str, model_list: list) -> None:
    from src.streamlit_app.llm_health import clear_cache, probe_ollama

    _label("API Key")
    current_key = cfg.api_key or ""
    new_key = st.text_input(
        "API Key",
        value=current_key,
        type="password",
        key=f"{provider}_api_key_input",
        label_visibility="collapsed",
        placeholder=_CLOUD_KEY_PLACEHOLDER.get(provider, "key…"),
    )
    st.markdown(
        f'<div style="font-size:10px;color:#8E867B;margin-bottom:4px;">'
        f'{_CLOUD_KEY_HELP.get(provider, "")}</div>',
        unsafe_allow_html=True,
    )
    if new_key != current_key:
        # API keys are session-only — never written to disk
        overrides: dict = st.session_state.get("_llm_overrides", {})
        overrides["api_key"] = new_key
        st.session_state["_llm_overrides"] = overrides
        clear_cache()
        st.rerun()

    _label("Model", top_margin=True)
    current_model = cfg.model if cfg.model in model_list else model_list[0]
    selected = st.selectbox(
        "Model",
        options=model_list,
        index=model_list.index(current_model),
        key=f"{provider}_model_select",
        label_visibility="collapsed",
    )
    if selected != cfg.model:
        _save_field_to_config("model", selected)
        clear_cache()
        st.rerun()

    probe = probe_ollama()
    if probe.ok:
        st.markdown(
            '<div style="font-size:11px;color:#3F6B45;margin-top:6px;">✓ Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        detail_html = (probe.detail or "not connected").replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<div style="font-size:11px;color:#8A4A11;margin-top:6px;'
            f'word-break:break-word;">✗ {detail_html}</div>',
            unsafe_allow_html=True,
        )
    if st.button("↺ Test connection", key=f"{provider}_refresh", width="stretch"):
        clear_cache()
        st.rerun()

    st.markdown(
        '<div style="font-size:10px;color:#8E867B;margin-top:8px;line-height:1.4;">'
        'Key is session-only. For persistence add it to Streamlit Cloud secrets.</div>',
        unsafe_allow_html=True,
    )


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
    """Persist a single llm.* field.

    Always writes to ``st.session_state["_llm_overrides"]`` so the change
    survives on Streamlit Community Cloud (where the filesystem is ephemeral).
    Also attempts to write to ``config.yaml`` for local development.
    """
    # Runtime override — works everywhere, including Streamlit Cloud
    overrides: dict = st.session_state.get("_llm_overrides", {})
    overrides[field] = value
    st.session_state["_llm_overrides"] = overrides

    # Best-effort local config.yaml write (skipped silently on Cloud)
    import yaml
    from src.config import DEFAULT_CONFIG_PATH
    path = DEFAULT_CONFIG_PATH
    try:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        cfg = yaml.safe_load(existing) or {}
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
