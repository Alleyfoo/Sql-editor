"""Bundled demo dataset for the Streamlit app.

This is a small, static CSV (`data/demo/sample_sales.csv`) that ships
with the repo so a visitor can explore Query Studio without uploading
their own file. It is loaded through the same ``src.ingestion.load_csv``
path as a user upload, so the read-only connection and executor
blocklist still apply.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.ingestion import load_csv


# Repo-relative location. ``parents[2]`` = ``<repo>/src/streamlit_app/..``
# -> ``<repo>``.
DEMO_PATH: Path = (
    Path(__file__).resolve().parents[2] / "data" / "demo" / "sample_sales.csv"
)
DEMO_NAME: str = "sample_sales.csv (demo)"
DEMO_DESCRIPTION: str = (
    "3 000 synthetic B2B orders, Jan 2023 – Mar 2025. "
    "Regions, categories, 309 customers, cost/margin, order status. "
    "Try: 'monthly revenue trend 2024', 'top 10 customers by margin', "
    "or 'compare EMEA vs AMER year-over-year'."
)


def load_demo() -> Tuple[object, Dict[str, str], pd.DataFrame, dict]:
    """Load the bundled demo CSV and return ``(conn, schema, df, meta)``.

    The returned tuple matches what ``components/header._handle_upload``
    needs to populate session state.
    """
    if not DEMO_PATH.exists():
        raise FileNotFoundError(
            f"Demo dataset missing: {DEMO_PATH}. "
            "Run `python data/demo/generate_sample_sales.py` to regenerate."
        )

    raw = DEMO_PATH.read_bytes()
    file_hash = hashlib.md5(raw).hexdigest()

    conn, schema = load_csv(DEMO_PATH)
    df = pd.read_csv(DEMO_PATH)

    meta = {
        "rows": len(df),
        "cols": len(df.columns),
        "size_bytes": len(raw),
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_hash": file_hash,
        "is_demo": True,
    }
    return conn, schema, df, meta


__all__ = ["DEMO_PATH", "DEMO_NAME", "DEMO_DESCRIPTION", "load_demo"]
