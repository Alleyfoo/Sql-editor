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

from src.ingestion import load_csv, load_multiple_csvs


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


_DEMO_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "demo"

SUPPLY_CHAIN_TABLES = {
    "products":             _DEMO_DIR / "products.csv",
    "suppliers":            _DEMO_DIR / "suppliers.csv",
    "received_inventory":   _DEMO_DIR / "received_inventory.csv",
}

SUPPLY_CHAIN_NAME = "Supply Chain Demo (3 tables)"
SUPPLY_CHAIN_DESCRIPTION = (
    "27 products · 10 suppliers · 800 receiving records. "
    "Tables share product_id and supplier_id — load to explore JOIN queries."
)

# Pre-built 3-way JOIN that runs immediately after loading
SUPPLY_CHAIN_SHOWCASE_SQL = """\
SELECT
    p.product_name,
    p.category,
    s.supplier_name,
    s.country,
    s.rating,
    s.on_time_delivery_pct,
    COUNT(r.po_id)              AS deliveries,
    SUM(r.quantity_received)    AS total_units,
    ROUND(SUM(r.total_cost), 2) AS total_value
FROM received_inventory r
JOIN products  p ON r.product_id  = p.product_id
JOIN suppliers s ON r.supplier_id = s.supplier_id
               AND r.product_id   = s.product_id
GROUP BY
    p.product_name, p.category,
    s.supplier_name, s.country, s.rating, s.on_time_delivery_pct
ORDER BY total_value DESC
LIMIT 20"""


def load_supply_chain_demo():
    """Load all three supply-chain CSVs into one SQLite connection.

    Returns ``(conn, tables_schema, meta)``.
    ``tables_schema`` is ``{table_name: {col: type}}``.
    """
    for name, path in SUPPLY_CHAIN_TABLES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Supply chain demo file missing: {path}. "
                "Run `python data/demo/generate_supply_chain.py` to regenerate."
            )

    conn, tables_schema = load_multiple_csvs(SUPPLY_CHAIN_TABLES)

    total_rows = sum(
        len(pd.read_csv(p)) for p in SUPPLY_CHAIN_TABLES.values()
    )
    meta = {
        "rows": total_rows,
        "cols": sum(len(s) for s in tables_schema.values()),
        "tables": len(SUPPLY_CHAIN_TABLES),
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_demo": True,
        "is_multi_table": True,
    }
    return conn, tables_schema, meta


__all__ = [
    "DEMO_PATH", "DEMO_NAME", "DEMO_DESCRIPTION", "load_demo",
    "SUPPLY_CHAIN_TABLES", "SUPPLY_CHAIN_NAME", "SUPPLY_CHAIN_DESCRIPTION",
    "SUPPLY_CHAIN_SHOWCASE_SQL", "load_supply_chain_demo",
]
