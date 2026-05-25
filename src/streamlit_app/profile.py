from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def compute_profile(file_hash: str, col: str, dtype: str, data_json: str) -> Dict[str, Any]:
    """Compute column stats once per CSV load. Keyed on file_hash."""
    import io
    series = pd.read_json(io.StringIO(data_json), typ="series")
    result: Dict[str, Any] = {
        "dtype": dtype,
        "count": int(series.count()),
        "null_count": int(series.isna().sum()),
        "total": len(series),
    }
    pct_complete = (result["count"] / result["total"] * 100) if result["total"] else 0
    result["pct_complete"] = round(pct_complete, 1)

    if dtype == "numeric":
        result["min"] = float(series.min()) if result["count"] else None
        result["max"] = float(series.max()) if result["count"] else None
        result["mean"] = float(series.mean()) if result["count"] else None
        # Mini histogram: 12 bins
        if result["count"] >= 2:
            counts, edges = pd.cut(series.dropna(), bins=12, retbins=True)
            result["hist_counts"] = counts.value_counts(sort=False).tolist()
        else:
            result["hist_counts"] = []
    elif dtype == "text":
        result["unique_count"] = int(series.nunique())
        if result["unique_count"] <= 12:
            result["top_values"] = series.value_counts().head(5).to_dict()
    elif dtype == "date":
        non_null = pd.to_datetime(series.dropna(), errors="coerce").dropna()
        if len(non_null):
            result["min_date"] = str(non_null.min().date())
            result["max_date"] = str(non_null.max().date())

    return result


def profile_from_df(df: pd.DataFrame, schema: Dict[str, str], file_hash: str) -> Dict[str, Dict]:
    profiles = {}
    for col, dtype in schema.items():
        try:
            profiles[col] = compute_profile(
                file_hash,
                col,
                dtype,
                df[col].to_json(),
            )
        except Exception:
            profiles[col] = {"dtype": dtype, "count": 0, "null_count": 0, "total": 0, "pct_complete": 0}
    return profiles
