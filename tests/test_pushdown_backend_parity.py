from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from src.ingestion import load_csv
from src.mixed_execution.planner import LogicalPlanner
from src.mixed_execution.plan_validator import validate_logical_plan
from src.mixed_execution.pushdown import DataFramePushdownBackend, SQLPushdownBackend

REPO_ROOT = Path(__file__).resolve().parents[1]


def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)
    out = df.copy()
    cols = list(out.columns)
    day_like = [c for c in cols if c == "day" or c == "date" or c.endswith("_day")]
    if "day" not in cols and len(day_like) == 1:
        out = out.rename(columns={day_like[0]: "day"})
        cols = list(out.columns)
    for col in cols:
        if col == "day" or col == "date" or col.endswith("_day"):
            out[col] = out[col].astype(str).str.slice(0, 10)
    return out.sort_values(list(out.columns), na_position="last").reset_index(drop=True)


@pytest.mark.parametrize(
    ("dataset_rel", "question"),
    [
        (
            "data/open_data/seattle_weather.csv",
            "Show the 10 wettest days by precipitation with date and precipitation.",
        ),
        (
            "data/open_data/seattle_weather.csv",
            "Average temp_max by weather, highest first.",
        ),
        (
            "data/open_data/usgs_all_month.csv",
            "How many earthquakes per day for the last 7 days, ordered by date ascending.",
        ),
    ],
)
def test_pushdown_backend_parity(dataset_rel: str, question: str) -> None:
    dataset_path = REPO_ROOT / dataset_rel
    source_df = pd.read_csv(dataset_path)

    conn, schema = load_csv(dataset_path)
    try:
        planner = LogicalPlanner()
        plan = planner.plan(question, schema, source_name="data").plan
        validation = validate_logical_plan(plan, schema)
        assert validation.ok, validation.errors

        sql_backend = SQLPushdownBackend()
        df_backend = DataFramePushdownBackend()
        assert sql_backend.supports(plan)
        assert df_backend.supports(plan)

        sql_result = sql_backend.execute(plan, conn, source_rows=len(source_df)).dataframe
        df_result = df_backend.execute(plan, source_df, source_rows=len(source_df)).dataframe

        pdt.assert_frame_equal(
            _canonicalize(sql_result),
            _canonicalize(df_result),
            check_dtype=False,
            rtol=1e-6,
            atol=1e-6,
        )
    finally:
        conn.close()
