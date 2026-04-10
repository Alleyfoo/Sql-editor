"""Open-data benchmark for SQL-fit vs Python-fit analytics tasks.

The goal is pragmatic scoping: measure where the NL->SQL pipeline is
reliable and where tasks drift into Python-style analytics (percentiles,
rolling windows, etc.).

Usage:
    python eval/open_data_sql_vs_python_eval.py --provider ollama --model gemma4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# Ensure "python eval/open_data_sql_vs_python_eval.py" can import src/* modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.executor import execute
from src.ingestion import load_csv
from src.llm.natural_language import (
    LLMConfig,
    OllamaClient,
    RouteToPythonError,
    nl_to_query_model,
)


DEFAULT_CASES = REPO_ROOT / "eval" / "golden" / "open_data" / "sql_vs_python_cases.json"
DEFAULT_REPORT_DIR = REPO_ROOT / "eval" / "reports"


@dataclass
class ProbeCase:
    id: str
    track: str
    dataset: str
    question: str
    validator: str

    @staticmethod
    def from_dict(payload: Dict[str, Any], index: int) -> "ProbeCase":
        if not isinstance(payload, dict):
            raise ValueError(f"cases[{index}] must be an object")

        cid = payload.get("id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"cases[{index}].id must be a non-empty string")

        track = payload.get("track")
        if track not in {"sql_fit", "python_fit"}:
            raise ValueError(f"cases[{index}].track must be sql_fit|python_fit")

        dataset = payload.get("dataset")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError(f"cases[{index}].dataset must be a non-empty string")

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"cases[{index}].question must be a non-empty string")

        validator = payload.get("validator")
        if not isinstance(validator, str) or not validator.strip():
            raise ValueError(f"cases[{index}].validator must be a non-empty string")

        return ProbeCase(
            id=cid.strip(),
            track=track,
            dataset=dataset.strip(),
            question=question.strip(),
            validator=validator.strip(),
        )


def load_cases(path: Path) -> List[ProbeCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("cases file must be a JSON array")
    return [ProbeCase.from_dict(item, i) for i, item in enumerate(raw)]


def _preview_rows(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    return df.head(n).where(pd.notna(df.head(n)), None).to_dict(orient="records")


def _guess_date_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return str(col)
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        sample = series.astype(str).head(20)
        if sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() >= 0.8:
            return str(col)
    return None


def _guess_numeric_col(df: pd.DataFrame, exclude: Optional[set[str]] = None) -> Optional[str]:
    excluded = exclude or set()
    for col in df.columns:
        if str(col) in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return str(col)
    for col in df.columns:
        if str(col) in excluded:
            continue
        parsed = pd.to_numeric(df[col], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return str(col)
    return None


def _validate_usgs_top10_strongest(result: pd.DataFrame, source: pd.DataFrame) -> Tuple[bool, str]:
    if len(result) != 10:
        return False, f"expected 10 rows, got {len(result)}"
    if "mag" not in result.columns:
        return False, "missing mag column in result"
    mags = pd.to_numeric(result["mag"], errors="coerce")
    if mags.isna().any():
        return False, "result mag column contains non-numeric values"
    if not bool((mags.diff().fillna(0) <= 0).all()):
        return False, "result mag values are not sorted descending"
    expected_max = float(pd.to_numeric(source["mag"], errors="coerce").max())
    actual_max = float(mags.iloc[0])
    if not math.isclose(actual_max, expected_max, rel_tol=0.0, abs_tol=1e-9):
        return False, f"first magnitude mismatch: got {actual_max}, expected {expected_max}"
    return True, "top-10 strongest magnitude check passed"


def _validate_usgs_avg_magtype_top10(result: pd.DataFrame, source: pd.DataFrame) -> Tuple[bool, str]:
    if result.empty:
        return False, "result is empty"
    if "magType" not in result.columns:
        return False, "missing magType column"
    numeric_col = _guess_numeric_col(result, exclude={"magType"})
    if numeric_col is None:
        return False, "missing numeric aggregation column"
    frame = source.copy()
    frame["mag"] = pd.to_numeric(frame["mag"], errors="coerce")
    expected = (
        frame.dropna(subset=["magType", "mag"])
        .groupby("magType", dropna=True)["mag"]
        .mean()
        .sort_values(ascending=False)
    )
    if expected.empty:
        return False, "source dataset has no valid magType/mag rows"
    expected_type = str(expected.index[0])
    expected_val = float(expected.iloc[0])
    actual_type = str(result.iloc[0]["magType"])
    actual_val = float(pd.to_numeric(pd.Series([result.iloc[0][numeric_col]]), errors="coerce").iloc[0])
    if actual_type != expected_type:
        return False, f"top magType mismatch: got {actual_type}, expected {expected_type}"
    if not math.isclose(actual_val, expected_val, rel_tol=0.0, abs_tol=1e-6):
        return False, f"top average mismatch: got {actual_val}, expected {expected_val}"
    return True, "average magnitude by magType check passed"


def _expected_usgs_last7_counts(source: pd.DataFrame) -> Dict[str, int]:
    parsed = pd.to_datetime(source["time"], errors="coerce", utc=True)
    dates = parsed.dt.strftime("%Y-%m-%d")
    valid = dates.dropna()
    if valid.empty:
        return {}
    max_day = datetime.strptime(str(valid.max()), "%Y-%m-%d").date()
    start_day = max_day - timedelta(days=6)
    frame = pd.DataFrame({"day": dates})
    frame = frame.dropna()
    frame["day_date"] = pd.to_datetime(frame["day"], errors="coerce").dt.date
    window = frame[(frame["day_date"] >= start_day) & (frame["day_date"] <= max_day)]
    counts = window.groupby("day", dropna=True).size().sort_index()
    return {str(day): int(val) for day, val in counts.items()}


def _validate_usgs_count_per_day_last7(result: pd.DataFrame, source: pd.DataFrame) -> Tuple[bool, str]:
    if result.empty:
        return False, "result is empty"
    day_col = _guess_date_col(result)
    if day_col is None:
        return False, "could not identify day column in result"
    count_col = _guess_numeric_col(result, exclude={day_col})
    if count_col is None:
        return False, "could not identify count column in result"
    actual_days = result[day_col].astype(str).str.slice(0, 10)
    actual_counts = pd.to_numeric(result[count_col], errors="coerce")
    if actual_counts.isna().any():
        return False, "count column has non-numeric values"
    actual = {str(k): int(v) for k, v in zip(actual_days, actual_counts)}
    expected = _expected_usgs_last7_counts(source)
    if actual != expected:
        return False, "daily count mapping does not match expected last-7-days counts"
    return True, "last-7-days daily count check passed"


def _validate_usgs_p90_magnitude(result: pd.DataFrame, source: pd.DataFrame) -> Tuple[bool, str]:
    if len(result) != 1:
        return False, f"expected scalar row, got {len(result)} rows"
    numeric_col = _guess_numeric_col(result)
    if numeric_col is None:
        return False, "no numeric scalar returned"
    actual = float(pd.to_numeric(result[numeric_col], errors="coerce").iloc[0])
    expected = float(pd.to_numeric(source["mag"], errors="coerce").dropna().quantile(0.9))
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=0.02):
        return False, f"p90 mismatch: got {actual}, expected {expected}"
    return True, "p90 magnitude check passed"


def _expected_usgs_rolling7_counts(source: pd.DataFrame) -> Dict[str, float]:
    parsed = pd.to_datetime(source["time"], errors="coerce", utc=True)
    days = parsed.dt.strftime("%Y-%m-%d")
    valid = days.dropna()
    if valid.empty:
        return {}
    max_day = datetime.strptime(str(valid.max()), "%Y-%m-%d").date()
    start_day = max_day - timedelta(days=13)
    frame = pd.DataFrame({"day": days}).dropna()
    frame["day_date"] = pd.to_datetime(frame["day"], errors="coerce").dt.date
    window = frame[(frame["day_date"] >= start_day) & (frame["day_date"] <= max_day)]
    counts = window.groupby("day", dropna=True).size().sort_index()
    rolling = counts.rolling(window=7, min_periods=1).mean()
    return {str(k): float(v) for k, v in rolling.items()}


def _validate_usgs_rolling7_daily_counts(result: pd.DataFrame, source: pd.DataFrame) -> Tuple[bool, str]:
    if result.empty:
        return False, "result is empty"
    day_col = _guess_date_col(result)
    if day_col is None:
        return False, "could not identify day column in result"
    metric_col = _guess_numeric_col(result, exclude={day_col})
    if metric_col is None:
        return False, "could not identify numeric metric column in result"

    actual = (
        pd.DataFrame(
            {
                "day": result[day_col].astype(str).str.slice(0, 10),
                "metric": pd.to_numeric(result[metric_col], errors="coerce"),
            }
        )
        .dropna()
        .set_index("day")["metric"]
        .sort_index()
    )
    expected = pd.Series(_expected_usgs_rolling7_counts(source), dtype="float64").sort_index()
    overlap = actual.index.intersection(expected.index)
    if len(overlap) < 5:
        return False, "insufficient day overlap with expected 14-day rolling window"
    mae = float((actual.loc[overlap] - expected.loc[overlap]).abs().mean())
    if mae > 0.05:
        return False, f"rolling average mismatch (MAE={mae:.4f})"
    return True, "7-day rolling average check passed"


def _validate_seattle_avg_tempmax_by_weather(
    result: pd.DataFrame, source: pd.DataFrame
) -> Tuple[bool, str]:
    if result.empty:
        return False, "result is empty"
    if "weather" not in result.columns:
        return False, "missing weather column"
    metric_col = _guess_numeric_col(result, exclude={"weather"})
    if metric_col is None:
        return False, "missing numeric aggregation column"

    expected = source.groupby("weather", dropna=True)["temp_max"].mean().sort_values(ascending=False)
    if expected.empty:
        return False, "source dataset has no temp_max/weather values"
    expected_weather = str(expected.index[0])
    expected_val = float(expected.iloc[0])
    actual_weather = str(result.iloc[0]["weather"])
    actual_val = float(pd.to_numeric(pd.Series([result.iloc[0][metric_col]]), errors="coerce").iloc[0])
    if actual_weather != expected_weather:
        return False, f"top weather mismatch: got {actual_weather}, expected {expected_weather}"
    if not math.isclose(actual_val, expected_val, rel_tol=0.0, abs_tol=1e-6):
        return False, f"top average temp mismatch: got {actual_val}, expected {expected_val}"
    return True, "average temp_max by weather check passed"


def _validate_seattle_top10_wettest(result: pd.DataFrame, source: pd.DataFrame) -> Tuple[bool, str]:
    if len(result) != 10:
        return False, f"expected 10 rows, got {len(result)}"
    if "precipitation" not in result.columns:
        return False, "missing precipitation column"
    actual = pd.to_numeric(result["precipitation"], errors="coerce")
    if actual.isna().any():
        return False, "precipitation column is not numeric"
    expected_max = float(pd.to_numeric(source["precipitation"], errors="coerce").max())
    actual_max = float(actual.iloc[0])
    if not math.isclose(actual_max, expected_max, rel_tol=0.0, abs_tol=1e-9):
        return False, f"max precipitation mismatch: got {actual_max}, expected {expected_max}"
    if not bool((actual.diff().fillna(0) <= 0).all()):
        return False, "precipitation values are not sorted descending"
    return True, "top wettest days check passed"


def _validate_seattle_rain_days_per_month(result: pd.DataFrame, source: pd.DataFrame) -> Tuple[bool, str]:
    if result.empty:
        return False, "result is empty"
    date_col = _guess_date_col(result)
    if date_col is None:
        return False, "missing month/date column in result"
    metric_col = _guess_numeric_col(result, exclude={date_col})
    if metric_col is None:
        return False, "missing count column in result"

    actual = (
        pd.DataFrame(
            {
                "month": result[date_col].astype(str).str.slice(0, 7),
                "count": pd.to_numeric(result[metric_col], errors="coerce"),
            }
        )
        .dropna()
        .groupby("month", dropna=True)["count"]
        .sum()
        .sort_index()
    )

    expected = (
        source[source["weather"].astype(str).str.lower() == "rain"]
        .assign(month=source["date"].astype(str).str.slice(0, 7))
        .groupby("month", dropna=True)
        .size()
        .sort_index()
    )
    if actual.to_dict() != expected.to_dict():
        return False, "rain-day monthly counts mismatch"
    return True, "rain-day monthly count check passed"


def _validate_seattle_rolling30_precip(result: pd.DataFrame, source: pd.DataFrame) -> Tuple[bool, str]:
    if result.empty:
        return False, "result is empty"
    date_col = _guess_date_col(result)
    if date_col is None:
        return False, "missing date column in result"
    metric_col = _guess_numeric_col(result, exclude={date_col})
    if metric_col is None:
        return False, "missing numeric metric column"

    ordered = source.assign(date=pd.to_datetime(source["date"], errors="coerce")).sort_values("date")
    ordered["roll30"] = pd.to_numeric(ordered["precipitation"], errors="coerce").rolling(30, min_periods=1).mean()
    expected_frame = ordered.dropna(subset=["date"]).tail(90).copy()
    expected_frame["day"] = expected_frame["date"].dt.strftime("%Y-%m-%d")
    expected = expected_frame.set_index("day")["roll30"]

    actual = (
        pd.DataFrame(
            {
                "day": result[date_col].astype(str).str.slice(0, 10),
                "metric": pd.to_numeric(result[metric_col], errors="coerce"),
            }
        )
        .dropna()
        .set_index("day")["metric"]
        .sort_index()
    )
    overlap = actual.index.intersection(expected.index)
    if len(overlap) < 20:
        return False, "insufficient overlap for 30-day rolling validation"
    mae = float((actual.loc[overlap] - expected.loc[overlap]).abs().mean())
    if mae > 0.05:
        return False, f"rolling 30-day precipitation mismatch (MAE={mae:.4f})"
    return True, "30-day rolling precipitation check passed"


def _validate_non_empty_result(result: pd.DataFrame, source: pd.DataFrame) -> Tuple[bool, str]:
    _ = source
    if result.empty:
        return False, "result is empty"
    return True, "result is non-empty"


def _validate_single_numeric_scalar(result: pd.DataFrame, source: pd.DataFrame) -> Tuple[bool, str]:
    _ = source
    if len(result) != 1:
        return False, f"expected 1 row, got {len(result)}"
    numeric_col = _guess_numeric_col(result)
    if numeric_col is None:
        return False, "no numeric scalar column found"
    value = pd.to_numeric(result[numeric_col], errors="coerce").iloc[0]
    if pd.isna(value):
        return False, "numeric scalar is NaN"
    return True, "single numeric scalar check passed"


def _find_column_by_tokens(df: pd.DataFrame, tokens: List[str]) -> Optional[str]:
    lowered = [t.lower() for t in tokens]
    for col in df.columns:
        name = str(col).lower()
        if all(token in name for token in lowered):
            return str(col)
    return None


def _find_numeric_metric_col(
    df: pd.DataFrame, exclude: set[str], preferred_tokens: Optional[List[str]] = None
) -> Optional[str]:
    preferred = [t.lower() for t in (preferred_tokens or [])]
    for col in df.columns:
        name = str(col)
        if name in exclude:
            continue
        if preferred and not any(tok in name.lower() for tok in preferred):
            continue
        parsed = pd.to_numeric(df[name], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return name
    return _guess_numeric_col(df, exclude=exclude)


def _validate_hsy_top5_busiest_departure(result: pd.DataFrame, source: pd.DataFrame) -> Tuple[bool, str]:
    dep_id_col = _find_column_by_tokens(result, ["departure", "station", "id"])
    dep_name_col = _find_column_by_tokens(result, ["departure", "station", "name"])
    if dep_id_col is None or dep_name_col is None:
        return False, "result missing departure station id/name columns"
    count_col = _find_numeric_metric_col(
        result,
        exclude={dep_id_col, dep_name_col},
        preferred_tokens=["count", "trip"],
    )
    if count_col is None:
        return False, "result missing trip count column"
    if len(result) < 5:
        return False, f"expected at least 5 rows, got {len(result)}"

    expected = (
        source.groupby(["Departure station id", "Departure station name"], dropna=False)
        .size()
        .reset_index(name="trip_count")
        .sort_values(["trip_count", "Departure station id"], ascending=[False, True])
        .head(5)
        .reset_index(drop=True)
    )
    actual = result.head(5).copy()
    actual["dep_id"] = pd.to_numeric(actual[dep_id_col], errors="coerce")
    actual["dep_name"] = actual[dep_name_col].astype(str)
    actual["trip_count"] = pd.to_numeric(actual[count_col], errors="coerce")
    if actual[["dep_id", "trip_count"]].isna().any().any():
        return False, "result contains non-numeric station id or trip count"

    exp_rows = [
        (int(row["Departure station id"]), str(row["Departure station name"]), int(row["trip_count"]))
        for _, row in expected.iterrows()
    ]
    act_rows = [
        (int(row["dep_id"]), str(row["dep_name"]), int(row["trip_count"]))
        for _, row in actual.iterrows()
    ]
    if exp_rows != act_rows:
        return False, "top-5 busiest departure stations mismatch"
    return True, "top-5 busiest departure stations check passed"


def _validate_hsy_top10_common_routes(result: pd.DataFrame, source: pd.DataFrame) -> Tuple[bool, str]:
    dep_id_col = _find_column_by_tokens(result, ["departure", "station", "id"])
    dep_name_col = _find_column_by_tokens(result, ["departure", "station", "name"])
    ret_id_col = _find_column_by_tokens(result, ["return", "station", "id"])
    ret_name_col = _find_column_by_tokens(result, ["return", "station", "name"])
    if None in {dep_id_col, dep_name_col, ret_id_col, ret_name_col}:
        return False, "result missing one or more route key columns"
    count_col = _find_numeric_metric_col(
        result,
        exclude={dep_id_col, dep_name_col, ret_id_col, ret_name_col},  # type: ignore[arg-type]
        preferred_tokens=["count", "trip"],
    )
    if count_col is None:
        return False, "result missing trip count column"
    if len(result) < 10:
        return False, f"expected at least 10 rows, got {len(result)}"

    expected = (
        source.groupby(
            [
                "Departure station id",
                "Departure station name",
                "Return station id",
                "Return station name",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="trip_count")
        .sort_values(
            ["trip_count", "Departure station id", "Return station id"],
            ascending=[False, True, True],
        )
        .head(10)
        .reset_index(drop=True)
    )
    actual = result.head(10).copy()
    actual["dep_id"] = pd.to_numeric(actual[dep_id_col], errors="coerce")
    actual["dep_name"] = actual[dep_name_col].astype(str)
    actual["ret_id"] = pd.to_numeric(actual[ret_id_col], errors="coerce")
    actual["ret_name"] = actual[ret_name_col].astype(str)
    actual["trip_count"] = pd.to_numeric(actual[count_col], errors="coerce")
    if actual[["dep_id", "ret_id", "trip_count"]].isna().any().any():
        return False, "result contains non-numeric route ids or count"

    exp_rows = [
        (
            int(row["Departure station id"]),
            str(row["Departure station name"]),
            int(row["Return station id"]),
            str(row["Return station name"]),
            int(row["trip_count"]),
        )
        for _, row in expected.iterrows()
    ]
    act_rows = [
        (
            int(row["dep_id"]),
            str(row["dep_name"]),
            int(row["ret_id"]),
            str(row["ret_name"]),
            int(row["trip_count"]),
        )
        for _, row in actual.iterrows()
    ]
    if exp_rows != act_rows:
        return False, "top-10 common routes mismatch"
    return True, "top-10 common routes check passed"


def _validate_hsy_top10_dep_avg_distance_min100(
    result: pd.DataFrame, source: pd.DataFrame
) -> Tuple[bool, str]:
    dep_id_col = _find_column_by_tokens(result, ["departure", "station", "id"])
    dep_name_col = _find_column_by_tokens(result, ["departure", "station", "name"])
    if dep_id_col is None or dep_name_col is None:
        return False, "result missing departure station id/name columns"
    avg_col = _find_numeric_metric_col(
        result,
        exclude={dep_id_col, dep_name_col},
        preferred_tokens=["avg", "average", "distance"],
    )
    if avg_col is None:
        return False, "result missing average distance column"
    if len(result) < 10:
        return False, f"expected at least 10 rows, got {len(result)}"

    frame = source.copy()
    frame["Covered distance (m)"] = pd.to_numeric(frame["Covered distance (m)"], errors="coerce")
    agg = (
        frame.groupby(["Departure station id", "Departure station name"], dropna=False)
        .agg(
            trip_count=("Covered distance (m)", "size"),
            avg_distance=("Covered distance (m)", "mean"),
        )
        .reset_index()
    )
    expected = (
        agg[agg["trip_count"] >= 100]
        .sort_values(["avg_distance", "Departure station id"], ascending=[False, True])
        .head(10)
        .reset_index(drop=True)
    )
    actual = result.head(10).copy()
    actual["dep_id"] = pd.to_numeric(actual[dep_id_col], errors="coerce")
    actual["dep_name"] = actual[dep_name_col].astype(str)
    actual["avg_distance"] = pd.to_numeric(actual[avg_col], errors="coerce")
    if actual[["dep_id", "avg_distance"]].isna().any().any():
        return False, "result contains non-numeric station id or average distance"

    for idx, exp in expected.iterrows():
        act = actual.iloc[idx]
        if int(act["dep_id"]) != int(exp["Departure station id"]):
            return False, "top-10 departure avg-distance ranking mismatch"
        if str(act["dep_name"]) != str(exp["Departure station name"]):
            return False, "departure station name mismatch in avg-distance result"
        if not math.isclose(float(act["avg_distance"]), float(exp["avg_distance"]), rel_tol=0.0, abs_tol=1e-6):
            return False, "average distance mismatch in top-10 departure stations"
    return True, "top-10 departure avg-distance (min 100) check passed"


def _validate_hsy_route_highest_avg_distance_min50(
    result: pd.DataFrame, source: pd.DataFrame
) -> Tuple[bool, str]:
    dep_id_col = _find_column_by_tokens(result, ["departure", "station", "id"])
    dep_name_col = _find_column_by_tokens(result, ["departure", "station", "name"])
    ret_id_col = _find_column_by_tokens(result, ["return", "station", "id"])
    ret_name_col = _find_column_by_tokens(result, ["return", "station", "name"])
    if None in {dep_id_col, dep_name_col, ret_id_col, ret_name_col}:
        return False, "result missing one or more route key columns"
    avg_col = _find_numeric_metric_col(
        result,
        exclude={dep_id_col, dep_name_col, ret_id_col, ret_name_col},  # type: ignore[arg-type]
        preferred_tokens=["avg", "average", "distance"],
    )
    if avg_col is None:
        return False, "result missing average distance column"
    if result.empty:
        return False, "result is empty"

    frame = source.copy()
    frame["Covered distance (m)"] = pd.to_numeric(frame["Covered distance (m)"], errors="coerce")
    agg = (
        frame.groupby(
            [
                "Departure station id",
                "Departure station name",
                "Return station id",
                "Return station name",
            ],
            dropna=False,
        )
        .agg(
            trip_count=("Covered distance (m)", "size"),
            avg_distance=("Covered distance (m)", "mean"),
        )
        .reset_index()
    )
    expected = (
        agg[agg["trip_count"] >= 50]
        .sort_values(["avg_distance", "Departure station id", "Return station id"], ascending=[False, True, True])
        .head(1)
        .reset_index(drop=True)
    )
    act = result.iloc[0]
    if int(pd.to_numeric(pd.Series([act[dep_id_col]]), errors="coerce").iloc[0]) != int(expected.iloc[0]["Departure station id"]):
        return False, "top route departure station id mismatch"
    if str(act[dep_name_col]) != str(expected.iloc[0]["Departure station name"]):
        return False, "top route departure station name mismatch"
    if int(pd.to_numeric(pd.Series([act[ret_id_col]]), errors="coerce").iloc[0]) != int(expected.iloc[0]["Return station id"]):
        return False, "top route return station id mismatch"
    if str(act[ret_name_col]) != str(expected.iloc[0]["Return station name"]):
        return False, "top route return station name mismatch"
    act_avg = float(pd.to_numeric(pd.Series([act[avg_col]]), errors="coerce").iloc[0])
    exp_avg = float(expected.iloc[0]["avg_distance"])
    if not math.isclose(act_avg, exp_avg, rel_tol=0.0, abs_tol=1e-6):
        return False, "top route average distance mismatch"
    return True, "highest average route distance (min 50) check passed"


VALIDATORS: Dict[str, Callable[[pd.DataFrame, pd.DataFrame], Tuple[bool, str]]] = {
    "usgs_top10_strongest": _validate_usgs_top10_strongest,
    "usgs_avg_magtype_top10": _validate_usgs_avg_magtype_top10,
    "usgs_count_per_day_last7": _validate_usgs_count_per_day_last7,
    "usgs_p90_magnitude": _validate_usgs_p90_magnitude,
    "usgs_rolling7_daily_counts": _validate_usgs_rolling7_daily_counts,
    "seattle_avg_tempmax_by_weather": _validate_seattle_avg_tempmax_by_weather,
    "seattle_top10_wettest": _validate_seattle_top10_wettest,
    "seattle_rain_days_per_month": _validate_seattle_rain_days_per_month,
    "seattle_rolling30_precip": _validate_seattle_rolling30_precip,
    "hsy_top5_busiest_departure": _validate_hsy_top5_busiest_departure,
    "hsy_top10_common_routes": _validate_hsy_top10_common_routes,
    "hsy_top10_dep_avg_distance_min100": _validate_hsy_top10_dep_avg_distance_min100,
    "hsy_route_highest_avg_distance_min50": _validate_hsy_route_highest_avg_distance_min50,
    "non_empty_result": _validate_non_empty_result,
    "single_numeric_scalar": _validate_single_numeric_scalar,
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--provider", choices=["ollama"], default="ollama")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args(argv)

    if not args.cases.exists():
        raise SystemExit(f"cases file not found: {args.cases}")

    cases = load_cases(args.cases)
    for case in cases:
        if case.validator not in VALIDATORS:
            raise SystemExit(f"unknown validator {case.validator!r} for case {case.id!r}")

    datasets = sorted({case.dataset for case in cases})
    sources: Dict[str, pd.DataFrame] = {}
    conns: Dict[str, Any] = {}
    schemas: Dict[str, Dict[str, str]] = {}
    for dataset in datasets:
        dataset_path = Path(dataset)
        if not dataset_path.is_absolute():
            dataset_path = REPO_ROOT / dataset_path
        if not dataset_path.exists():
            raise SystemExit(f"dataset not found: {dataset}")
        sources[dataset] = pd.read_csv(dataset_path)
        conn, schema = load_csv(dataset_path)
        conns[dataset] = conn
        schemas[dataset] = schema

    client = OllamaClient(host=args.host, model=args.model, timeout=args.timeout)
    results: List[Dict[str, Any]] = []
    latencies: List[float] = []

    try:
        for case in cases:
            started = time.perf_counter()
            sql: Optional[str] = None
            reply: Optional[str] = None
            row_count = 0
            preview: List[Dict[str, Any]] = []
            error: Optional[str] = None
            passed = False
            validation_note = ""
            routed_to_python = False

            schema = schemas[case.dataset]
            source_df = sources[case.dataset]

            try:
                model = nl_to_query_model(
                    case.question,
                    schema,
                    client=client,
                    config=LLMConfig(host=args.host, model=args.model, timeout=args.timeout),
                )
                sql = model.to_sql()
                reply = model.reply
                output_df = execute(conns[case.dataset], sql)
                row_count = int(len(output_df))
                preview = _preview_rows(output_df)
                passed, validation_note = VALIDATORS[case.validator](output_df, source_df)
            except RouteToPythonError as exc:
                routed_to_python = True
                error = str(exc)
                validation_note = "routed to python analytics path"
                passed = case.track == "python_fit"
            except Exception as exc:
                error = str(exc)
                validation_note = "execution failed before validation"
                passed = False

            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            latencies.append(latency_ms)
            results.append(
                {
                    "id": case.id,
                    "track": case.track,
                    "dataset": case.dataset,
                    "question": case.question,
                    "validator": case.validator,
                    "pass": passed,
                    "validation_note": validation_note,
                    "sql": sql,
                    "reply": reply,
                    "row_count": row_count,
                    "result_preview": preview,
                    "error": error,
                    "routed_to_python": routed_to_python,
                    "latency_ms": latency_ms,
                }
            )
    finally:
        for conn in conns.values():
            try:
                conn.close()
            except Exception:
                pass

    total = len(results)
    passed_total = sum(1 for r in results if r["pass"])
    sql_cases = [r for r in results if r["track"] == "sql_fit"]
    py_cases = [r for r in results if r["track"] == "python_fit"]
    sql_passed = sum(1 for r in sql_cases if r["pass"])
    py_passed = sum(1 for r in py_cases if r["pass"])
    routed_total = sum(1 for r in results if r.get("routed_to_python"))
    sql_routed = sum(1 for r in sql_cases if r.get("routed_to_python"))
    py_routed = sum(1 for r in py_cases if r.get("routed_to_python"))

    summary = {
        "cases_total": total,
        "cases_passed": passed_total,
        "pass_rate": (passed_total / total) if total else 0.0,
        "sql_fit_total": len(sql_cases),
        "sql_fit_passed": sql_passed,
        "sql_fit_pass_rate": (sql_passed / len(sql_cases)) if sql_cases else 0.0,
        "python_fit_total": len(py_cases),
        "python_fit_passed": py_passed,
        "python_fit_pass_rate": (py_passed / len(py_cases)) if py_cases else 0.0,
        "routed_to_python_total": routed_total,
        "sql_fit_routed_to_python": sql_routed,
        "python_fit_routed_to_python": py_routed,
        "latency_ms_p50": (
            sorted(latencies)[len(latencies) // 2] if latencies else 0.0
        ),
        "latency_ms_avg": (sum(latencies) / len(latencies)) if latencies else 0.0,
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": args.model,
        "cases_path": str(args.cases),
        "summary": summary,
        "results": results,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"open_data_sql_vs_python_{args.provider}_{args.model}_{_utc_stamp()}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"summary": summary, "report_path": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
