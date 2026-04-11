"""Tri-arm open-data benchmark: SQL vs Python vs Agent+Skills.

Usage:
    python eval/open_data_tri_arm_eval.py --provider ollama --model gemma4 --skill-profile local_v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Ensure "python eval/open_data_tri_arm_eval.py" can import src/* modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.open_data_sql_vs_python_eval import (  # noqa: E402
    DEFAULT_CASES,
    DEFAULT_REPORT_DIR,
    VALIDATORS,
    ProbeCase,
    load_cases,
)
from src.executor import execute  # noqa: E402
from src.ingestion import load_csv  # noqa: E402
from src.llm.natural_language import (  # noqa: E402
    LLMError,
    LLMConfig,
    OllamaClient,
    RouteToPythonError,
    nl_to_query_model,
)


ARMS = ("sql", "python", "skills")
DEFAULT_SKILL_PROFILE_DIR = REPO_ROOT / "skills" / "local_data_agent" / "profiles"


@dataclass
class SkillOperationPlan:
    skill_profile: str
    dataset_path: str
    question: str
    operation_id: str
    matched_pattern: str
    checkpoints: List[str]
    executable: bool


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _preview_rows(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    return df.head(n).where(pd.notna(df.head(n)), None).to_dict(orient="records")


def _find_column_by_tokens(df: pd.DataFrame, tokens: List[str]) -> Optional[str]:
    lowered = [t.lower() for t in tokens]
    for col in df.columns:
        name = str(col).lower()
        if all(token in name for token in lowered):
            return str(col)
    return None


def _require_column_by_tokens(df: pd.DataFrame, tokens: List[str], label: str) -> str:
    found = _find_column_by_tokens(df, tokens)
    if found is None:
        raise ValueError(f"missing required column for {label}: tokens={tokens!r}")
    return found


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _last_n_day_window(series: pd.Series, days: int) -> pd.DataFrame:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    day_text = parsed.dt.strftime("%Y-%m-%d")
    frame = pd.DataFrame({"day": day_text}).dropna()
    if frame.empty:
        return frame
    max_day = datetime.strptime(str(frame["day"].max()), "%Y-%m-%d").date()
    start_day = max_day - timedelta(days=days - 1)
    frame["day_date"] = pd.to_datetime(frame["day"], errors="coerce").dt.date
    return frame[(frame["day_date"] >= start_day) & (frame["day_date"] <= max_day)].copy()


def _resolve_dataset_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _op_usgs_top10_strongest(source: pd.DataFrame) -> pd.DataFrame:
    _require_columns(source, ["time", "place", "mag"])
    frame = source.copy()
    frame["mag"] = pd.to_numeric(frame["mag"], errors="coerce")
    return (
        frame.dropna(subset=["mag"])
        .sort_values("mag", ascending=False)
        [["time", "place", "mag"]]
        .head(10)
        .reset_index(drop=True)
    )


def _op_usgs_avg_magtype_top10(source: pd.DataFrame) -> pd.DataFrame:
    _require_columns(source, ["magType", "mag"])
    frame = source.copy()
    frame["mag"] = pd.to_numeric(frame["mag"], errors="coerce")
    return (
        frame.dropna(subset=["magType", "mag"])
        .groupby("magType", dropna=True)["mag"]
        .mean()
        .reset_index(name="avg_mag")
        .sort_values("avg_mag", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )


def _op_usgs_count_per_day_last7(source: pd.DataFrame) -> pd.DataFrame:
    _require_columns(source, ["time"])
    window = _last_n_day_window(source["time"], days=7)
    if window.empty:
        return pd.DataFrame(columns=["day", "daily_count"])
    return (
        window.groupby("day", dropna=True)
        .size()
        .reset_index(name="daily_count")
        .sort_values("day")
        .reset_index(drop=True)
    )


def _op_usgs_p90_magnitude(source: pd.DataFrame) -> pd.DataFrame:
    _require_columns(source, ["mag"])
    vals = pd.to_numeric(source["mag"], errors="coerce").dropna()
    if vals.empty:
        return pd.DataFrame([{"p90_magnitude": float("nan")}])
    return pd.DataFrame([{"p90_magnitude": float(vals.quantile(0.9))}])


def _op_usgs_rolling7_daily_counts(source: pd.DataFrame) -> pd.DataFrame:
    _require_columns(source, ["time"])
    window = _last_n_day_window(source["time"], days=14)
    if window.empty:
        return pd.DataFrame(columns=["day", "rolling_daily_count"])
    counts = (
        window.groupby("day", dropna=True)
        .size()
        .sort_index()
        .astype("float64")
        .rolling(window=7, min_periods=1)
        .mean()
    )
    return counts.reset_index(name="rolling_daily_count")


def _op_seattle_avg_tempmax_by_weather(source: pd.DataFrame) -> pd.DataFrame:
    _require_columns(source, ["weather", "temp_max"])
    frame = source.copy()
    frame["temp_max"] = pd.to_numeric(frame["temp_max"], errors="coerce")
    return (
        frame.dropna(subset=["weather", "temp_max"])
        .groupby("weather", dropna=True)["temp_max"]
        .mean()
        .reset_index(name="avg_temp_max")
        .sort_values("avg_temp_max", ascending=False)
        .reset_index(drop=True)
    )


def _op_seattle_top10_wettest(source: pd.DataFrame) -> pd.DataFrame:
    _require_columns(source, ["date", "precipitation"])
    frame = source.copy()
    frame["precipitation"] = pd.to_numeric(frame["precipitation"], errors="coerce")
    return (
        frame.dropna(subset=["precipitation"])
        .sort_values("precipitation", ascending=False)
        [["date", "precipitation"]]
        .head(10)
        .reset_index(drop=True)
    )


def _op_seattle_rain_days_per_month(source: pd.DataFrame) -> pd.DataFrame:
    _require_columns(source, ["date", "weather"])
    rainy = source[source["weather"].astype(str).str.lower() == "rain"].copy()
    if rainy.empty:
        return pd.DataFrame(columns=["month", "rain_days"])
    rainy["month"] = rainy["date"].astype(str).str.slice(0, 7)
    return (
        rainy.groupby("month", dropna=True)
        .size()
        .reset_index(name="rain_days")
        .sort_values("month")
        .reset_index(drop=True)
    )


def _op_seattle_rolling30_precip(source: pd.DataFrame) -> pd.DataFrame:
    _require_columns(source, ["date", "precipitation"])
    frame = source.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["precipitation"] = pd.to_numeric(frame["precipitation"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    frame["rolling_precipitation"] = frame["precipitation"].rolling(30, min_periods=1).mean()
    out = frame[["date", "rolling_precipitation"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def _op_hsy_top5_busiest_departure(source: pd.DataFrame) -> pd.DataFrame:
    dep_id = _require_column_by_tokens(source, ["departure", "station", "id"], "departure station id")
    dep_name = _require_column_by_tokens(source, ["departure", "station", "name"], "departure station name")
    return (
        source.groupby([dep_id, dep_name], dropna=False)
        .size()
        .reset_index(name="trip_count")
        .sort_values(["trip_count", dep_id], ascending=[False, True])
        .head(5)
        .reset_index(drop=True)
    )


def _op_hsy_top10_common_routes(source: pd.DataFrame) -> pd.DataFrame:
    dep_id = _require_column_by_tokens(source, ["departure", "station", "id"], "departure station id")
    dep_name = _require_column_by_tokens(source, ["departure", "station", "name"], "departure station name")
    ret_id = _require_column_by_tokens(source, ["return", "station", "id"], "return station id")
    ret_name = _require_column_by_tokens(source, ["return", "station", "name"], "return station name")
    return (
        source.groupby([dep_id, dep_name, ret_id, ret_name], dropna=False)
        .size()
        .reset_index(name="trip_count")
        .sort_values(["trip_count", dep_id, ret_id], ascending=[False, True, True])
        .head(10)
        .reset_index(drop=True)
    )


def _op_hsy_top10_dep_avg_distance_min100(source: pd.DataFrame) -> pd.DataFrame:
    dep_id = _require_column_by_tokens(source, ["departure", "station", "id"], "departure station id")
    dep_name = _require_column_by_tokens(source, ["departure", "station", "name"], "departure station name")
    dist = _require_column_by_tokens(source, ["covered", "distance"], "covered distance")
    frame = source.copy()
    frame[dist] = pd.to_numeric(frame[dist], errors="coerce")
    agg = (
        frame.groupby([dep_id, dep_name], dropna=False)
        .agg(
            trip_count=(dist, "size"),
            avg_distance=(dist, "mean"),
        )
        .reset_index()
    )
    return (
        agg[agg["trip_count"] >= 100]
        .sort_values(["avg_distance", dep_id], ascending=[False, True])
        .head(10)
        .reset_index(drop=True)
    )


def _op_hsy_route_highest_avg_distance_min50(source: pd.DataFrame) -> pd.DataFrame:
    dep_id = _require_column_by_tokens(source, ["departure", "station", "id"], "departure station id")
    dep_name = _require_column_by_tokens(source, ["departure", "station", "name"], "departure station name")
    ret_id = _require_column_by_tokens(source, ["return", "station", "id"], "return station id")
    ret_name = _require_column_by_tokens(source, ["return", "station", "name"], "return station name")
    dist = _require_column_by_tokens(source, ["covered", "distance"], "covered distance")
    frame = source.copy()
    frame[dist] = pd.to_numeric(frame[dist], errors="coerce")
    agg = (
        frame.groupby([dep_id, dep_name, ret_id, ret_name], dropna=False)
        .agg(
            trip_count=(dist, "size"),
            avg_distance=(dist, "mean"),
        )
        .reset_index()
    )
    return (
        agg[agg["trip_count"] >= 50]
        .sort_values(["avg_distance", dep_id, ret_id], ascending=[False, True, True])
        .head(1)
        .reset_index(drop=True)
    )


def _op_hsy_trips_per_departure_hour(source: pd.DataFrame) -> pd.DataFrame:
    dep_time = _require_column_by_tokens(source, ["departure"], "departure timestamp")
    frame = source.copy()
    parsed = pd.to_datetime(frame[dep_time], errors="coerce")
    frame["hour_of_day"] = parsed.dt.hour
    return (
        frame.dropna(subset=["hour_of_day"])
        .groupby("hour_of_day", dropna=True)
        .size()
        .reset_index(name="trip_count")
        .sort_values("hour_of_day")
        .reset_index(drop=True)
    )


def _op_hsy_same_station_round_trips_count(source: pd.DataFrame) -> pd.DataFrame:
    dep_id = _require_column_by_tokens(source, ["departure", "station", "id"], "departure station id")
    ret_id = _require_column_by_tokens(source, ["return", "station", "id"], "return station id")
    dep_vals = pd.to_numeric(source[dep_id], errors="coerce")
    ret_vals = pd.to_numeric(source[ret_id], errors="coerce")
    count = int((dep_vals == ret_vals).fillna(False).sum())
    return pd.DataFrame([{"trip_count": count}])


def _op_hsy_stations_more_departures_than_returns(source: pd.DataFrame) -> pd.DataFrame:
    dep_id = _require_column_by_tokens(source, ["departure", "station", "id"], "departure station id")
    dep_name = _require_column_by_tokens(source, ["departure", "station", "name"], "departure station name")
    ret_id = _require_column_by_tokens(source, ["return", "station", "id"], "return station id")
    ret_name = _require_column_by_tokens(source, ["return", "station", "name"], "return station name")
    dep = source.groupby([dep_id, dep_name], dropna=False).size().reset_index(name="departure_trip_count")
    ret = source.groupby([ret_id, ret_name], dropna=False).size().reset_index(name="return_trip_count")
    dep = dep.rename(columns={dep_id: "station_id", dep_name: "station_name"})
    ret = ret.rename(columns={ret_id: "station_id", ret_name: "station_name"})
    merged = dep.merge(ret, on=["station_id", "station_name"], how="inner")
    merged["difference"] = (
        pd.to_numeric(merged["departure_trip_count"], errors="coerce")
        - pd.to_numeric(merged["return_trip_count"], errors="coerce")
    )
    return (
        merged[merged["difference"] > 0]
        .sort_values(["difference", "station_id"], ascending=[False, True])
        .head(10)
        .reset_index(drop=True)
    )


def _op_hsy_top10_same_station_percentage_min100_departures(source: pd.DataFrame) -> pd.DataFrame:
    dep_id = _require_column_by_tokens(source, ["departure", "station", "id"], "departure station id")
    dep_name = _require_column_by_tokens(source, ["departure", "station", "name"], "departure station name")
    ret_id = _require_column_by_tokens(source, ["return", "station", "id"], "return station id")
    frame = source.copy()
    frame["same_station"] = (
        pd.to_numeric(frame[dep_id], errors="coerce") == pd.to_numeric(frame[ret_id], errors="coerce")
    ).fillna(False)
    agg = (
        frame.groupby([dep_id, dep_name], dropna=False)
        .agg(
            departure_trip_count=("same_station", "size"),
            same_station_trip_count=("same_station", "sum"),
        )
        .reset_index()
    )
    agg["same_station_pct"] = (
        100.0
        * pd.to_numeric(agg["same_station_trip_count"], errors="coerce")
        / pd.to_numeric(agg["departure_trip_count"], errors="coerce")
    )
    return (
        agg[agg["departure_trip_count"] >= 100]
        .sort_values(["same_station_pct", dep_id], ascending=[False, True])
        .head(10)
        .reset_index(drop=True)
    )


def _op_non_empty_result(source: pd.DataFrame) -> pd.DataFrame:
    return source.head(20).copy()


def _op_single_numeric_scalar(source: pd.DataFrame) -> pd.DataFrame:
    for col in source.columns:
        numeric = pd.to_numeric(source[col], errors="coerce")
        if numeric.notna().mean() >= 0.8:
            return pd.DataFrame([{"value": float(numeric.dropna().mean())}])
    return pd.DataFrame([{"value": 0.0}])


OPERATION_EXECUTORS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "usgs_top10_strongest": _op_usgs_top10_strongest,
    "usgs_avg_magtype_top10": _op_usgs_avg_magtype_top10,
    "usgs_count_per_day_last7": _op_usgs_count_per_day_last7,
    "usgs_p90_magnitude": _op_usgs_p90_magnitude,
    "usgs_rolling7_daily_counts": _op_usgs_rolling7_daily_counts,
    "seattle_avg_tempmax_by_weather": _op_seattle_avg_tempmax_by_weather,
    "seattle_top10_wettest": _op_seattle_top10_wettest,
    "seattle_rain_days_per_month": _op_seattle_rain_days_per_month,
    "seattle_rolling30_precip": _op_seattle_rolling30_precip,
    "hsy_top5_busiest_departure": _op_hsy_top5_busiest_departure,
    "hsy_top10_common_routes": _op_hsy_top10_common_routes,
    "hsy_top10_dep_avg_distance_min100": _op_hsy_top10_dep_avg_distance_min100,
    "hsy_route_highest_avg_distance_min50": _op_hsy_route_highest_avg_distance_min50,
    "hsy_trips_per_departure_hour": _op_hsy_trips_per_departure_hour,
    "hsy_same_station_round_trips_count": _op_hsy_same_station_round_trips_count,
    "hsy_stations_more_departures_than_returns": _op_hsy_stations_more_departures_than_returns,
    "hsy_top10_same_station_percentage_min100_departures": _op_hsy_top10_same_station_percentage_min100_departures,
    "non_empty_result": _op_non_empty_result,
    "single_numeric_scalar": _op_single_numeric_scalar,
}


QUESTION_RULES: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(10|ten)\s+strongest\b.*\bearthquake", re.IGNORECASE), "usgs_top10_strongest"),
    (re.compile(r"\baverage magnitude by magtype\b", re.IGNORECASE), "usgs_avg_magtype_top10"),
    (re.compile(r"\bhow many earthquakes per day\b.*\blast 7 days\b", re.IGNORECASE), "usgs_count_per_day_last7"),
    (re.compile(r"\b90th percentile\b.*\bmagnitude\b", re.IGNORECASE), "usgs_p90_magnitude"),
    (
        re.compile(r"\b(7-day|7 day)\b.*\b(moving|rolling)\b.*\b(earthquake|daily)\b", re.IGNORECASE),
        "usgs_rolling7_daily_counts",
    ),
    (re.compile(r"\baverage temp[_\s]?max by weather\b", re.IGNORECASE), "seattle_avg_tempmax_by_weather"),
    (re.compile(r"\b(10|ten)\s+wettest\b", re.IGNORECASE), "seattle_top10_wettest"),
    (re.compile(r"\brain(y)? days per month\b", re.IGNORECASE), "seattle_rain_days_per_month"),
    (
        re.compile(r"\b(30-day|30 day)\b.*\b(rolling|moving)\b.*\bprecipitation\b", re.IGNORECASE),
        "seattle_rolling30_precip",
    ),
    (re.compile(r"\b(5|five)\s+busiest departure stations\b", re.IGNORECASE), "hsy_top5_busiest_departure"),
    (re.compile(r"\b(10|ten)\s+most common\b.*\broutes\b", re.IGNORECASE), "hsy_top10_common_routes"),
    (
        re.compile(r"\blongest average trip distance\b.*\bat least 100 trips\b", re.IGNORECASE),
        "hsy_top10_dep_avg_distance_min100",
    ),
    (
        re.compile(r"\bhighest average covered distance\b.*\bat least 50 trips\b", re.IGNORECASE),
        "hsy_route_highest_avg_distance_min50",
    ),
    (re.compile(r"\bfor each departure hour\b", re.IGNORECASE), "hsy_trips_per_departure_hour"),
    (
        re.compile(r"\bstart and end at the same station\b|\bdeparture station id equals return station id\b", re.IGNORECASE),
        "hsy_same_station_round_trips_count",
    ),
    (
        re.compile(r"\bmore departures than returns\b", re.IGNORECASE),
        "hsy_stations_more_departures_than_returns",
    ),
    (
        re.compile(r"\bpercentage of trips that return to the same station\b", re.IGNORECASE),
        "hsy_top10_same_station_percentage_min100_departures",
    ),
]


def infer_operation_id(question: str, *, fallback: str = "non_empty_result") -> str:
    text = (question or "").strip()
    for pattern, op_id in QUESTION_RULES:
        if pattern.search(text):
            return op_id
    return fallback


def _execute_operation_id(operation_id: str, source_df: pd.DataFrame) -> pd.DataFrame:
    fn = OPERATION_EXECUTORS.get(operation_id)
    if fn is None:
        raise ValueError(f"unsupported operation_id: {operation_id}")
    return fn(source_df)


def run_python_arm(question: str, dataset_path: str | Path) -> pd.DataFrame:
    """Run deterministic pandas analytics for the Python arm."""
    path = _resolve_dataset_path(dataset_path)
    source_df = pd.read_csv(path)
    op_id = infer_operation_id(question)
    return _execute_operation_id(op_id, source_df)


def _load_skill_profile(skill_profile: str | Path | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(skill_profile, dict):
        return skill_profile
    profile_path = Path(skill_profile)
    if not profile_path.exists():
        profile_path = DEFAULT_SKILL_PROFILE_DIR / f"{skill_profile}.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"skill profile not found: {skill_profile}")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("skill profile must be a JSON object")
    return payload


def build_skill_operation_plan(
    question: str, dataset_path: str | Path, skill_profile: str | Path | Dict[str, Any]
) -> SkillOperationPlan:
    profile = _load_skill_profile(skill_profile)
    profile_name = str(profile.get("profile") or profile.get("name") or "custom")
    checkpoints = profile.get("checkpoints") or []
    if not isinstance(checkpoints, list):
        raise ValueError("skill profile checkpoints must be a list")
    route_entries = profile.get("routes") or []
    if not isinstance(route_entries, list):
        raise ValueError("skill profile routes must be a list")

    matched_pattern = "<fallback>"
    operation_id = ""
    for entry in route_entries:
        if not isinstance(entry, dict):
            continue
        pattern_text = entry.get("pattern")
        entry_op = entry.get("operation_id")
        if not isinstance(pattern_text, str) or not isinstance(entry_op, str):
            continue
        if re.search(pattern_text, question, re.IGNORECASE):
            operation_id = entry_op
            matched_pattern = pattern_text
            break

    if not operation_id:
        fallback = str(profile.get("fallback_operation_id") or "non_empty_result")
        operation_id = infer_operation_id(question, fallback=fallback)

    executable = operation_id in OPERATION_EXECUTORS
    return SkillOperationPlan(
        skill_profile=profile_name,
        dataset_path=str(_resolve_dataset_path(dataset_path)),
        question=question,
        operation_id=operation_id,
        matched_pattern=matched_pattern,
        checkpoints=[str(cp) for cp in checkpoints],
        executable=executable,
    )


def run_skill_arm(
    question: str,
    dataset_path: str | Path,
    skill_profile: str | Path | Dict[str, Any],
) -> pd.DataFrame:
    """Run skill-guided deterministic analytics for the Skills arm."""
    plan = build_skill_operation_plan(question, dataset_path, skill_profile)
    if not plan.executable:
        raise LLMError(f"skill plan produced non-executable operation: {plan.operation_id}")
    source_df = pd.read_csv(plan.dataset_path)
    return _execute_operation_id(plan.operation_id, source_df)


def _mock_sql_plan_for_question(question: str) -> Dict[str, Any]:
    op_id = infer_operation_id(question)
    if op_id == "usgs_top10_strongest":
        return {
            "reply": "Top 10 strongest earthquakes by magnitude.",
            "selected_columns": ["time", "place", "mag"],
            "order_by": [["mag", "DESC"]],
            "limit": 10,
        }
    if op_id == "usgs_avg_magtype_top10":
        return {
            "reply": "Average magnitude by magType, highest first.",
            "selected_columns": ["magType"],
            "group_by": ["magType"],
            "aggregations": [{"function": "AVG", "column": "mag", "alias": "avg_mag"}],
            "order_by": [["avg_mag", "DESC"]],
            "limit": 10,
        }
    if op_id == "usgs_count_per_day_last7":
        return {
            "reply": "Daily earthquake counts for the last 7 days.",
            "selected_columns": ["time"],
            "group_by": ["time"],
            "aggregations": [{"function": "COUNT", "column": "*", "alias": "daily_count"}],
            "filters": [{"column": "time", "operator": ">=", "value": "date_7_days_ago"}],
            "date_buckets": {"time": "day"},
            "order_by": [["time", "ASC"]],
        }
    if op_id == "seattle_avg_tempmax_by_weather":
        return {
            "reply": "Average temp_max by weather.",
            "selected_columns": ["weather"],
            "group_by": ["weather"],
            "aggregations": [{"function": "AVG", "column": "temp_max", "alias": "avg_temp_max"}],
            "order_by": [["avg_temp_max", "DESC"]],
        }
    if op_id == "seattle_top10_wettest":
        return {
            "reply": "Top 10 wettest days by precipitation.",
            "selected_columns": ["date", "precipitation"],
            "order_by": [["precipitation", "DESC"]],
            "limit": 10,
        }
    if op_id == "seattle_rain_days_per_month":
        return {
            "reply": "Daily rain counts, sortable to monthly totals.",
            "selected_columns": ["date"],
            "group_by": ["date"],
            "aggregations": [{"function": "COUNT", "column": "*", "alias": "rain_days"}],
            "filters": [{"column": "weather", "operator": "=", "value": "rain"}],
            "order_by": [["date", "ASC"]],
        }
    if op_id == "hsy_top5_busiest_departure":
        return {
            "reply": "Top 5 busiest departure stations by trip count.",
            "selected_columns": ["Departure station id", "Departure station name"],
            "group_by": ["Departure station id", "Departure station name"],
            "aggregations": [{"function": "COUNT", "column": "*", "alias": "trip_count"}],
            "order_by": [["trip_count", "DESC"], ["Departure station id", "ASC"]],
            "limit": 5,
        }
    if op_id == "hsy_top10_common_routes":
        return {
            "reply": "Top 10 most common routes.",
            "selected_columns": [
                "Departure station id",
                "Departure station name",
                "Return station id",
                "Return station name",
            ],
            "group_by": [
                "Departure station id",
                "Departure station name",
                "Return station id",
                "Return station name",
            ],
            "aggregations": [{"function": "COUNT", "column": "*", "alias": "trip_count"}],
            "order_by": [["trip_count", "DESC"], ["Departure station id", "ASC"], ["Return station id", "ASC"]],
            "limit": 10,
        }
    if op_id == "hsy_top10_dep_avg_distance_min100":
        return {
            "reply": "Top departure stations by average covered distance with min trip threshold.",
            "selected_columns": ["Departure station id", "Departure station name"],
            "group_by": ["Departure station id", "Departure station name"],
            "aggregations": [
                {"function": "AVG", "column": "Covered distance (m)", "alias": "avg_distance"},
                {"function": "COUNT", "column": "*", "alias": "trip_count"},
            ],
            "having": [{"column": "trip_count", "operator": ">=", "value": 100}],
            "order_by": [["avg_distance", "DESC"], ["Departure station id", "ASC"]],
            "limit": 10,
        }
    if op_id == "hsy_route_highest_avg_distance_min50":
        return {
            "reply": "Route with highest average covered distance with min trip threshold.",
            "selected_columns": [
                "Departure station id",
                "Departure station name",
                "Return station id",
                "Return station name",
            ],
            "group_by": [
                "Departure station id",
                "Departure station name",
                "Return station id",
                "Return station name",
            ],
            "aggregations": [
                {"function": "AVG", "column": "Covered distance (m)", "alias": "avg_distance"},
                {"function": "COUNT", "column": "*", "alias": "trip_count"},
            ],
            "having": [{"column": "trip_count", "operator": ">=", "value": 50}],
            "order_by": [["avg_distance", "DESC"], ["Departure station id", "ASC"], ["Return station id", "ASC"]],
            "limit": 1,
        }
    return {"reply": "Fallback selection.", "selected_columns": []}


class MockSQLClient:
    """OllamaClient-like mock for tests and offline tri-arm smoke runs."""

    def generate_json(self, system: str, user: str) -> Dict[str, Any]:
        _ = system
        match = re.search(r"User request:\s*(.*?)\nJSON:", user, re.DOTALL)
        question = match.group(1).strip() if match else ""
        return _mock_sql_plan_for_question(question)


def _run_sql_arm(
    case: ProbeCase,
    schema: Dict[str, str],
    conn: Any,
    source_df: pd.DataFrame,
    client: Any,
    llm_config: LLMConfig,
) -> Dict[str, Any]:
    started = time.perf_counter()
    sql: Optional[str] = None
    reply: Optional[str] = None
    output_df: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    validation_note = ""
    validator_pass = False
    routed = {"flag": False, "reason": ""}
    try:
        model = nl_to_query_model(case.question, schema, client=client, config=llm_config)
        sql = model.to_sql()
        reply = model.reply
        output_df = execute(conn, sql)
        validator_pass, validation_note = VALIDATORS[case.validator](output_df, source_df)
    except RouteToPythonError as exc:
        routed = {"flag": True, "reason": str(exc.reason)}
        error = str(exc)
        validation_note = "routed to python analytics path"
        validator_pass = False
    except Exception as exc:
        error = str(exc)
        validation_note = "execution failed before validation"
        validator_pass = False

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    row_count = int(len(output_df)) if output_df is not None else 0
    preview = _preview_rows(output_df) if output_df is not None else []
    return {
        "arm": "sql",
        "id": case.id,
        "track": case.track,
        "dataset": case.dataset,
        "question": case.question,
        "validator": case.validator,
        "validator_pass": validator_pass,
        "pass": validator_pass,
        "validation_note": validation_note,
        "sql": sql,
        "reply": reply,
        "row_count": row_count,
        "result_preview": preview,
        "error": error,
        "routed": routed,
        "latency_ms": latency_ms,
    }


def _run_python_or_skill_arm(
    arm: str,
    case: ProbeCase,
    source_df: pd.DataFrame,
    skill_profile: str | Path | Dict[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    output_df: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    validation_note = ""
    validator_pass = False
    operation_id = ""
    operation_plan: Optional[Dict[str, Any]] = None

    try:
        if arm == "python":
            operation_id = infer_operation_id(case.question)
            output_df = _execute_operation_id(operation_id, source_df)
        elif arm == "skills":
            plan = build_skill_operation_plan(case.question, case.dataset, skill_profile)
            operation_plan = asdict(plan)
            operation_id = plan.operation_id
            if not plan.executable:
                raise LLMError(f"skill plan produced non-executable operation: {plan.operation_id}")
            output_df = _execute_operation_id(plan.operation_id, source_df)
        else:
            raise ValueError(f"unsupported arm: {arm}")
        validator_pass, validation_note = VALIDATORS[case.validator](output_df, source_df)
    except Exception as exc:
        error = str(exc)
        validation_note = "execution failed before validation"
        validator_pass = False

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    row_count = int(len(output_df)) if output_df is not None else 0
    preview = _preview_rows(output_df) if output_df is not None else []
    return {
        "arm": arm,
        "id": case.id,
        "track": case.track,
        "dataset": case.dataset,
        "question": case.question,
        "validator": case.validator,
        "validator_pass": validator_pass,
        "pass": validator_pass,
        "validation_note": validation_note,
        "row_count": row_count,
        "result_preview": preview,
        "error": error,
        "routed": {"flag": False, "reason": ""},
        "latency_ms": latency_ms,
        "operation_id": operation_id,
        "operation_plan": operation_plan,
    }


def _metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    data = list(rows)
    total = len(data)
    passed = sum(1 for row in data if row.get("validator_pass"))
    routed = sum(1 for row in data if bool((row.get("routed") or {}).get("flag")))
    latencies = [float(row.get("latency_ms", 0.0)) for row in data]
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total) if total else 0.0,
        "routed_total": routed,
        "latency_ms_avg": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "latency_ms_p50": (sorted(latencies)[len(latencies) // 2] if latencies else 0.0),
    }


def build_summary(results: List[Dict[str, Any]], cases: List[ProbeCase]) -> Dict[str, Any]:
    by_arm = {arm: _metrics(r for r in results if r.get("arm") == arm) for arm in ARMS}

    by_track: Dict[str, Any] = {}
    tracks = sorted({case.track for case in cases})
    for track in tracks:
        track_cases = [case for case in cases if case.track == track]
        by_track[track] = {
            "total_cases": len(track_cases),
            "by_arm": {
                arm: _metrics(
                    row for row in results if row.get("track") == track and row.get("arm") == arm
                )
                for arm in ARMS
            },
        }

    sql_rows = [row for row in results if row.get("arm") == "sql"]
    tp = sum(1 for row in sql_rows if row.get("track") == "python_fit" and row.get("routed", {}).get("flag"))
    tn = sum(1 for row in sql_rows if row.get("track") == "sql_fit" and not row.get("routed", {}).get("flag"))
    fp = sum(1 for row in sql_rows if row.get("track") == "sql_fit" and row.get("routed", {}).get("flag"))
    fn = sum(1 for row in sql_rows if row.get("track") == "python_fit" and not row.get("routed", {}).get("flag"))
    sql_routing = {
        "routed_total": sum(1 for row in sql_rows if row.get("routed", {}).get("flag")),
        "routed_sql_fit": fp,
        "routed_python_fit": tp,
        "not_routed_sql_fit": tn,
        "not_routed_python_fit": fn,
        "confusion": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
    }

    return {
        "cases_total": len(cases),
        "arm_runs_total": len(results),
        "by_arm": by_arm,
        "by_track": by_track,
        "sql_routing": sql_routing,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--provider", choices=["ollama", "mock"], default="ollama")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--skill-profile", default="local_v1")
    args = parser.parse_args(argv)

    if not args.cases.exists():
        raise SystemExit(f"cases file not found: {args.cases}")

    cases = load_cases(args.cases)
    for case in cases:
        if case.validator not in VALIDATORS:
            raise SystemExit(f"unknown validator {case.validator!r} for case {case.id!r}")

    # Validate skill profile early.
    _load_skill_profile(args.skill_profile)

    datasets = sorted({case.dataset for case in cases})
    sources: Dict[str, pd.DataFrame] = {}
    conns: Dict[str, Any] = {}
    schemas: Dict[str, Dict[str, str]] = {}
    for dataset in datasets:
        dataset_path = _resolve_dataset_path(dataset)
        if not dataset_path.exists():
            raise SystemExit(f"dataset not found: {dataset}")
        sources[dataset] = pd.read_csv(dataset_path)
        conn, schema = load_csv(dataset_path)
        conns[dataset] = conn
        schemas[dataset] = schema

    if args.provider == "mock":
        client: Any = MockSQLClient()
    else:
        client = OllamaClient(host=args.host, model=args.model, timeout=args.timeout)

    llm_config = LLMConfig(host=args.host, model=args.model, timeout=args.timeout)
    results: List[Dict[str, Any]] = []

    try:
        for case in cases:
            source_df = sources[case.dataset]
            conn = conns[case.dataset]
            schema = schemas[case.dataset]

            results.append(_run_sql_arm(case, schema, conn, source_df, client, llm_config))
            results.append(_run_python_or_skill_arm("python", case, source_df, args.skill_profile))
            results.append(_run_python_or_skill_arm("skills", case, source_df, args.skill_profile))
    finally:
        for conn in conns.values():
            try:
                conn.close()
            except Exception:
                pass

    summary = build_summary(results, cases)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": args.model,
        "skill_profile": str(args.skill_profile),
        "cases_path": str(args.cases),
        "summary": summary,
        "results": results,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    safe_profile = re.sub(r"[^A-Za-z0-9._-]+", "-", str(args.skill_profile)).strip("-") or "profile"
    report_path = (
        args.report_dir
        / f"open_data_tri_arm_{args.provider}_{args.model}_{safe_profile}_{_utc_stamp()}.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "report_path": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
