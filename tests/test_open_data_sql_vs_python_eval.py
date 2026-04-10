from __future__ import annotations

from pathlib import Path

import pandas as pd

from eval.open_data_sql_vs_python_eval import (
    _validate_hsy_route_highest_avg_distance_min50,
    _validate_hsy_top5_busiest_departure,
    _validate_non_empty_result,
    _validate_seattle_rolling30_precip,
    _validate_single_numeric_scalar,
    _validate_usgs_p90_magnitude,
    _validate_usgs_top10_strongest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_usgs() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "data" / "open_data" / "usgs_all_month.csv")


def _load_seattle() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "data" / "open_data" / "seattle_weather.csv")


def test_validate_usgs_top10_strongest_passes_on_reference_slice() -> None:
    source = _load_usgs()
    expected = source.sort_values("mag", ascending=False)[["time", "place", "mag"]].head(10).reset_index(drop=True)
    ok, note = _validate_usgs_top10_strongest(expected, source)
    assert ok, note


def test_validate_usgs_p90_magnitude_rejects_wrong_scalar() -> None:
    source = _load_usgs()
    wrong = pd.DataFrame([{"value": float(source["mag"].mean())}])
    ok, note = _validate_usgs_p90_magnitude(wrong, source)
    assert not ok
    assert "p90 mismatch" in note


def test_validate_seattle_rolling30_precip_passes_on_reference_slice() -> None:
    source = _load_seattle()
    ordered = source.assign(date=pd.to_datetime(source["date"], errors="coerce")).sort_values("date")
    ordered["roll30"] = (
        pd.to_numeric(ordered["precipitation"], errors="coerce")
        .rolling(30, min_periods=1)
        .mean()
    )
    expected = ordered[["date", "roll30"]].rename(columns={"date": "date", "roll30": "rolling_precipitation"})
    expected["date"] = expected["date"].dt.strftime("%Y-%m-%d")
    ok, note = _validate_seattle_rolling30_precip(expected, source)
    assert ok, note


def test_validate_non_empty_result() -> None:
    ok, note = _validate_non_empty_result(pd.DataFrame([{"x": 1}]), pd.DataFrame())
    assert ok, note
    ok2, _ = _validate_non_empty_result(pd.DataFrame(), pd.DataFrame())
    assert not ok2


def test_validate_single_numeric_scalar() -> None:
    ok, note = _validate_single_numeric_scalar(pd.DataFrame([{"value": 42}]), pd.DataFrame())
    assert ok, note
    ok2, _ = _validate_single_numeric_scalar(pd.DataFrame([{"value": "x"}]), pd.DataFrame())
    assert not ok2


def test_validate_hsy_top5_busiest_departure_on_synthetic_data() -> None:
    source = pd.DataFrame(
        [
            {"Departure station id": 1, "Departure station name": "A"},
            {"Departure station id": 1, "Departure station name": "A"},
            {"Departure station id": 2, "Departure station name": "B"},
            {"Departure station id": 2, "Departure station name": "B"},
            {"Departure station id": 2, "Departure station name": "B"},
            {"Departure station id": 3, "Departure station name": "C"},
            {"Departure station id": 4, "Departure station name": "D"},
            {"Departure station id": 5, "Departure station name": "E"},
            {"Departure station id": 6, "Departure station name": "F"},
        ]
    )
    result = pd.DataFrame(
        [
            {"Departure station id": 2, "Departure station name": "B", "trip_count": 3},
            {"Departure station id": 1, "Departure station name": "A", "trip_count": 2},
            {"Departure station id": 3, "Departure station name": "C", "trip_count": 1},
            {"Departure station id": 4, "Departure station name": "D", "trip_count": 1},
            {"Departure station id": 5, "Departure station name": "E", "trip_count": 1},
        ]
    )
    ok, note = _validate_hsy_top5_busiest_departure(result, source)
    assert ok, note


def test_validate_hsy_route_highest_avg_distance_min50_on_synthetic_data() -> None:
    rows = []
    for _ in range(60):
        rows.append(
            {
                "Departure station id": 1,
                "Departure station name": "A",
                "Return station id": 2,
                "Return station name": "B",
                "Covered distance (m)": 1000,
            }
        )
    for _ in range(55):
        rows.append(
            {
                "Departure station id": 3,
                "Departure station name": "C",
                "Return station id": 4,
                "Return station name": "D",
                "Covered distance (m)": 1500,
            }
        )
    source = pd.DataFrame(rows)
    result = pd.DataFrame(
        [
            {
                "Departure station id": 3,
                "Departure station name": "C",
                "Return station id": 4,
                "Return station name": "D",
                "avg_distance": 1500.0,
            }
        ]
    )
    ok, note = _validate_hsy_route_highest_avg_distance_min50(result, source)
    assert ok, note
