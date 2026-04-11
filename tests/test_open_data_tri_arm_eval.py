from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import eval.open_data_sql_vs_python_eval as legacy_eval
import eval.open_data_tri_arm_eval as tri_eval
from src.llm.natural_language import parse_query_plan


REPO_ROOT = Path(__file__).resolve().parents[1]


def _usgs_path() -> Path:
    return REPO_ROOT / "data" / "open_data" / "usgs_all_month.csv"


def _seattle_path() -> Path:
    return REPO_ROOT / "data" / "open_data" / "seattle_weather.csv"


def test_run_python_arm_percentile_case() -> None:
    dataset = _usgs_path()
    source = pd.read_csv(dataset)
    result = tri_eval.run_python_arm(
        "What is the 90th percentile of magnitude as one number?",
        dataset,
    )
    ok, note = tri_eval.VALIDATORS["usgs_p90_magnitude"](result, source)
    assert ok, note


def test_run_python_arm_rolling_window_case() -> None:
    dataset = _usgs_path()
    source = pd.read_csv(dataset)
    result = tri_eval.run_python_arm(
        "For the last 14 days, return each day and the 7-day moving average of daily earthquake counts.",
        dataset,
    )
    ok, note = tri_eval.VALIDATORS["usgs_rolling7_daily_counts"](result, source)
    assert ok, note


def test_skill_plan_is_executable() -> None:
    plan = tri_eval.build_skill_operation_plan(
        "Show the 10 wettest days by precipitation with date and precipitation.",
        _seattle_path(),
        "local_v1",
    )
    assert plan.executable
    assert plan.operation_id == "seattle_top10_wettest"
    assert plan.checkpoints


def test_build_summary_shape_and_sql_routing_counters() -> None:
    cases = [
        tri_eval.ProbeCase(
            id="c1",
            track="sql_fit",
            dataset="data/open_data/seattle_weather.csv",
            question="q1",
            validator="seattle_top10_wettest",
        ),
        tri_eval.ProbeCase(
            id="c2",
            track="python_fit",
            dataset="data/open_data/usgs_all_month.csv",
            question="q2",
            validator="usgs_p90_magnitude",
        ),
    ]
    rows = [
        {"arm": "sql", "track": "sql_fit", "validator_pass": True, "routed": {"flag": False}, "latency_ms": 10},
        {"arm": "sql", "track": "python_fit", "validator_pass": False, "routed": {"flag": True}, "latency_ms": 12},
        {"arm": "python", "track": "sql_fit", "validator_pass": True, "routed": {"flag": False}, "latency_ms": 3},
        {"arm": "python", "track": "python_fit", "validator_pass": True, "routed": {"flag": False}, "latency_ms": 4},
        {"arm": "skills", "track": "sql_fit", "validator_pass": True, "routed": {"flag": False}, "latency_ms": 5},
        {"arm": "skills", "track": "python_fit", "validator_pass": True, "routed": {"flag": False}, "latency_ms": 6},
    ]
    summary = tri_eval.build_summary(rows, cases)
    assert set(summary["by_arm"].keys()) == {"sql", "python", "skills"}
    assert set(summary["by_track"].keys()) == {"python_fit", "sql_fit"}
    assert summary["sql_routing"]["confusion"]["true_positive"] == 1
    assert summary["sql_routing"]["confusion"]["true_negative"] == 1
    assert summary["sql_routing"]["confusion"]["false_positive"] == 0
    assert summary["sql_routing"]["confusion"]["false_negative"] == 0


def test_tri_arm_integration_with_mock_provider(tmp_path: Path) -> None:
    cases = [
        {
            "id": "seattle_top10_wettest",
            "track": "sql_fit",
            "dataset": "data/open_data/seattle_weather.csv",
            "question": "Show the 10 wettest days by precipitation with date and precipitation.",
            "validator": "seattle_top10_wettest",
        },
        {
            "id": "usgs_p90",
            "track": "python_fit",
            "dataset": "data/open_data/usgs_all_month.csv",
            "question": "What is the 90th percentile of magnitude as one number?",
            "validator": "usgs_p90_magnitude",
        },
    ]
    cases_path = tmp_path / "tri_cases.json"
    cases_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    rc = tri_eval.main(
        [
            "--provider",
            "mock",
            "--model",
            "mock",
            "--cases",
            str(cases_path),
            "--report-dir",
            str(tmp_path),
            "--skill-profile",
            "local_v1",
        ]
    )
    assert rc == 0

    reports = list(tmp_path.glob("open_data_tri_arm_*.json"))
    assert reports, "tri-arm report was not written"
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    summary = report["summary"]
    assert summary["arm_runs_total"] == 6
    assert summary["by_arm"]["python"]["passed"] == 2
    assert summary["by_arm"]["skills"]["passed"] == 2
    assert summary["sql_routing"]["confusion"]["true_positive"] == 1
    assert summary["sql_routing"]["confusion"]["true_negative"] == 1

    required_fields = {"arm", "routed", "validator_pass", "latency_ms"}
    for row in report["results"]:
        assert required_fields.issubset(row.keys())


def test_legacy_open_data_eval_summary_contract_still_works(tmp_path: Path, monkeypatch) -> None:
    cases = [
        {
            "id": "legacy_seattle_top10_wettest",
            "track": "sql_fit",
            "dataset": "data/open_data/seattle_weather.csv",
            "question": "Show the 10 wettest days by precipitation with date and precipitation.",
            "validator": "seattle_top10_wettest",
        }
    ]
    cases_path = tmp_path / "legacy_cases.json"
    cases_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    class _DummyClient:
        def __init__(self, host: str, model: str, timeout: float = 60.0) -> None:
            _ = (host, model, timeout)

    def _fake_nl_to_query_model(nl: str, schema: dict, **kwargs):  # type: ignore[no-untyped-def]
        _ = (nl, kwargs)
        payload = {
            "reply": "Top wettest days.",
            "selected_columns": ["date", "precipitation"],
            "order_by": [["precipitation", "DESC"]],
            "limit": 10,
        }
        return parse_query_plan(payload, schema)

    monkeypatch.setattr(legacy_eval, "OllamaClient", _DummyClient)
    monkeypatch.setattr(legacy_eval, "nl_to_query_model", _fake_nl_to_query_model)

    rc = legacy_eval.main(
        [
            "--provider",
            "ollama",
            "--model",
            "dummy",
            "--cases",
            str(cases_path),
            "--report-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    reports = list(tmp_path.glob("open_data_sql_vs_python_*.json"))
    assert reports
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    summary = report["summary"]
    assert "by_arm" not in summary
    assert {"sql_fit_total", "python_fit_total", "routed_to_python_total"}.issubset(summary.keys())

