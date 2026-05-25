from __future__ import annotations

from pathlib import Path

from eval.open_data_sql_vs_python_eval import VALIDATORS
from src.mixed_execution.engine import MixedExecutionEngine
from src.mixed_execution.logical_plan import (
    LogicalPlan,
    OutputColumn,
    PlanFilter,
    PlanOrder,
    PlanSource,
    PostProcessingStep,
)
from src.mixed_execution.plan_validator import validate_logical_plan
from src.mixed_execution.router import SourceProfile, route_plan
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _seattle_path() -> Path:
    return REPO_ROOT / "data" / "open_data" / "seattle_weather.csv"


def _usgs_path() -> Path:
    return REPO_ROOT / "data" / "open_data" / "usgs_all_month.csv"


def test_plan_validator_rejects_unknown_column() -> None:
    schema = {"date": "date", "precipitation": "numeric"}
    plan = LogicalPlan(
        source=PlanSource(kind="table_or_file", name="weather"),
        projection=["ghost_column"],
        filters=[PlanFilter(column="date", op="IS NOT NULL")],
        group_by=[],
        aggregates=[],
        order_by=[PlanOrder(expr="date", direction="asc")],
        post_processing=[],
        expected_output_schema=[OutputColumn(name="date", type="date")],
        metadata={},
    )
    result = validate_logical_plan(plan, schema)
    assert not result.ok
    assert any("unknown projection column" in err for err in result.errors)


def test_router_prefers_pushdown_for_simple_case() -> None:
    plan = LogicalPlan(
        source=PlanSource(kind="table_or_file", name="weather"),
        projection=["date", "precipitation"],
        filters=[PlanFilter(column="precipitation", op="IS NOT NULL")],
        group_by=[],
        aggregates=[],
        order_by=[PlanOrder(expr="precipitation", direction="desc")],
        limit=10,
        post_processing=[],
        expected_output_schema=[],
        metadata={},
    )
    decision = route_plan(plan, SourceProfile(rows_estimate=500_000, is_remote=False, header_confidence=1.0))
    assert decision.route == "pushdown"


def test_router_prefers_hybrid_for_analytics_after_pushdown() -> None:
    plan = LogicalPlan(
        source=PlanSource(kind="table_or_file", name="earthquakes"),
        projection=["time", "mag"],
        filters=[PlanFilter(column="time", op=">=", value="NOW_MINUS_14_DAYS")],
        group_by=[],
        aggregates=[],
        order_by=[],
        post_processing=[],
        expected_output_schema=[],
        metadata={},
    )
    plan = LogicalPlan(
        source=plan.source,
        projection=plan.projection,
        filters=plan.filters,
        group_by=[],
        aggregates=[],
        order_by=[],
        post_processing=[PostProcessingStep(kind="rolling_mean", params={})],
        expected_output_schema=[],
        metadata={},
    )
    decision = route_plan(plan, SourceProfile(rows_estimate=250_000, is_remote=False, header_confidence=1.0))
    assert decision.route in {"hybrid", "python"}


def test_engine_pushdown_case_executes_and_validates() -> None:
    engine = MixedExecutionEngine()
    source = pd.read_csv(_seattle_path())
    run = engine.run(
        question="Show the 10 wettest days by precipitation with date and precipitation.",
        dataset_path=_seattle_path(),
        expected_route_family="pushdown",
    )
    assert run.plan_valid
    assert run.route == "pushdown"
    assert run.result is not None
    ok, note = VALIDATORS["seattle_top10_wettest"](run.result.dataframe, source)
    assert ok, note
    assert run.schema_correct
    assert run.backend == "dataframe_pushdown"
    assert run.routing_artifact["table_type"] in {
        "structured_table",
        "label_indexed_report",
        "ambiguous_table",
        "mixed_header_table",
    }
    assert isinstance(run.routing_artifact["reason_codes"], list)


def test_engine_percentile_case_routes_non_pushdown() -> None:
    engine = MixedExecutionEngine()
    source = pd.read_csv(_usgs_path())
    run = engine.run(
        question="What is the 90th percentile of magnitude as one number?",
        dataset_path=_usgs_path(),
        expected_route_family="hybrid_or_python",
    )
    assert run.plan_valid
    assert run.route in {"hybrid", "python"}
    assert run.result is not None
    assert run.schema_correct
    ok, note = VALIDATORS["usgs_p90_magnitude"](run.result.dataframe, source)
    assert ok, note
    assert "route_" in " ".join(run.routing_artifact["reason_codes"])


def test_engine_cleaning_first_route_executes_end_to_end() -> None:
    engine = MixedExecutionEngine()
    source = pd.read_csv(_seattle_path())
    run = engine.run(
        question="Show the 10 wettest days by precipitation with date and precipitation.",
        dataset_path=_seattle_path(),
        expected_route_family="cleaning_first",
        header_confidence=0.5,
    )
    assert run.plan_valid
    assert run.route == "cleaning_first"
    assert run.execution_route == "pushdown"
    assert run.result is not None
    assert run.schema_correct
    ok, note = VALIDATORS["seattle_top10_wettest"](run.result.dataframe, source)
    assert ok, note
    assert run.routing_artifact["gate_triggered"] is True
    assert run.routing_artifact["redirect_reason"]
