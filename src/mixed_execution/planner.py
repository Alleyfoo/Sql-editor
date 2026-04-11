from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .logical_plan import (
    LogicalPlan,
    OutputColumn,
    PlanAggregate,
    PlanFilter,
    PlanOrder,
    PlanSource,
    PostProcessingStep,
)


def _source(name: str) -> PlanSource:
    return PlanSource(kind="table_or_file", name=name)


def _out(name: str, col_type: str) -> OutputColumn:
    return OutputColumn(name=name, type=col_type)


def _find_col(schema: Dict[str, str], tokens: List[str]) -> Optional[str]:
    low_tokens = [t.lower() for t in tokens]
    for col in schema:
        name = col.lower()
        if all(t in name for t in low_tokens):
            return col
    return None


def _must_col(schema: Dict[str, str], tokens: List[str], label: str) -> str:
    found = _find_col(schema, tokens)
    if found is None:
        raise ValueError(f"missing column for {label}: tokens={tokens!r}")
    return found


@dataclass(frozen=True)
class PlanIntent:
    operation_id: str
    plan: LogicalPlan


class LogicalPlanner:
    """Converts NL to a typed LogicalPlan without executing anything."""

    def __init__(self) -> None:
        self._rules: List[Tuple[re.Pattern[str], Callable[[str, Dict[str, str], str], PlanIntent]]] = [
            (re.compile(r"\b(10|ten)\s+strongest\b.*\bearthquake", re.IGNORECASE), self._plan_usgs_top10),
            (re.compile(r"\baverage magnitude by magtype\b", re.IGNORECASE), self._plan_usgs_avg_magtype),
            (re.compile(r"\blast 7 days\b.*\bearthquakes per day\b|\bhow many earthquakes per day\b.*\blast 7 days\b", re.IGNORECASE), self._plan_usgs_last7_counts),
            (re.compile(r"\b90th percentile\b.*\bmagnitude\b", re.IGNORECASE), self._plan_usgs_p90),
            (re.compile(r"\b(rolling|moving)\b.*\b7[- ]day\b.*\bearthquake\b|\b7[- ]day\b.*\b(rolling|moving)\b", re.IGNORECASE), self._plan_usgs_rolling7),
            (re.compile(r"\baverage temp[_ ]?max by weather\b", re.IGNORECASE), self._plan_seattle_avg_temp),
            (re.compile(r"\b(10|ten)\s+wettest\b", re.IGNORECASE), self._plan_seattle_top10_wettest),
            (re.compile(r"\brain(y)? days per month\b", re.IGNORECASE), self._plan_seattle_rain_days),
            (re.compile(r"\b30[- ]day\b.*\brolling\b.*\bprecipitation\b", re.IGNORECASE), self._plan_seattle_roll30),
        ]

    def plan(self, question: str, source_schema: Dict[str, str], source_name: str = "data") -> PlanIntent:
        text = (question or "").strip()
        if not text:
            return PlanIntent(
                operation_id="empty_request",
                plan=LogicalPlan(
                    source=_source(source_name),
                    projection=[],
                    filters=[],
                    group_by=[],
                    aggregates=[],
                    order_by=[],
                    post_processing=[],
                    expected_output_schema=[],
                    metadata={"fallback_used": True, "fallback_reason": "empty_question"},
                ),
            )

        for pattern, builder in self._rules:
            if pattern.search(text):
                return builder(text, source_schema, source_name)

        # Safe fallback: deterministic projection-only plan, no hallucinated columns.
        projection = list(source_schema.keys())[:5]
        expected = [_out(c, source_schema.get(c, "text")) for c in projection]
        return PlanIntent(
            operation_id="projection_fallback",
            plan=LogicalPlan(
                source=_source(source_name),
                projection=projection,
                filters=[],
                group_by=[],
                aggregates=[],
                order_by=[],
                post_processing=[],
                expected_output_schema=expected,
                metadata={"fallback_used": True, "fallback_reason": "no_rule_match"},
            ),
        )

    def _plan_usgs_top10(self, _: str, schema: Dict[str, str], source_name: str) -> PlanIntent:
        time_col = _must_col(schema, ["time"], "time")
        place_col = _must_col(schema, ["place"], "place")
        mag_col = _must_col(schema, ["mag"], "mag")
        plan = LogicalPlan(
            source=_source(source_name),
            projection=[time_col, place_col, mag_col],
            filters=[PlanFilter(column=mag_col, op="IS NOT NULL")],
            group_by=[],
            aggregates=[],
            order_by=[PlanOrder(expr=mag_col, direction="desc")],
            limit=10,
            post_processing=[],
            expected_output_schema=[
                _out(time_col, schema[time_col]),
                _out(place_col, schema[place_col]),
                _out(mag_col, "float"),
            ],
            metadata={"fallback_used": False},
        )
        return PlanIntent("usgs_top10_strongest", plan)

    def _plan_usgs_avg_magtype(self, _: str, schema: Dict[str, str], source_name: str) -> PlanIntent:
        magtype_col = _must_col(schema, ["magtype"], "magType")
        mag_col = _must_col(schema, ["mag"], "mag")
        plan = LogicalPlan(
            source=_source(source_name),
            projection=[magtype_col],
            filters=[PlanFilter(column=mag_col, op="IS NOT NULL")],
            group_by=[magtype_col],
            aggregates=[PlanAggregate(fn="avg", column=mag_col, alias="avg_mag")],
            order_by=[PlanOrder(expr="avg_mag", direction="desc")],
            limit=10,
            post_processing=[],
            expected_output_schema=[_out(magtype_col, schema[magtype_col]), _out("avg_mag", "float")],
            metadata={"fallback_used": False},
        )
        return PlanIntent("usgs_avg_magtype_top10", plan)

    def _plan_usgs_last7_counts(self, _: str, schema: Dict[str, str], source_name: str) -> PlanIntent:
        time_col = _must_col(schema, ["time"], "time")
        plan = LogicalPlan(
            source=_source(source_name),
            projection=[],
            filters=[PlanFilter(column=time_col, op=">=", value="NOW_MINUS_7_DAYS")],
            group_by=[f"date({time_col})"],
            aggregates=[PlanAggregate(fn="count", column="*", alias="daily_count")],
            order_by=[PlanOrder(expr=f"date({time_col})", direction="asc")],
            post_processing=[],
            expected_output_schema=[_out("date", "date"), _out("daily_count", "integer")],
            metadata={"fallback_used": False},
        )
        return PlanIntent("usgs_count_per_day_last7", plan)

    def _plan_usgs_p90(self, _: str, schema: Dict[str, str], source_name: str) -> PlanIntent:
        mag_col = _must_col(schema, ["mag"], "mag")
        plan = LogicalPlan(
            source=_source(source_name),
            projection=[mag_col],
            filters=[PlanFilter(column=mag_col, op="IS NOT NULL")],
            group_by=[],
            aggregates=[],
            order_by=[],
            post_processing=[PostProcessingStep(kind="percentile", params={"column": mag_col, "q": 0.9})],
            expected_output_schema=[_out("p90_magnitude", "float")],
            metadata={"fallback_used": False},
        )
        return PlanIntent("usgs_p90_magnitude", plan)

    def _plan_usgs_rolling7(self, _: str, schema: Dict[str, str], source_name: str) -> PlanIntent:
        time_col = _must_col(schema, ["time"], "time")
        plan = LogicalPlan(
            source=_source(source_name),
            projection=[],
            filters=[PlanFilter(column=time_col, op=">=", value="NOW_MINUS_14_DAYS")],
            group_by=[f"date({time_col})"],
            aggregates=[PlanAggregate(fn="count", column="*", alias="daily_count")],
            order_by=[PlanOrder(expr=f"date({time_col})", direction="asc")],
            post_processing=[
                PostProcessingStep(
                    kind="rolling_mean",
                    params={"input_column": "daily_count", "window": 7, "output_column": "rolling_daily_count"},
                )
            ],
            expected_output_schema=[_out("day", "date"), _out("rolling_daily_count", "float")],
            metadata={"fallback_used": False},
        )
        return PlanIntent("usgs_rolling7_daily_counts", plan)

    def _plan_seattle_avg_temp(self, _: str, schema: Dict[str, str], source_name: str) -> PlanIntent:
        weather_col = _must_col(schema, ["weather"], "weather")
        temp_col = _must_col(schema, ["temp", "max"], "temp_max")
        plan = LogicalPlan(
            source=_source(source_name),
            projection=[weather_col],
            filters=[PlanFilter(column=temp_col, op="IS NOT NULL")],
            group_by=[weather_col],
            aggregates=[PlanAggregate(fn="avg", column=temp_col, alias="avg_temp_max")],
            order_by=[PlanOrder(expr="avg_temp_max", direction="desc")],
            post_processing=[],
            expected_output_schema=[_out(weather_col, schema[weather_col]), _out("avg_temp_max", "float")],
            metadata={"fallback_used": False},
        )
        return PlanIntent("seattle_avg_tempmax_by_weather", plan)

    def _plan_seattle_top10_wettest(self, _: str, schema: Dict[str, str], source_name: str) -> PlanIntent:
        date_col = _must_col(schema, ["date"], "date")
        precip_col = _must_col(schema, ["precip"], "precipitation")
        plan = LogicalPlan(
            source=_source(source_name),
            projection=[date_col, precip_col],
            filters=[PlanFilter(column=precip_col, op="IS NOT NULL")],
            group_by=[],
            aggregates=[],
            order_by=[PlanOrder(expr=precip_col, direction="desc")],
            limit=10,
            post_processing=[],
            expected_output_schema=[_out(date_col, "date"), _out(precip_col, "float")],
            metadata={"fallback_used": False},
        )
        return PlanIntent("seattle_top10_wettest", plan)

    def _plan_seattle_rain_days(self, _: str, schema: Dict[str, str], source_name: str) -> PlanIntent:
        date_col = _must_col(schema, ["date"], "date")
        weather_col = _must_col(schema, ["weather"], "weather")
        plan = LogicalPlan(
            source=_source(source_name),
            projection=[],
            filters=[PlanFilter(column=weather_col, op="=", value="rain")],
            group_by=[f"date({date_col})"],
            aggregates=[PlanAggregate(fn="count", column="*", alias="rain_days_daily")],
            order_by=[PlanOrder(expr=f"date({date_col})", direction="asc")],
            post_processing=[],
            expected_output_schema=[_out("day", "date"), _out("rain_days_daily", "integer")],
            metadata={"fallback_used": False},
        )
        return PlanIntent("seattle_rain_days_per_month", plan)

    def _plan_seattle_roll30(self, _: str, schema: Dict[str, str], source_name: str) -> PlanIntent:
        date_col = _must_col(schema, ["date"], "date")
        precip_col = _must_col(schema, ["precip"], "precipitation")
        plan = LogicalPlan(
            source=_source(source_name),
            projection=[date_col, precip_col],
            filters=[],
            group_by=[],
            aggregates=[],
            order_by=[PlanOrder(expr=date_col, direction="asc")],
            post_processing=[
                PostProcessingStep(
                    kind="rolling_mean",
                    params={"input_column": precip_col, "window": 30, "output_column": "rolling_precipitation"},
                )
            ],
            expected_output_schema=[_out(date_col, "date"), _out("rolling_precipitation", "float")],
            metadata={"fallback_used": False},
        )
        return PlanIntent("seattle_rolling30_precip", plan)
