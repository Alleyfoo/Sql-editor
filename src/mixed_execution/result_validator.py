from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .logical_plan import LogicalPlan


@dataclass(frozen=True)
class ResultValidation:
    schema_correct: bool
    errors: List[str]


def validate_result_schema(plan: LogicalPlan, result_df: pd.DataFrame) -> ResultValidation:
    errors: List[str] = []
    expected = plan.expected_output_schema
    if not expected:
        return ResultValidation(schema_correct=True, errors=[])

    present = set(result_df.columns)
    for col in expected:
        if col.name not in present:
            errors.append(f"missing expected column: {col.name!r}")

    # Keep type checks lightweight and deterministic.
    for col in expected:
        if col.name not in result_df.columns:
            continue
        series = result_df[col.name]
        if col.type in {"integer", "numeric", "float"}:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().mean() < 0.8:
                errors.append(f"column {col.name!r} expected numeric-compatible values")
        elif col.type == "date":
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() < 0.8:
                errors.append(f"column {col.name!r} expected date-compatible values")
        elif col.type == "text":
            # Any series can be represented as text; no strict constraint.
            pass

    return ResultValidation(schema_correct=not errors, errors=errors)

