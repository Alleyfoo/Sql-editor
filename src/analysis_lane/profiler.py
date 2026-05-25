from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .models import (
    AnalysisProfile,
    CategorySummary,
    ColumnSummary,
    DateCoverage,
    Hint,
    MetricSummary,
)


def _detect_date_columns(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            out.append(str(col))
            continue
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.8:
            out.append(str(col))
    return out


def build_analysis_profile(df: pd.DataFrame, schema: Dict[str, str]) -> AnalysisProfile:
    dimensions = [c for c, t in schema.items() if t == "text"]
    metrics = [c for c, t in schema.items() if t in {"numeric", "integer", "float"}]
    date_cols = [c for c, t in schema.items() if t == "date"]
    if not date_cols:
        date_cols = _detect_date_columns(df)

    nulls: List[ColumnSummary] = []
    for col in df.columns:
        col_name = str(col)
        series = df[col]
        null_count = int(series.isna().sum())
        non_null = int(series.notna().sum())
        null_rate = float(null_count / len(series)) if len(series) else 0.0
        nulls.append(
            ColumnSummary(
                name=col_name,
                type=schema.get(col_name, "text"),
                non_null=non_null,
                null_count=null_count,
                null_rate=round(null_rate, 6),
            )
        )

    summary_stats: List[MetricSummary] = []
    for col in metrics:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            summary_stats.append(
                MetricSummary(column=col, count=0, mean=None, std=None, min=None, p50=None, p90=None, max=None)
            )
            continue
        summary_stats.append(
            MetricSummary(
                column=col,
                count=int(values.count()),
                mean=float(values.mean()),
                std=float(values.std(ddof=0)),
                min=float(values.min()),
                p50=float(values.quantile(0.5)),
                p90=float(values.quantile(0.9)),
                max=float(values.max()),
            )
        )

    coverage: List[DateCoverage] = []
    for col in date_cols:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce").dropna()
        if parsed.empty:
            continue
        days = parsed.dt.strftime("%Y-%m-%d")
        coverage.append(
            DateCoverage(
                column=col,
                min_date=str(days.min()),
                max_date=str(days.max()),
                periods=int(days.nunique()),
            )
        )

    top_categories: List[CategorySummary] = []
    for col in dimensions:
        if col not in df.columns:
            continue
        counts = df[col].astype(str).value_counts(dropna=False).head(5)
        top_categories.append(
            CategorySummary(
                column=col,
                top_values=[
                    {"value": str(idx), "count": int(val)}
                    for idx, val in counts.items()
                ],
            )
        )

    trend_hints: List[Hint] = []
    if coverage and metrics:
        date_col = coverage[0].column
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        for metric in metrics[:2]:
            if metric not in df.columns:
                continue
            series = pd.to_numeric(df[metric], errors="coerce")
            frame = pd.DataFrame({"t": parsed, "m": series}).dropna()
            if len(frame) < 5:
                continue
            frame = frame.sort_values("t")
            x = (frame["t"].astype("int64") // 10**9).astype("float64")
            y = frame["m"].astype("float64")
            # deterministic slope using first/last points (robust for small fixtures)
            slope = float((y.iloc[-1] - y.iloc[0]) / max(1.0, x.iloc[-1] - x.iloc[0]))
            direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"
            trend_hints.append(Hint(kind="trend", column=metric, message=f"{metric} shows {direction} movement"))

    anomaly_hints: List[Hint] = []
    for metric in metrics[:3]:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if len(values) < 10:
            continue
        std = float(values.std(ddof=0))
        if std == 0:
            continue
        z = (values - float(values.mean())) / std
        outliers = int((z.abs() >= 3.0).sum())
        if outliers > 0:
            anomaly_hints.append(Hint(kind="anomaly", column=metric, message=f"{outliers} high-zscore points in {metric}"))

    return AnalysisProfile(
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        dimensions=dimensions,
        metrics=metrics,
        nulls=nulls,
        summary_stats=summary_stats,
        date_coverage=coverage,
        top_categories=top_categories,
        trend_hints=trend_hints,
        anomaly_hints=anomaly_hints,
    )
