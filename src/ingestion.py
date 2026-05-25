"""CSV ingestion: pandas -> in-memory SQLite -> read-only handle.

The table is always named ``data`` in Phase 1 (single-CSV). After the
dataframe is written, the connection is switched to read-only via
``PRAGMA query_only = ON`` and a SQLite authorizer that rejects every
non-SELECT action code. Together these enforce the spec's "no writes
ever" guarantee. (The spec suggested ``mode=ro`` via URI, but SQLite
requires an on-disk file for that flag, and the spec also forbids
writing the DB to disk — so we use the pragma + authorizer route
instead, which is SQLite-native and equivalent in effect.)

Type inference produces three buckets: ``numeric``, ``text``, ``date``.
Date detection parses object columns via ``pd.to_datetime`` and accepts
the column as a date if >= 90 % of non-null values parse successfully.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


TABLE_NAME = "data"
_DATE_THRESHOLD = 0.9

# SQLite authorizer action codes we allow on the read-only connection.
# Anything else is denied. Action codes: https://www.sqlite.org/c3ref/c_alter_table.html
_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)


def _infer_column_type(series: pd.Series) -> str:
    """Return one of 'numeric', 'date', 'text' for a pandas Series."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    ):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    # Try to parse object columns as dates. Suppress the "could not infer
    # format" warning — mixed-format fallback is an expected code path.
    non_null = series.dropna()
    if len(non_null) == 0:
        return "text"
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(
                non_null, errors="coerce", utc=False, format="mixed"
            )
    except (ValueError, TypeError):
        return "text"
    hit_rate = parsed.notna().mean()
    if hit_rate >= _DATE_THRESHOLD:
        return "date"
    return "text"


def infer_schema(df: pd.DataFrame) -> Dict[str, str]:
    """Return ``{column_name: type_bucket}`` for every column in ``df``."""
    return {str(col): _infer_column_type(df[col]) for col in df.columns}


def _readonly_authorizer(action: int, *_args) -> int:
    if action in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _make_readonly(conn: sqlite3.Connection) -> None:
    """Flip ``conn`` to read-only mode. Any subsequent write raises."""
    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(_readonly_authorizer)


def load_multiple_csvs(
    table_paths: Dict[str, "str | Path"],
) -> Tuple[sqlite3.Connection, Dict[str, Dict[str, str]]]:
    """Load several CSVs into one in-memory SQLite connection, each in a named table.

    ``table_paths`` maps SQLite table name → CSV file path.
    Returns ``(conn, tables_schema)`` where ``tables_schema`` is
    ``{table_name: {col: type}}``.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    tables_schema: Dict[str, Dict[str, str]] = {}

    for table_name, path in table_paths.items():
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, index=False, if_exists="replace")
        tables_schema[table_name] = infer_schema(df)

    conn.commit()
    _make_readonly(conn)
    return conn, tables_schema


def load_csv(path: str | Path) -> Tuple[sqlite3.Connection, Dict[str, str]]:
    """Load a CSV into an in-memory SQLite connection locked read-only.

    Returns ``(conn, schema)``. The caller owns ``conn`` and must close it.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    schema = infer_schema(df)

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql(TABLE_NAME, conn, index=False, if_exists="replace")
    conn.commit()

    _make_readonly(conn)
    return conn, schema


__all__ = ["load_csv", "load_multiple_csvs", "infer_schema", "TABLE_NAME", "_make_readonly"]
