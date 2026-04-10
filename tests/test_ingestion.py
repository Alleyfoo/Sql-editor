"""Tests for src.ingestion."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion import TABLE_NAME, infer_schema, load_csv


def test_infer_schema_numeric_text_date():
    df = pd.DataFrame(
        {
            "n": [1, 2, 3],
            "t": ["alpha", "beta", "gamma"],
            "d": ["2024-01-01", "2024-02-01", "2024-03-01"],
        }
    )
    schema = infer_schema(df)
    assert schema["n"] == "numeric"
    assert schema["t"] == "text"
    assert schema["d"] == "date"


def test_infer_schema_empty_column_is_text():
    df = pd.DataFrame({"x": [None, None, None]})
    assert infer_schema(df)["x"] == "text"


def test_infer_schema_mixed_mostly_text():
    df = pd.DataFrame({"x": ["a", "b", "c", "2024-01-01"]})
    assert infer_schema(df)["x"] == "text"


def test_load_csv_returns_readonly_connection(tmp_path: Path):
    csv = tmp_path / "sample.csv"
    csv.write_text("id,name,joined\n1,alice,2024-01-01\n2,bob,2024-02-01\n")

    conn, schema = load_csv(csv)
    try:
        # Schema inferred correctly.
        assert schema == {"id": "numeric", "name": "text", "joined": "date"}

        # Data readable.
        rows = list(conn.execute(f"SELECT id, name FROM {TABLE_NAME} ORDER BY id"))
        assert rows == [(1, "alice"), (2, "bob")]

        # Writes must fail — connection is read-only.
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(f"INSERT INTO {TABLE_NAME} VALUES (3, 'carol', '2024-03-01')")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(f"DROP TABLE {TABLE_NAME}")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(f"UPDATE {TABLE_NAME} SET name = 'zed'")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(f"DELETE FROM {TABLE_NAME}")
    finally:
        conn.close()


def test_load_csv_missing_file():
    with pytest.raises(FileNotFoundError):
        load_csv("/nonexistent/path/to/file.csv")
