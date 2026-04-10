"""Tests for src.executor."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.executor import BLOCKED_KEYWORDS, ExecutionError, execute


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE data (id INTEGER, name TEXT)")
    c.executemany(
        "INSERT INTO data VALUES (?, ?)",
        [(1, "alice"), (2, "bob"), (3, "carol")],
    )
    c.commit()
    return c


def test_happy_path_returns_dataframe(conn):
    df = execute(conn, "SELECT id, name FROM data ORDER BY id")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id", "name"]
    assert len(df) == 3


@pytest.mark.parametrize("keyword", sorted(BLOCKED_KEYWORDS))
def test_blocklist_rejects_each_banned_keyword(conn, keyword):
    with pytest.raises(ExecutionError):
        execute(conn, f"SELECT * FROM data; {keyword} TABLE data")


def test_rejects_non_select(conn):
    with pytest.raises(ExecutionError):
        execute(conn, "DROP TABLE data")


def test_wraps_sqlite_error(conn):
    with pytest.raises(ExecutionError) as ei:
        execute(conn, "SELECT nonexistent_column FROM data")
    assert "SQLite error" in str(ei.value)


def test_blocklist_ignores_keywords_inside_string_literal(conn):
    """A banned keyword inside a quoted value should NOT trip the blocklist."""
    df = execute(conn, "SELECT * FROM data WHERE name = 'DROP TABLE data'")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_rejects_multiple_statements(conn):
    with pytest.raises(ExecutionError):
        execute(conn, "SELECT 1; SELECT 2")
