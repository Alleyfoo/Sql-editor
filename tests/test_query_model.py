"""Tests for src.query_model."""

from __future__ import annotations

import pytest

from src.query_model import (
    Filter,
    QueryModel,
    _assert_select_only,
    quote_ident,
    quote_value,
)


def test_select_star_when_no_columns():
    sql = QueryModel().to_sql()
    assert sql == 'SELECT * FROM "data"'


def test_select_specific_columns():
    m = QueryModel(selected_columns=["a", "b"])
    assert m.to_sql() == 'SELECT "a", "b" FROM "data"'


def test_quote_ident_escapes_embedded_quotes():
    assert quote_ident('he said "hi"') == '"he said ""hi"""'


def test_quote_value_handles_text_numbers_and_null():
    assert quote_value(None) == "NULL"
    assert quote_value(42) == "42"
    assert quote_value(3.14) == "3.14"
    assert quote_value("O'Reilly") == "'O''Reilly'"


@pytest.mark.parametrize(
    "op,value,fragment",
    [
        ("=", 42, '"age" = 42'),
        ("!=", "bob", "\"name\" != 'bob'"),
        ("<", 10, '"age" < 10'),
        (">", 10, '"age" > 10'),
        ("<=", 10, '"age" <= 10'),
        (">=", 10, '"age" >= 10'),
        ("LIKE", "foo%", "\"name\" LIKE 'foo%'"),
        ("NOT LIKE", "x%", "\"name\" NOT LIKE 'x%'"),
    ],
)
def test_filter_operators(op, value, fragment):
    # Pick a column name that matches the fragment above.
    col = "age" if '"age"' in fragment else "name"
    f = Filter(column=col, operator=op, value=value)
    assert f.to_sql() == fragment


def test_filter_is_null_variants():
    assert Filter(column="a", operator="IS NULL").to_sql() == '"a" IS NULL'
    assert (
        Filter(column="a", operator="IS NOT NULL").to_sql() == '"a" IS NOT NULL'
    )


def test_filter_between():
    f = Filter(column="age", operator="BETWEEN", value=(1, 10))
    assert f.to_sql() == '"age" BETWEEN 1 AND 10'


def test_filter_between_requires_tuple():
    with pytest.raises(ValueError):
        Filter(column="age", operator="BETWEEN", value=5).to_sql()


def test_where_clause_and_or():
    m = QueryModel(
        selected_columns=["a"],
        filters=[
            Filter(column="a", operator=">", value=1),
            Filter(column="b", operator="=", value="x", logical="OR"),
            Filter(column="c", operator="IS NULL", logical="AND"),
        ],
    )
    assert m.to_sql() == (
        'SELECT "a" FROM "data" '
        'WHERE "a" > 1 OR "b" = \'x\' AND "c" IS NULL'
    )


def test_limit_clause():
    m = QueryModel(limit=100)
    assert m.to_sql() == 'SELECT * FROM "data" LIMIT 100'


def test_invalid_limit_rejected():
    with pytest.raises(ValueError):
        QueryModel(limit=-1).to_sql()


def test_sql_injection_in_filter_value_stays_safe():
    """A hostile value must not break out of its quoted literal."""
    hostile = "'; DROP TABLE data;--"
    m = QueryModel(
        selected_columns=["a"],
        filters=[Filter(column="a", operator="=", value=hostile)],
    )
    sql = m.to_sql()
    # The hostile value should be fully contained inside single quotes, with
    # the embedded quote doubled.
    assert "'''; DROP TABLE data;--'" in sql
    # And _assert_select_only must still accept the whole thing because
    # DROP lives inside a string literal.
    _assert_select_only(sql)


def test_assert_select_only_rejects_ddl():
    with pytest.raises(ValueError):
        _assert_select_only("DROP TABLE data")


def test_assert_select_only_rejects_multiple_statements():
    with pytest.raises(ValueError):
        _assert_select_only("SELECT 1; SELECT 2")


def test_assert_select_only_rejects_comment_only():
    with pytest.raises(ValueError):
        _assert_select_only("-- just a comment")


def test_assert_select_only_rejects_ddl_after_comment():
    with pytest.raises(ValueError):
        _assert_select_only("/* hi */ DELETE FROM data")
