from __future__ import annotations

from src.marker_catalog_slice import (
    extract_parts_dataframe,
    parse_markdown_tables,
    table_to_dataframe,
)


def test_parse_markdown_tables_basic() -> None:
    md = """
## Section A

| Part nr. | Description |   |
|----------|-------------|---|
| TP1      | Lock        |   |
| TP2      | Cap         | 1 |
"""
    tables, images = parse_markdown_tables(md)
    assert len(tables) == 1
    assert images == []
    assert tables[0].heading == "Section A"


def test_table_to_dataframe_drops_empty_columns() -> None:
    md = """
| Part nr. | Description |   |
|----------|-------------|---|
| TP1      | Lock        |   |
| TP2      | Cap         |   |
"""
    tables, _ = parse_markdown_tables(md)
    df = table_to_dataframe(tables[0])
    assert "part_nr" in df.columns
    assert "description" in df.columns
    assert len(df.columns) == 2
    assert len(df) == 2


def test_extract_parts_dataframe_filters_part_rows() -> None:
    md = """
## Anti-theft

| Part nr. | Description |
|----------|-------------|
| TP0004   | Discus lock |
| note     | not a part  |
"""
    tables, _ = parse_markdown_tables(md)
    parts = extract_parts_dataframe(tables)
    assert len(parts) == 1
    assert parts.iloc[0]["part_nr"] == "TP0004"
    assert parts.iloc[0]["description"] == "Discus lock"
