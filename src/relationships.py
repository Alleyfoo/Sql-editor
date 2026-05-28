"""Relationship detection for multi-table datasets.

Detects potential foreign key relationships between tables by:
1. Column name matching (e.g., product_id in both tables)
2. Type compatibility (both numeric or both text)

Returns a list of relationships that can be used for JOIN suggestions.
"""

from __future__ import annotations

from typing import Dict, List, Set


def detect_relationships(
    tables_schema: Dict[str, Dict[str, str]]
) -> List[Dict[str, str]]:
    """Detect potential foreign key relationships between tables.

    Args:
        tables_schema: {table_name: {col: type}} mapping

    Returns:
        List of relationship dicts with keys:
        - left_table, left_col
        - right_table, right_col
        - type: "FK" (foreign key)
        - confidence: "high" (exact name match) or "medium" (suffix match)
    """
    relationships = []
    table_names = list(tables_schema.keys())

    # Build a map of column names to tables
    col_to_tables: Dict[str, List[str]] = {}
    for table_name, schema in tables_schema.items():
        for col_name in schema.keys():
            col_to_tables.setdefault(col_name, []).append(table_name)

    # Find columns that appear in multiple tables (potential FKs)
    for col_name, tables in col_to_tables.items():
        if len(tables) < 2:
            continue

        # Check all pairs of tables that share this column
        for i, left_table in enumerate(tables):
            for right_table in tables[i + 1:]:
                left_type = tables_schema[left_table][col_name]
                right_type = tables_schema[right_table][col_name]

                # Only match if types are compatible
                if not _types_compatible(left_type, right_type):
                    continue

                # Determine confidence based on column name pattern
                confidence = _assess_confidence(col_name)

                relationships.append({
                    "left_table": left_table,
                    "left_col": col_name,
                    "right_table": right_table,
                    "right_col": col_name,
                    "type": "FK",
                    "confidence": confidence,
                })

    # Sort by confidence (high first), then by table names
    relationships.sort(
        key=lambda r: (
            0 if r["confidence"] == "high" else 1,
            r["left_table"],
            r["right_table"],
        )
    )

    return relationships


def _types_compatible(type1: str, type2: str) -> bool:
    """Check if two column types are compatible for joining."""
    # Same type is always compatible
    if type1 == type2:
        return True

    # Numeric and text can sometimes be joined (e.g., ID as string vs int)
    # But we'll be conservative and only allow exact matches for now
    return False


def _assess_confidence(col_name: str) -> str:
    """Assess confidence level based on column name pattern.

    High confidence: columns ending with _id (typical FK pattern)
    Medium confidence: other shared column names
    """
    col_lower = col_name.lower()

    # High confidence: typical FK naming patterns
    if col_lower.endswith("_id"):
        return "high"
    if col_lower.endswith("_key"):
        return "high"
    if col_lower in ("id", "key"):
        return "high"

    # Medium confidence: other shared columns
    return "medium"


def get_joinable_columns(
    relationships: List[Dict[str, str]]
) -> Dict[str, Set[str]]:
    """Build a map of which columns are joinable.

    Returns:
        {table_name: set of joinable column names}
    """
    joinable: Dict[str, Set[str]] = {}

    for rel in relationships:
        left_table = rel["left_table"]
        left_col = rel["left_col"]
        right_table = rel["right_table"]
        right_col = rel["right_col"]

        joinable.setdefault(left_table, set()).add(left_col)
        joinable.setdefault(right_table, set()).add(right_col)

    return joinable


def format_relationship_label(rel: Dict[str, str]) -> str:
    """Format a relationship as a human-readable label.

    Example: "products.product_id <-> received_inventory.product_id"
    """
    left = f"{rel['left_table']}.{rel['left_col']}"
    right = f"{rel['right_table']}.{rel['right_col']}"
    return f"{left} <-> {right}"


__all__ = [
    "detect_relationships",
    "get_joinable_columns",
    "format_relationship_label",
]
