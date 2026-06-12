from __future__ import annotations

import textwrap

from streamlit.testing.v1 import AppTest

from src.query_model import JOIN_TYPES


def _composer_join_script() -> str:
    return textwrap.dedent(
        """
        import streamlit as st
        from src.query_model import QueryModel
        from src.streamlit_app.components import composer

        st.session_state.setdefault(
            "schema",
            {
                "product_id": "numeric",
                "product_name": "text",
                "supplier_id": "numeric",
                "supplier_name": "text",
            },
        )
        st.session_state.setdefault(
            "tables",
            {
                "products": {
                    "product_id": "numeric",
                    "product_name": "text",
                    "supplier_id": "numeric",
                },
                "suppliers": {
                    "supplier_id": "numeric",
                    "supplier_name": "text",
                },
            },
        )
        st.session_state.setdefault(
            "relationships",
            [
                {
                    "left_table": "products",
                    "left_col": "supplier_id",
                    "right_table": "suppliers",
                    "right_col": "supplier_id",
                    "type": "FK",
                    "confidence": "high",
                }
            ],
        )
        st.session_state.setdefault("model", QueryModel(table="products"))
        st.session_state.setdefault(
            "join_rows",
            [
                {
                    "left_table": "products",
                    "left_col": "supplier_id",
                    "right_table": "suppliers",
                    "right_col": "supplier_id",
                    "join_type": "INNER",
                }
            ],
        )

        composer.render()
        """
    )


def test_join_composer_renders_join_type_selector() -> None:
    at = AppTest.from_string(_composer_join_script()).run()

    assert not at.exception, f"composer raised: {at.exception}"
    join_type = next(sb for sb in at.selectbox if sb.key == "join_type_0")
    assert join_type.options == list(JOIN_TYPES)
    assert join_type.value == "INNER"


def test_join_type_selector_updates_sql_preview() -> None:
    at = AppTest.from_string(_composer_join_script()).run()
    join_type = next(sb for sb in at.selectbox if sb.key == "join_type_0")

    join_type.set_value("LEFT").run()

    assert not at.exception, f"composer raised after JOIN type edit: {at.exception}"
    join_rows = at.session_state.filtered_state.get("join_rows")
    assert join_rows[0]["join_type"] == "LEFT"
    assert "LEFT JOIN" in at.session_state.filtered_state.get("last_sql", "")
