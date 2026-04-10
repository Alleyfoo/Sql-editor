from __future__ import annotations

from src.phase05_slice import detect_header_row


def test_detect_header_row_displaced() -> None:
    rows = [
        ["Monthly Export", "", "", ""],
        ["Generated 2026-04-10", "", "", ""],
        ["Region", "Product", "Revenue", "Order Date"],
        ["North", "Widget A", "1200", "2026-04-01"],
        ["South", "Widget B", "900", "2026-04-02"],
    ]
    idx, confidence = detect_header_row(rows)
    assert idx == 2
    assert 0.0 <= confidence <= 1.0


def test_detect_header_row_shifted_right() -> None:
    rows = [
        ["", "", "", "Account ID", "Country", "Balance"],
        ["", "", "", "1", "FI", "12.3"],
        ["", "", "", "2", "SE", "20.0"],
    ]
    idx, _confidence = detect_header_row(rows)
    assert idx == 0

