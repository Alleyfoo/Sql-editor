"""Generate deterministic dirty Excel/CSV fixtures for cleaning capability eval."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent / "fixtures"


def _write_xlsx(name: str, rows: list[list[object]]) -> None:
    pd.DataFrame(rows).to_excel(BASE / name, header=False, index=False)


def _write_csv(name: str, rows: list[list[object]]) -> None:
    pd.DataFrame(rows).to_csv(BASE / name, header=False, index=False)


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)

    # Existing baseline fixtures.
    _write_xlsx(
        "sales_displaced_header.xlsx",
        [
            ["Monthly Sales Export", "", "", ""],
            ["Generated: 2026-01-07", "", "", ""],
            ["Region", "Product", "Revenue", "Order Date"],
            ["North", "Widget A", 1200, "2026-01-01"],
            ["South", "Widget B", 980, "2026-01-02"],
        ],
    )
    _write_xlsx(
        "audit_banner_offset.xlsx",
        [
            ["ACME Internal Report", "", "", ""],
            ["Do not edit manually", "", "", ""],
            ["", "", "", ""],
            ["customer_id", "issue_type", "status", "opened_at"],
            [101, "billing", "open", "2026-03-01"],
            [102, "login", "closed", "2026-03-02"],
        ],
    )
    _write_xlsx(
        "leading_empty_cells.xlsx",
        [
            ["Extracted from ERP", "", "", "", ""],
            ["", "", "ID", "Country", "Amount"],
            ["", "", 1, "FI", 10.5],
            ["", "", 2, "SE", 20.0],
        ],
    )
    _write_csv(
        "dirty_export_reference.csv",
        [
            ["Support Dashboard Export", "", "", ""],
            ["Generated 2026-03-30", "", "", ""],
            ["Team", "Owner", "Tickets Open", "Updated At"],
            ["Platform", "Ari", 14, "2026-03-29"],
            ["Data", "Lea", 5, "2026-03-29"],
        ],
    )

    # Harder fixtures.
    _write_xlsx(
        "marketing_preface_4rows.xlsx",
        [
            ["Marketing KPI Dump", "", "", "", ""],
            ["Region: EU", "", "", "", ""],
            ["Confidential", "", "", "", ""],
            ["Generated 2026-02-11", "", "", "", ""],
            ["Campaign", "Spend", "Clicks", "CTR %", "Launch Date"],
            ["Spring Promo", 2500, 12000, 0.043, "2026-02-01"],
        ],
    )
    _write_xlsx(
        "hr_double_blank_then_header.xlsx",
        [
            ["HR Export", "", "", ""],
            ["", "", "", ""],
            ["", "", "", ""],
            ["Employee ID", "Team", "Salary EUR", "Start Date"],
            [1001, "Data", 52000, "2024-01-02"],
        ],
    )
    _write_xlsx(
        "finance_symbols_header.xlsx",
        [
            ["Finance Summary", "", "", ""],
            ["Week 12", "", "", ""],
            ["Cost (USD)", "Gross Margin %", "Net Profit $", "Posting Date"],
            [1020.5, 33.2, 440.1, "2026-03-19"],
        ],
    )
    _write_xlsx(
        "header_with_units.xlsx",
        [
            ["Factory Sensor Export", "", "", ""],
            ["Sensor readings", "", "", ""],
            ["Temp C", "Pressure kPa", "Humidity %", "Recorded At"],
            [23.1, 101.3, 40.0, "2026-01-01 10:00"],
        ],
    )
    _write_xlsx(
        "sparse_header_cells.xlsx",
        [
            ["Site health report", "", "", "", ""],
            ["Site ID", "", "Status", "", "Updated At"],
            [2001, "", "ok", "", "2026-04-01"],
        ],
    )
    _write_xlsx(
        "two_tables_first_one.xlsx",
        [
            ["North Ops Table", "", "", ""],
            ["Week", "Owner", "Open Tickets", "Closed Tickets"],
            [10, "Ari", 12, 8],
            [11, "Lea", 7, 9],
            ["", "", "", ""],
            ["Archive table", "", "", ""],
            ["Month", "Owner", "Escalations", "Resolved"],
            ["Jan", "Mia", 2, 5],
        ],
    )
    _write_xlsx(
        "numbered_preamble.xlsx",
        [
            ["1) exported by ERP", "", "", "", ""],
            ["2) values in local currency", "", "", "", ""],
            ["3) no edits", "", "", "", ""],
            ["4) approved by finance", "", "", "", ""],
            ["5) generated 2026-03-30", "", "", "", ""],
            ["SKU", "Category", "Units Sold", "Revenue USD", "Updated At"],
            ["A-1", "Hardware", 42, 4900.5, "2026-03-30"],
        ],
    )
    _write_xlsx(
        "shifted_right_3cols.xlsx",
        [
            ["", "", "", "Account ID", "Country", "Balance"],
            ["", "", "", 5001, "FI", 1200.0],
            ["", "", "", 5002, "SE", 980.0],
        ],
    )
    _write_xlsx(
        "mixed_language_ascii_headers.xlsx",
        [
            ["Myynti raportti", "", "", ""],
            ["Sisainen kaytto", "", "", ""],
            ["Region", "Myynti EUR", "Paivamaara", "Tuote Ryhma"],
            ["North", 1200, "2026-03-01", "A"],
        ],
    )
    _write_xlsx(
        "header_with_trailing_noise.xlsx",
        [
            ["Summary Export", "", "", "", "", ""],
            ["Generated now", "", "", "", "", ""],
            ["Team", "Owner", "Velocity", "Defects", "", ""],
            ["Platform", "Ari", 31, 2, "", ""],
        ],
    )
    _write_xlsx(
        "empty_row_before_header.xlsx",
        [
            ["Weekly KPI", "", "", ""],
            ["", "", "", ""],
            ["Service", "Region", "SLA Breach", "Updated At"],
            ["Checkout", "EU", 1, "2026-04-04"],
        ],
    )
    _write_csv(
        "ops_export_messy.csv",
        [
            ["Operations Queue Export", "", "", ""],
            ["Internal", "", "", ""],
            ["2026-04-01", "", "", ""],
            ["Queue", "Owner", "Pending", "Avg Wait Min"],
            ["L1", "Ari", 4, 12.3],
        ],
    )
    _write_csv(
        "inventory_banner.csv",
        [
            ["Inventory Snapshot", "", "", ""],
            ["warehouse-1", "", "", ""],
            ["Item Code", "Warehouse", "On Hand", "Reorder Point"],
            ["ITM-1", "W1", 45, 10],
        ],
    )
    _write_csv(
        "finance_offset.csv",
        [
            ["Finance Export", "", "", ""],
            ["Prepared for board", "", "", ""],
            ["Currency EUR", "", "", ""],
            ["Use with care", "", "", ""],
            ["Month", "Opex", "Capex", "Cashflow"],
            ["2026-01", 2100, 420, 380],
        ],
    )
    _write_xlsx(
        "notes_then_header.xlsx",
        [
            ["Project tracking", "", "", "", ""],
            ["notes: stage names changed", "", "", "", ""],
            ["sync: 2026-04-02", "", "", "", ""],
            ["Project", "Phase", "Owner 1", "Owner 2", "Risk Score"],
            ["Alpha", "Build", "Ari", "Lea", 3],
        ],
    )


if __name__ == "__main__":
    main()

