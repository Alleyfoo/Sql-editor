"""Generate three linked supply-chain demo CSVs.

    products.csv          — product catalogue (product_id, product_name, ...)
    suppliers.csv         — supplier catalogue with product_id (many rows per
                            supplier, one row per product they supply)
    received_inventory.csv — purchase-order receiving records linking
                              product_id + supplier_id

Run from the repo root:
    python data/demo/generate_supply_chain.py
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
random.seed(SEED)

OUT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

CATEGORIES = ["Electronics", "Software", "Accessories", "Hardware", "Services"]

PRODUCTS_BY_CATEGORY: dict[str, list[str]] = {
    "Electronics": [
        "Laptop Pro 15", "Wireless Headset", "4K Monitor", "USB-C Hub",
        "Webcam HD", "Mechanical Keyboard", "Ergonomic Mouse",
    ],
    "Software": [
        "Analytics Platform", "Security Suite", "CRM License",
        "ERP Module", "Dev Toolchain", "BI Dashboard",
    ],
    "Accessories": [
        "Monitor Arm", "Cable Management Kit", "Laptop Stand",
        "Desk Organiser", "Screen Cleaner Kit",
    ],
    "Hardware": [
        "Server Rack Unit", "Network Switch 24-port", "SSD 1TB",
        "RAM 32GB DDR5", "NVMe Enclosure",
    ],
    "Services": [
        "Onboarding Package", "Extended Warranty", "Remote Support Hours",
        "Training Bundle",
    ],
}

SUPPLIER_NAMES = [
    "Nordic Tech Supply", "Baltic Components Ltd", "EastEuro Parts",
    "TechLink GmbH", "ScandiSource AB", "Iberian Electronics SA",
    "Alpine IT Supplies", "Adriatic Hardware", "Benelux Tech BV",
    "Celtic Systems Ltd",
]

COUNTRIES = [
    "Finland", "Germany", "Estonia", "Sweden", "Poland",
    "Spain", "Austria", "Croatia", "Netherlands", "Ireland",
]

WAREHOUSES = ["Helsinki WH", "Berlin WH", "Tallinn WH", "Warsaw WH", "Madrid WH"]

PAYMENT_TERMS = ["Net 30", "Net 60", "Net 90", "2/10 Net 30"]

# ---------------------------------------------------------------------------
# Build products
# ---------------------------------------------------------------------------

products: list[dict] = []
pid = 1001
for category, names in PRODUCTS_BY_CATEGORY.items():
    for name in names:
        unit_cost = round(random.uniform(15, 800), 2)
        products.append({
            "product_id": f"P{pid}",
            "product_name": name,
            "category": category,
            "unit_cost": unit_cost,
            "unit_price": round(unit_cost * random.uniform(1.3, 2.8), 2),
            "weight_kg": round(random.uniform(0.1, 12.0), 2),
            "is_active": random.choices([1, 0], weights=[90, 10])[0],
        })
        pid += 1

product_ids = [p["product_id"] for p in products]

# ---------------------------------------------------------------------------
# Build suppliers  (each supplier covers 3–8 products, with product_id column)
# ---------------------------------------------------------------------------

supplier_base: list[dict] = []
for i, (name, country) in enumerate(zip(SUPPLIER_NAMES, COUNTRIES)):
    supplier_base.append({
        "supplier_id": f"S{101 + i}",
        "supplier_name": name,
        "country": country,
        "rating": round(random.uniform(2.5, 5.0), 1),
        "lead_time_days": random.randint(3, 45),
        "payment_terms": random.choice(PAYMENT_TERMS),
        "on_time_delivery_pct": round(random.uniform(70, 99), 1),
    })

# Expand: one row per (supplier, product) pair they supply
suppliers: list[dict] = []
supplier_product_pairs: list[tuple[str, str]] = []

for sb in supplier_base:
    n_products = random.randint(3, 8)
    supplied = random.sample(product_ids, k=min(n_products, len(product_ids)))
    for pid_val in supplied:
        row = dict(sb)
        row["product_id"] = pid_val
        suppliers.append(row)
        supplier_product_pairs.append((sb["supplier_id"], pid_val))

# ---------------------------------------------------------------------------
# Build received_inventory  (~800 receiving records over 2 years)
# ---------------------------------------------------------------------------

start_date = date(2023, 1, 1)
end_date = date(2025, 12, 31)
span_days = (end_date - start_date).days

received: list[dict] = []
po_counter = 5001

for _ in range(800):
    supplier_id, pid_val = random.choice(supplier_product_pairs)
    received_date = start_date + timedelta(days=random.randint(0, span_days))
    quantity = random.randint(1, 200)
    # Find unit_cost from products
    unit_cost = next(p["unit_cost"] for p in products if p["product_id"] == pid_val)
    received.append({
        "po_id": f"PO{po_counter}",
        "product_id": pid_val,
        "supplier_id": supplier_id,
        "received_date": received_date.isoformat(),
        "quantity_received": quantity,
        "unit_cost": unit_cost,
        "total_cost": round(quantity * unit_cost, 2),
        "warehouse": random.choice(WAREHOUSES),
        "quality_pass": random.choices([1, 0], weights=[92, 8])[0],
    })
    po_counter += 1

received.sort(key=lambda r: r["received_date"])

# ---------------------------------------------------------------------------
# Write CSVs
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} rows -> {path}")


write_csv(OUT / "products.csv", products)
write_csv(OUT / "suppliers.csv", suppliers)
write_csv(OUT / "received_inventory.csv", received)

print("\nExample JOIN questions:")
print("  * Which suppliers deliver products in the Electronics category?")
print("    JOIN suppliers ON product_id, JOIN products ON product_id, WHERE category = 'Electronics'")
print("  * Total received inventory value by warehouse this year?")
print("    received_inventory GROUP BY warehouse, SUM(total_cost)")
print("  * Which suppliers have below-average on-time delivery and high lead times?")
print("    suppliers WHERE on_time_delivery_pct < AVG, ORDER BY lead_time_days DESC")
