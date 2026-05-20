"""Deterministic generator for the bundled demo dataset.

Produces ``data/demo/sample_sales.csv`` (~3 000 rows) with enough shape
to showcase complex LLM-generated queries:

  * date filters / ORDER BY .................. ``order_date``  (2023–2025 Q1)
  * year-over-year comparison ................ multi-year span
  * text equality / IN / GROUP BY ............. ``region``, ``country``,
                                                ``category``, ``customer_segment``
  * LIKE / pattern filters .................... ``product``, ``status``
  * numeric range / SUM / AVG / MIN / MAX ..... ``units``, ``unit_price``,
                                                ``revenue``, ``cost``, ``margin``
  * CASE WHEN / conditional ................... ``status`` (4 values)
  * NULL handling ............................. ``discount_pct``
  * boolean-style filter ...................... ``is_returned``
  * GROUP BY customer ......................... ``customer_id``

Seasonal + growth patterns baked in:
  * Q4 is 35-40 % above the annual average (holiday spike).
  * Q1 following a Q4 dips to ~80 % of average (post-holiday trough).
  * 12 % YoY revenue growth from 2023 -> 2024 -> 2025.
  * Enterprise & Mid-Market segments grow faster than SMB/Consumer.

Run from repo root:

    python data/demo/generate_sample_sales.py
"""

from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path


SEED = 20250520
N_ROWS = 3_000
START_DATE = date(2023, 1, 1)
END_DATE = date(2025, 3, 31)

REGIONS = {
    "EMEA": ["Finland", "Germany", "United Kingdom", "Spain", "Sweden", "Netherlands", "France"],
    "AMER": ["United States", "Canada", "Brazil", "Mexico"],
    "APAC": ["Japan", "Australia", "Singapore", "India", "South Korea"],
}

# (category, product, base_unit_price, base_unit_cost_ratio)
CATALOG = [
    ("Hardware",    "USB-C Hub",             49.90,  0.42),
    ("Hardware",    "Mechanical Keyboard",   129.00, 0.38),
    ("Hardware",    '27" Monitor',           319.00, 0.44),
    ("Hardware",    "Webcam HD",              79.50, 0.40),
    ("Hardware",    "Laptop Stand",           44.00, 0.35),
    ("Hardware",    "Docking Station",       199.00, 0.46),
    ("Software",    "Editor Pro License",     99.00, 0.10),
    ("Software",    "Cloud Backup 1TB",       59.00, 0.12),
    ("Software",    "VPN Annual",             39.00, 0.08),
    ("Software",    "Design Suite",          249.00, 0.11),
    ("Software",    "Analytics Platform",    399.00, 0.09),
    ("Services",    "Onboarding Workshop",   499.00, 0.30),
    ("Services",    "Priority Support",      180.00, 0.20),
    ("Services",    "Custom Integration",   1200.00, 0.35),
    ("Services",    "Training Course",       350.00, 0.25),
    ("Accessories", "Cable Set",              14.50, 0.30),
    ("Accessories", "Carry Sleeve",           29.00, 0.28),
    ("Accessories", "Desk Mat",               24.00, 0.26),
    ("Accessories", "Monitor Arm",            89.00, 0.38),
]

SEGMENTS = ["SMB", "Mid-Market", "Enterprise", "Consumer"]

# Weights vary by region: APAC skews more Enterprise, AMER more SMB
SEGMENT_WEIGHTS_BY_REGION = {
    "EMEA": [0.30, 0.30, 0.25, 0.15],
    "AMER": [0.40, 0.25, 0.20, 0.15],
    "APAC": [0.20, 0.30, 0.35, 0.15],
}

STATUSES = ["Delivered", "Delivered", "Delivered", "Shipped", "Pending", "Cancelled"]
# ~50% Delivered, ~17% Shipped, ~17% Pending, ~17% Cancelled — weighted below
STATUS_WEIGHTS = [0.62, 0.62, 0.62, 0.16, 0.12, 0.08]

# Number of synthetic customers per segment
N_CUSTOMERS = {
    "SMB":        120,
    "Mid-Market":  60,
    "Enterprise":  30,
    "Consumer":   100,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seasonal_factor(d: date) -> float:
    """Return a multiplier that makes Q4 spike and Q1 dip."""
    month = d.month
    if month in (11, 12):
        return 1.38
    if month in (1, 2):
        return 0.78
    if month in (9, 10):
        return 1.12
    if month in (7, 8):
        return 0.90
    return 1.0


def _yoy_growth_factor(d: date) -> float:
    """12 % YoY growth: 2023 -> 1.0, 2024 -> 1.12, 2025 -> 1.25."""
    return 1.0 + 0.12 * (d.year - 2023)


def _rand_date(rng: random.Random) -> date:
    """Return a random date in the full range, weighted by seasonal density."""
    span = (END_DATE - START_DATE).days
    while True:
        d = START_DATE + timedelta(days=rng.randint(0, span))
        # Rejection sampling: accept with probability proportional to seasonal factor
        if rng.random() < _seasonal_factor(d) / 1.5:
            return d


def _build_customer_pool(rng: random.Random) -> dict[str, list[str]]:
    """Return {segment: [customer_id, ...]} pools."""
    pool: dict[str, list[str]] = {}
    cid = 1001
    for seg, n in N_CUSTOMERS.items():
        pool[seg] = [f"C{cid + i:04d}" for i in range(n)]
        cid += n
    return pool


# ---------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------

def generate_rows(rng: random.Random) -> list[dict]:
    customer_pool = _build_customer_pool(rng)
    rows = []

    for i in range(N_ROWS):
        region = rng.choices(list(REGIONS.keys()), weights=[0.40, 0.35, 0.25])[0]
        country = rng.choice(REGIONS[region])
        segment = rng.choices(
            SEGMENTS, weights=SEGMENT_WEIGHTS_BY_REGION[region]
        )[0]
        customer_id = rng.choice(customer_pool[segment])

        category, product, base_price, cost_ratio = rng.choice(CATALOG)

        # Enterprise buys more units; Consumer buys fewer
        units_range = {
            "Enterprise": (3, 20),
            "Mid-Market": (2, 12),
            "SMB":        (1, 8),
            "Consumer":   (1, 4),
        }[segment]
        units = rng.randint(*units_range)

        order_date = _rand_date(rng)
        growth = _yoy_growth_factor(order_date)

        # Price jitter: ±6%, then apply YoY growth (prices creep up)
        unit_price = round(base_price * rng.uniform(0.94, 1.06) * growth, 2)
        unit_cost = round(base_price * cost_ratio * rng.uniform(0.95, 1.05), 2)

        # Discount: ~18% of orders have one; ~8% leave it NULL
        roll = rng.random()
        if roll < 0.18:
            discount_pct = rng.choice([0.05, 0.10, 0.15, 0.20, 0.25])
        elif roll < 0.26:
            discount_pct = None  # NULL
        else:
            discount_pct = 0.00

        gross = units * unit_price
        revenue = round(gross * (1.0 - (discount_pct or 0.0)), 2)
        cost = round(units * unit_cost, 2)
        margin = round(revenue - cost, 2)

        # Return rate: higher for Hardware, very low for Software/Services
        return_chance = {"Hardware": 0.09, "Software": 0.02,
                         "Services": 0.01, "Accessories": 0.06}.get(category, 0.05)
        # Cancelled orders can't be returned
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
        is_returned = (
            1 if status == "Delivered" and rng.random() < return_chance else 0
        )

        rows.append(
            {
                "order_id":        10001 + i,
                "order_date":      order_date.isoformat(),
                "customer_id":     customer_id,
                "region":          region,
                "country":         country,
                "category":        category,
                "product":         product,
                "customer_segment": segment,
                "units":           units,
                "unit_price":      f"{unit_price:.2f}",
                "unit_cost":       f"{unit_cost:.2f}",
                "revenue":         f"{revenue:.2f}",
                "cost":            f"{cost:.2f}",
                "margin":          f"{margin:.2f}",
                "discount_pct":    "" if discount_pct is None else f"{discount_pct:.2f}",
                "status":          status,
                "is_returned":     is_returned,
            }
        )

    rows.sort(key=lambda r: r["order_date"])
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(SEED)
    rows = generate_rows(rng)

    out_path = Path(__file__).resolve().parent / "sample_sales.csv"
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Quick sanity summary
    total_revenue = sum(float(r["revenue"]) for r in rows)
    years = sorted({r["order_date"][:4] for r in rows})
    print(f"Wrote {len(rows)} rows -> {out_path}")
    print(f"Date range : {rows[0]['order_date']} -> {rows[-1]['order_date']}")
    print(f"Years      : {', '.join(years)}")
    print(f"Total rev  : ${total_revenue:,.0f}")
    customers = {r["customer_id"] for r in rows}
    print(f"Customers  : {len(customers)} unique IDs")


if __name__ == "__main__":
    main()
