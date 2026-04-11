# Mixed Execution Route Summary (Open-Data Cases)

This is the recommended route family per current benchmark case.

| Case ID | Recommended Route | Why |
|---|---|---|
| `usgs_top10_strongest` | `pushdown` | top-k sort + projection are cheap and safe in pushdown |
| `usgs_avg_magtype_top10` | `pushdown` | group-by + avg + order-by are native pushdown operations |
| `usgs_count_per_day_last7` | `pushdown` | selective date filter + grouped count; no advanced post-processing |
| `usgs_p90_magnitude` | `python` | percentile is analytics-heavy and safer in Python |
| `usgs_rolling7_daily_counts` | `hybrid` | grouped daily counts in pushdown, rolling window in Python |
| `seattle_avg_tempmax_by_weather` | `pushdown` | group-by + avg + order-by are native pushdown operations |
| `seattle_top10_wettest` | `pushdown` | top-k sort + projection are native pushdown operations |
| `seattle_rain_days_per_month` | `pushdown` | filtered grouped counts are pushdown-friendly |
| `seattle_rolling30_precip` | `hybrid` | projection/order in pushdown, rolling window in Python |

Route categories:

- `pushdown`: SQL/data-source pushdown only.
- `hybrid`: pushdown reduction first, Python post-processing second.
- `python`: Python analytics execution.
- `cleaning_first`: only when header/schema confidence is low.

