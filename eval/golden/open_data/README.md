# Open-Data Benchmark Cases

This folder holds case sets for `eval/open_data_sql_vs_python_eval.py`.

## Add a new dataset track (e.g. HSY)

1. Put your CSV in `data/open_data/`.
2. Add one or more case objects in `sql_vs_python_cases.json`:

```json
{
  "id": "hsy_example_case",
  "track": "sql_fit",
  "dataset": "data/open_data/hsy_dataset.csv",
  "question": "Top 10 categories by total value.",
  "validator": "non_empty_result"
}
```

## Built-in validators

- `non_empty_result`
- `single_numeric_scalar`
- dataset-specific strict validators used by the reference USGS/Seattle cases

For strict validation on a new dataset, add a validator function in
`eval/open_data_sql_vs_python_eval.py` and reference its name in the case.

## Included HSY case pack

- `hsy_2021_04_eval_pack.json` targets `data/open_data/2021-04.csv`
  (note the dataset file name uses a dash, not a dot).
