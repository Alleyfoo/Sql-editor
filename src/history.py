"""Query history logger.

Each successfully executed query is appended as a JSON line to
``artifacts/query_history.jsonl``:

    {"ts": "2026-04-10T12:34:56+00:00", "sql": "SELECT ...", "rows": 42}

Failed executions are surfaced in the UI only; they are not logged.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_HISTORY_PATH = Path("artifacts") / "query_history.jsonl"


def log_query(sql: str, row_count: int, path: Path | str = DEFAULT_HISTORY_PATH) -> None:
    """Append a single query record to the history file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sql": sql,
        "rows": int(row_count),
    }
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


__all__ = ["log_query", "DEFAULT_HISTORY_PATH"]
