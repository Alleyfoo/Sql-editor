"""Parse marker-pdf markdown output into queryable artifacts.

Input: marker output folder containing `catalog.md` (and optional images/meta).
Output: artifacts/phase05/<run-id>/{clean.csv,schema.json,ingest_summary.json}
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .ingestion import infer_schema
from .phase05_slice import DEFAULT_ARTIFACTS_DIR, run_query_for_run


_TABLE_SEP_RE = re.compile(r"^\|\s*[-: ]+\|")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_PART_CODE_RE = re.compile(r"^[A-Z]{1,5}\d{2,}(-[A-Z0-9]+)?$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_header(text: str, index: int) -> str:
    raw = _TAG_RE.sub("", (text or "")).strip().lower()
    raw = raw.replace("nr.", "nr")
    raw = _NON_ALNUM_RE.sub("_", raw).strip("_")
    return raw or f"col_{index+1}"


def _clean_cell(text: str) -> str:
    out = _TAG_RE.sub("", text or "")
    out = out.replace("&nbsp;", " ")
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _parse_md_row(line: str) -> List[str]:
    body = line.strip().strip("|")
    parts = [p.strip() for p in body.split("|")]
    return parts


@dataclass
class ParsedTable:
    heading: str
    headers: List[str]
    rows: List[List[str]]
    start_line: int

    @property
    def non_empty_cells(self) -> int:
        return sum(1 for row in self.rows for c in row if c)


def parse_markdown_tables(markdown: str) -> Tuple[List[ParsedTable], List[str]]:
    lines = markdown.splitlines()
    tables: List[ParsedTable] = []
    images = _IMAGE_RE.findall(markdown)
    current_heading = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("#"):
            current_heading = re.sub(r"^#+\s*", "", stripped)
            current_heading = _clean_cell(current_heading.replace("*", ""))
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if _TABLE_SEP_RE.match(next_line):
                headers = _parse_md_row(stripped)
                i += 2
                rows: List[List[str]] = []
                while i < len(lines):
                    row_line = lines[i].strip()
                    if not row_line.startswith("|"):
                        break
                    rows.append(_parse_md_row(row_line))
                    i += 1
                if rows:
                    tables.append(
                        ParsedTable(
                            heading=current_heading,
                            headers=headers,
                            rows=rows,
                            start_line=i + 1,
                        )
                    )
                continue
        i += 1
    return tables, images


def table_to_dataframe(table: ParsedTable) -> pd.DataFrame:
    width = max(len(table.headers), max((len(r) for r in table.rows), default=0))
    headers = table.headers + [""] * (width - len(table.headers))
    clean_headers = [_normalize_header(h, idx) for idx, h in enumerate(headers)]

    padded_rows: List[List[str]] = []
    for row in table.rows:
        values = row + [""] * (width - len(row))
        padded_rows.append([_clean_cell(v) for v in values])

    df = pd.DataFrame(padded_rows, columns=clean_headers)

    # Drop fully empty columns.
    drop_cols = []
    for col in df.columns:
        if (df[col].astype(str).str.strip() == "").all():
            drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # If the first column is empty for a row but second has value and looks like an id,
    # shift left to recover partial markdown table rows.
    if len(df.columns) >= 2:
        c0, c1 = df.columns[0], df.columns[1]
        for idx, row in df.iterrows():
            if str(row[c0]).strip() == "" and str(row[c1]).strip():
                df.at[idx, c0] = row[c1]
                df.at[idx, c1] = ""

    # Remove rows that are just decorative remnants.
    df = df[
        (df.astype(str).apply(lambda r: any(v.strip() for v in r), axis=1))
    ].copy()

    if table.heading:
        df.insert(0, "section", table.heading)

    return df


def _find_col_by_keywords(columns: Sequence[str], keywords: Sequence[str]) -> Optional[str]:
    lower = {c: c.lower() for c in columns}
    for col, low in lower.items():
        if all(k in low for k in keywords):
            return col
    for col, low in lower.items():
        if any(k in low for k in keywords):
            return col
    return None


def _looks_like_part_code(value: str) -> bool:
    text = (value or "").strip().upper()
    if not text or len(text) > 32:
        return False
    return bool(_PART_CODE_RE.match(text))


def extract_parts_dataframe(tables: Sequence[ParsedTable]) -> pd.DataFrame:
    parts_rows: List[Dict[str, Any]] = []
    for idx, table in enumerate(tables):
        df = table_to_dataframe(table)
        if df.empty:
            continue
        columns = list(df.columns)
        part_col = _find_col_by_keywords(columns, ("part", "nr"))
        desc_col = _find_col_by_keywords(columns, ("description",))

        # Fallback: first two non-section columns for catalog-like tables.
        non_section = [c for c in columns if c != "section"]
        if part_col is None and non_section:
            part_col = non_section[0]
        if desc_col is None and len(non_section) >= 2:
            desc_col = non_section[1]

        if not part_col or not desc_col or part_col == desc_col:
            continue

        for _, row in df.iterrows():
            part = _clean_cell(str(row.get(part_col, "")))
            desc = _clean_cell(str(row.get(desc_col, "")))
            if not part or not desc:
                continue
            if not _looks_like_part_code(part):
                continue
            parts_rows.append(
                {
                    "section": _clean_cell(str(row.get("section", table.heading))),
                    "part_nr": part,
                    "description": desc,
                    "source_table_index": idx,
                }
            )

    if not parts_rows:
        return pd.DataFrame(columns=["section", "part_nr", "description", "source_table_index"])
    out = pd.DataFrame(parts_rows)
    out = out.drop_duplicates(subset=["section", "part_nr", "description"]).reset_index(drop=True)
    return out


def write_parts_sqlite(parts_df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        parts_df.to_sql("parts", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_parts_part_nr ON parts(part_nr)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_parts_section ON parts(section)")
        conn.commit()
    finally:
        conn.close()


def ingest_marker_catalog(
    marker_output_dir: Path,
    *,
    run_id: str,
    artifacts_root: Path = DEFAULT_ARTIFACTS_DIR,
) -> Dict[str, Any]:
    md_path = marker_output_dir / "catalog.md"
    if not md_path.exists():
        raise FileNotFoundError(f"catalog.md not found in {marker_output_dir}")

    markdown = md_path.read_text(encoding="utf-8", errors="replace")
    tables, images = parse_markdown_tables(markdown)
    if not tables:
        raise RuntimeError("no markdown tables found in catalog.md")

    # Choose largest table as primary dataset.
    primary = max(tables, key=lambda t: t.non_empty_cells)
    df = table_to_dataframe(primary)
    if df.empty:
        raise RuntimeError("parsed primary table is empty")

    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    clean_csv = run_dir / "clean.csv"
    schema_json = run_dir / "schema.json"
    summary_json = run_dir / "ingest_summary.json"
    images_json = run_dir / "unresolved_images.json"
    table_json = run_dir / "table_inventory.json"
    parts_csv = run_dir / "parts.csv"
    parts_db = run_dir / "parts.db"

    df.to_csv(clean_csv, index=False)
    schema = infer_schema(df)
    schema_json.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    table_inventory = [
        {
            "heading": t.heading,
            "start_line": t.start_line,
            "rows": len(t.rows),
            "columns": max(len(t.headers), max((len(r) for r in t.rows), default=0)),
            "non_empty_cells": t.non_empty_cells,
        }
        for t in tables
    ]
    table_json.write_text(json.dumps(table_inventory, indent=2), encoding="utf-8")
    images_json.write_text(json.dumps(images, indent=2), encoding="utf-8")

    parts_df = extract_parts_dataframe(tables)
    parts_df.to_csv(parts_csv, index=False)
    write_parts_sqlite(parts_df, parts_db)

    summary = {
        "ts": _utc_now_iso(),
        "run_id": run_id,
        "source_dir": str(marker_output_dir),
        "source_markdown": str(md_path),
        "tables_found": len(tables),
        "primary_table": {
            "heading": primary.heading,
            "rows": len(primary.rows),
            "non_empty_cells": primary.non_empty_cells,
        },
        "rows_in_clean_csv": int(len(df)),
        "columns_in_clean_csv": int(len(df.columns)),
        "images_referenced": len(images),
        "parts_rows": int(len(parts_df)),
        "warnings": ["has_unresolved_images"] if images else [],
        "artifacts": {
            "clean_csv": str(clean_csv),
            "schema_json": str(schema_json),
            "table_inventory_json": str(table_json),
            "unresolved_images_json": str(images_json),
            "parts_csv": str(parts_csv),
            "parts_db": str(parts_db),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Marker catalog markdown slice parser.")
    p.add_argument("--input-dir", required=True, help="Folder containing catalog.md")
    p.add_argument("--run-id", default="phase05-marker-catalog")
    p.add_argument("--ask", default="", help="Optional NL query to run after ingest")
    p.add_argument(
        "--show-parts",
        action="store_true",
        help="Print extracted Part nr + Description rows after ingest.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    summary = ingest_marker_catalog(Path(args.input_dir), run_id=args.run_id)
    print(json.dumps(summary, indent=2))

    if args.show_parts:
        parts_path = Path(summary["artifacts"]["parts_csv"])
        if parts_path.exists():
            df = pd.read_csv(parts_path)
            print(df.to_string(index=False))

    if args.ask.strip():
        result = run_query_for_run(args.run_id, args.ask.strip())
        print(json.dumps({"query_result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
