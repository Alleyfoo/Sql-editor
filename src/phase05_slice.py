"""Phase 0.5 vertical slice: PDF -> table -> clean/schema -> query -> JSON.

This module keeps the implementation intentionally narrow:
- ingest: extract one table from PDF and emit clean artifacts
- query: run existing NL->JSON->SQL path against clean.csv
- serve: thin local HTTP endpoint for querying produced runs
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pdfplumber

from .config import load_config
from .executor import ExecutionError, execute
from .ingestion import infer_schema, load_csv
from .llm.natural_language import LLMError, load_llm_config, nl_to_query_model


DEFAULT_INPUTS_DIR = Path("data") / "pdf_inputs"
DEFAULT_ARTIFACTS_DIR = Path("artifacts") / "phase05"


@dataclass
class IngestArtifacts:
    run_id: str
    pdf_path: str
    run_dir: str
    raw_table_csv: str
    clean_csv: str
    schema_json: str
    ingest_summary_json: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_header(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[%$€£¥]", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _dedupe_headers(headers: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for raw in headers:
        h = _norm_header(raw)
        if not h:
            continue
        n = seen.get(h, 0) + 1
        seen[h] = n
        out.append(h if n == 1 else f"{h}_{n}")
    return out


def _numeric_like_ratio(cells: List[str]) -> float:
    if not cells:
        return 0.0
    pat = re.compile(r"^\s*[-+]?\d+(\.\d+)?\s*$")
    count = sum(1 for c in cells if pat.match(c))
    return count / len(cells)


def _alpha_like_ratio(cells: List[str]) -> float:
    if not cells:
        return 0.0
    count = sum(1 for c in cells if any(ch.isalpha() for ch in c))
    return count / len(cells)


def detect_header_row(rows: List[List[str]]) -> Tuple[int, float]:
    """Heuristic header-row detection with confidence score 0..1."""
    best_idx = 0
    best_score = float("-inf")
    limit = min(len(rows), 20)
    for idx in range(limit):
        row = rows[idx]
        cells = [c.strip() for c in row if c and c.strip()]
        if len(cells) < 2:
            continue
        alpha = _alpha_like_ratio(cells)
        numeric = _numeric_like_ratio(cells)
        score = (len(cells) * 0.8) + (alpha * 2.0) - (numeric * 1.6) - (idx * 0.03)
        if score > best_score:
            best_idx = idx
            best_score = score

    confidence = max(0.0, min(1.0, (best_score + 1.5) / 6.0))
    return best_idx, confidence


def _coerce_rows(rows: List[List[Any]]) -> List[List[str]]:
    width = max((len(r) for r in rows), default=0)
    out: List[List[str]] = []
    for row in rows:
        current = []
        for i in range(width):
            cell = row[i] if i < len(row) else ""
            current.append("" if cell is None else str(cell).strip())
        out.append(current)
    return out


def _extract_text_grid(page: pdfplumber.page.Page) -> List[List[str]]:
    """Fallback extractor for PDFs where table extraction fails.

    Splits text lines on common column separators:
    - 2+ spaces
    - tab
    - pipe
    - comma
    """
    text = page.extract_text() or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows: List[List[str]] = []
    splitter = re.compile(r"\s{2,}|\t|\s\|\s|,")
    for line in lines:
        parts = [p.strip() for p in splitter.split(line) if p.strip()]
        if len(parts) >= 2:
            rows.append(parts)
    return _coerce_rows(rows) if rows else []


def _pick_best_table(
    pdf_path: Path,
    *,
    page_indices: Optional[Sequence[int]] = None,
    target_page_1based: int = 0,
    target_table_1based: int = 0,
    expected_columns: Optional[Sequence[str]] = None,
) -> Tuple[int, int, List[List[str]], int]:
    best_rows: List[List[str]] = []
    best_page = -1
    best_table_idx = -1
    best_score = -1
    scanned_pages = 0
    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
        if target_page_1based > 0:
            page_zero = target_page_1based - 1
            candidate_pages = [page_zero] if 0 <= page_zero < total_pages else []
        elif page_indices:
            candidate_pages = [i for i in page_indices if 0 <= i < total_pages]
        else:
            candidate_pages = list(range(total_pages))

        expected_norm = {_norm_header(c) for c in (expected_columns or []) if c}

        def _table_header_overlap(rows: List[List[str]]) -> int:
            if not expected_norm or not rows:
                return 0
            header_sample = rows[: min(3, len(rows))]
            tokens = set()
            for row in header_sample:
                for cell in row:
                    norm = _norm_header(cell)
                    if norm:
                        tokens.add(norm)
            return len(tokens & expected_norm)

        for page_idx in candidate_pages:
            page = pdf.pages[page_idx]
            scanned_pages += 1
            extracted = page.extract_tables() or []
            for table_idx, table in enumerate(extracted):
                if not table:
                    continue
                rows = _coerce_rows(table)
                height = len(rows)
                width = max((len(r) for r in rows), default=0)
                if height < 2 or width < 2:
                    continue
                non_empty = sum(1 for r in rows for c in r if c)
                score = non_empty + (height * width // 3)
                score += _table_header_overlap(rows) * 200
                if target_table_1based > 0 and (table_idx + 1) == target_table_1based:
                    score += 10_000
                if score > best_score:
                    best_score = score
                    best_page = page_idx
                    best_table_idx = table_idx
                    best_rows = rows
            # Fallback: derive a grid from text lines when no table was found.
            if not extracted:
                rows = _extract_text_grid(page)
                height = len(rows)
                width = max((len(r) for r in rows), default=0)
                if height >= 2 and width >= 2:
                    non_empty = sum(1 for r in rows for c in r if c)
                    score = non_empty + (height * width // 4)
                    score += _table_header_overlap(rows) * 200
                    if score > best_score:
                        best_score = score
                        best_page = page_idx
                        best_table_idx = -1
                        best_rows = rows
    if not best_rows:
        raise RuntimeError(f"no tabular data found in PDF: {pdf_path}")
    return best_page, best_table_idx, best_rows, scanned_pages


def _sample_page_indices(total_pages: int, sample_size: int, seed: int) -> List[int]:
    if sample_size <= 0 or sample_size >= total_pages:
        return list(range(total_pages))
    rng = random.Random(seed)
    return sorted(rng.sample(list(range(total_pages)), sample_size))


def _build_clean_dataframe(rows: List[List[str]], header_row_idx: int) -> pd.DataFrame:
    picked_idx = header_row_idx
    non_empty_cols: List[int] = []
    headers: List[str] = []

    # Try detected row and a few following rows for usable header names.
    scan_start = max(0, header_row_idx - 3)
    scan_end = min(len(rows), header_row_idx + 5)
    for idx in range(scan_start, scan_end):
        candidate = rows[idx]
        candidate_cols: List[int] = []
        for i, cell in enumerate(candidate):
            has_header = bool(cell.strip())
            has_future_data = any(
                (r[i].strip() if i < len(r) else "")
                for r in rows[idx + 1 : min(len(rows), idx + 8)]
            )
            if has_header or has_future_data:
                candidate_cols.append(i)
        candidate_headers = _dedupe_headers(
            [candidate[i] if candidate[i].strip() else f"label_{i+1}" for i in candidate_cols]
        )
        if len(candidate_headers) >= 2:
            picked_idx = idx
            non_empty_cols = candidate_cols
            headers = candidate_headers
            break

    # Last resort: synthetic headers from non-empty column positions.
    if not headers:
        fallback_cols = [i for i, cell in enumerate(rows[header_row_idx]) if cell.strip()]
        if not fallback_cols:
            fallback_cols = list(range(len(rows[header_row_idx])))
        non_empty_cols = fallback_cols
        headers = [f"col_{i+1}" for i in range(len(non_empty_cols))]

    records: List[List[str]] = []
    for row in rows[picked_idx + 1 :]:
        values = [row[i].strip() if i < len(row) else "" for i in non_empty_cols]
        if any(v != "" for v in values):
            records.append(values[: len(headers)])
    df = pd.DataFrame(records, columns=headers)
    df = _cleanup_sparse_columns(df)

    def _data_like_row_count(frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        count = 0
        for _, r in frame.iterrows():
            cells = [str(v).strip() for v in r.tolist() if str(v).strip()]
            if not cells:
                continue
            if _numeric_like_ratio(cells) >= 0.2:
                count += 1
                continue
            if any(any(ch.isdigit() for ch in c) for c in cells):
                count += 1
        return count

    # If the basic strategy produced very little data, retry with multi-row
    # header flattening to handle sparse/merged PDF table exports.
    if len(df) < 2 or _data_like_row_count(df) == 0:
        fallback = _build_from_multirow_headers(rows)
        if len(fallback) > len(df) or _data_like_row_count(fallback) > _data_like_row_count(df):
            return fallback
    return df


def _build_from_multirow_headers(rows: List[List[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    width = max((len(r) for r in rows), default=0)
    padded = []
    for row in rows:
        padded.append([row[i].strip() if i < len(row) else "" for i in range(width)])

    keep_cols = [i for i in range(width) if any(r[i] for r in padded)]
    if not keep_cols:
        return pd.DataFrame()
    compact = [[r[i] for i in keep_cols] for r in padded]

    def _is_data_row(cells: List[str]) -> bool:
        non_empty = [c for c in cells if c]
        if len(non_empty) < 2:
            return False
        num = _numeric_like_ratio(non_empty)
        alpha = _alpha_like_ratio(non_empty)
        return num >= 0.25 and alpha >= 0.25

    data_start = 1
    for idx, row in enumerate(compact[: min(25, len(compact))]):
        if _is_data_row(row):
            data_start = idx
            break

    header_rows = compact[:data_start]
    if not header_rows:
        header_rows = [compact[0]]
        data_start = 1

    raw_headers: List[str] = []
    for col_idx in range(len(compact[0])):
        parts: List[str] = []
        for hrow in header_rows:
            cell = hrow[col_idx].strip()
            if cell and cell not in parts:
                parts.append(cell)
        if not parts:
            parts = ["label" if col_idx == 0 else f"col_{col_idx+1}"]
        raw_headers.append(" ".join(parts))

    headers = _dedupe_headers(raw_headers)
    records: List[List[str]] = []
    for row in compact[data_start:]:
        if any(c for c in row):
            records.append(row[: len(headers)])
    df = pd.DataFrame(records, columns=headers)
    return _cleanup_sparse_columns(df)


def _cleanup_sparse_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()

    def _non_empty_count(col: pd.Series) -> int:
        return int((col.astype(str).str.strip() != "").sum())

    cols = list(work.columns)
    drops: set[str] = set()

    # Move values from adjacent generic columns into empty semantic columns.
    for i, col in enumerate(cols):
        col_count = _non_empty_count(work[col])
        if col_count != 0 or col.startswith("col_"):
            continue
        neighbors: List[str] = []
        if i - 1 >= 0:
            neighbors.append(cols[i - 1])
        if i + 1 < len(cols):
            neighbors.append(cols[i + 1])
        for neighbor in neighbors:
            if neighbor in work.columns and neighbor.startswith("col_") and _non_empty_count(work[neighbor]) > 0:
                # Coalesce into semantic column and drop synthetic source.
                work[col] = work[neighbor]
                drops.add(neighbor)
                break

    if drops:
        work = work.drop(columns=[c for c in drops if c in work.columns], errors="ignore")

    empty_generic = [
        c
        for c in list(work.columns)
        if c.startswith("col_") and _non_empty_count(work[c]) == 0
    ]
    if empty_generic:
        work = work.drop(columns=empty_generic, errors="ignore")
    return work


def ingest_pdf(
    pdf_path: Path,
    *,
    run_id: Optional[str] = None,
    artifacts_root: Path = DEFAULT_ARTIFACTS_DIR,
    sample_pages: int = 0,
    page_seed: int = 42,
    target_page: int = 0,
    target_table: int = 0,
    expected_columns: Optional[Sequence[str]] = None,
) -> IngestArtifacts:
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    run = run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = artifacts_root / run
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with pdfplumber.open(str(pdf_path)) as _pdf:
        total_pages = len(_pdf.pages)
    selected_pages = _sample_page_indices(total_pages, sample_pages, page_seed)

    page_idx, table_idx, rows, scanned_pages = _pick_best_table(
        pdf_path,
        page_indices=selected_pages,
        target_page_1based=target_page,
        target_table_1based=target_table,
        expected_columns=expected_columns,
    )
    extract_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    header_row_idx, confidence = detect_header_row(rows)
    clean_df = _build_clean_dataframe(rows, header_row_idx)
    clean_ms = (time.perf_counter() - t1) * 1000.0

    raw_df = pd.DataFrame(rows)
    raw_csv = run_dir / "raw_table.csv"
    raw_df.to_csv(raw_csv, index=False, header=False)

    clean_csv = run_dir / "clean.csv"
    clean_df.to_csv(clean_csv, index=False)

    schema = infer_schema(clean_df)
    schema_json = run_dir / "schema.json"
    schema_json.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    warnings: List[str] = []
    if confidence < 0.6:
        warnings.append("low_header_confidence")
    if len(clean_df.columns) < 2:
        warnings.append("few_columns_detected")
    if len(clean_df) == 0:
        warnings.append("no_data_rows_after_header")

    summary = {
        "ts": _utc_now_iso(),
        "run_id": run,
        "pdf_path": str(pdf_path),
        "table_selected": {"page_index": page_idx, "table_index": table_idx},
        "selection_hints": {
            "target_page": target_page,
            "target_table": target_table,
            "expected_columns": list(expected_columns or []),
        },
        "page_sampling": {
            "total_pages": total_pages,
            "scanned_pages": scanned_pages,
            "sample_pages": sample_pages if sample_pages > 0 else total_pages,
            "seed": page_seed,
            "indices": selected_pages,
        },
        "header_row_index": header_row_idx,
        "header_confidence": round(confidence, 4),
        "row_count": int(len(clean_df)),
        "column_count": int(len(clean_df.columns)),
        "warnings": warnings,
        "timing_ms": {
            "extract_ms": round(extract_ms, 3),
            "clean_ms": round(clean_ms, 3),
            "end_to_end_ms": round(extract_ms + clean_ms, 3),
        },
    }
    summary_json = run_dir / "ingest_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return IngestArtifacts(
        run_id=run,
        pdf_path=str(pdf_path),
        run_dir=str(run_dir),
        raw_table_csv=str(raw_csv),
        clean_csv=str(clean_csv),
        schema_json=str(schema_json),
        ingest_summary_json=str(summary_json),
    )


def run_query_for_run(
    run_id: str,
    question: str,
    *,
    artifacts_root: Path = DEFAULT_ARTIFACTS_DIR,
) -> Dict[str, Any]:
    run_dir = artifacts_root / run_id
    clean_csv = run_dir / "clean.csv"
    if not clean_csv.exists():
        raise FileNotFoundError(f"clean.csv not found for run_id={run_id!r}")

    config = load_config()
    llm_config = load_llm_config(config)

    t0 = time.perf_counter()
    conn, schema = load_csv(clean_csv)
    load_ms = (time.perf_counter() - t0) * 1000.0

    try:
        t1 = time.perf_counter()
        model = nl_to_query_model(question, schema, config=llm_config)
        plan_ms = (time.perf_counter() - t1) * 1000.0

        sql = model.to_sql()

        t2 = time.perf_counter()
        result_df = execute(conn, sql)
        exec_ms = (time.perf_counter() - t2) * 1000.0
    finally:
        conn.close()

    query_plan_json = run_dir / "query_plan.json"
    query_plan_json.write_text(json.dumps(asdict(model), indent=2), encoding="utf-8")

    query_sql_path = run_dir / "query.sql"
    query_sql_path.write_text(sql + "\n", encoding="utf-8")

    result_csv = run_dir / "query_result.csv"
    result_df.to_csv(result_csv, index=False)

    summary = {
        "ts": _utc_now_iso(),
        "run_id": run_id,
        "question": question,
        "reply": model.reply,
        "rows": int(len(result_df)),
        "warnings": [],
        "timing_ms": {
            "load_ms": round(load_ms, 3),
            "plan_ms": round(plan_ms, 3),
            "exec_ms": round(exec_ms, 3),
            "end_to_end_ms": round(load_ms + plan_ms + exec_ms, 3),
        },
    }
    query_summary_json = run_dir / "query_summary.json"
    query_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "question": question,
        "reply": model.reply,
        "sql": sql,
        "rows": result_df.to_dict(orient="records"),
        "warnings": [],
        "artifacts": {
            "query_plan_json": str(query_plan_json),
            "query_sql": str(query_sql_path),
            "query_result_csv": str(result_csv),
            "query_summary_json": str(query_summary_json),
        },
        "timing_ms": summary["timing_ms"],
    }


class _SliceHandler(BaseHTTPRequestHandler):
    """Simple local endpoint for querying a prepared Phase 0.5 run."""

    artifacts_root: Path = DEFAULT_ARTIFACTS_DIR

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler signature
        if self.path != "/query":
            self._write_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw) if raw else {}
            run_id = str(payload.get("run_id") or "").strip()
            ask = str(payload.get("ask") or "").strip()
            if not run_id or not ask:
                self._write_json(400, {"error": "run_id and ask are required"})
                return
            result = run_query_for_run(run_id, ask, artifacts_root=self.artifacts_root)
            self._write_json(200, result)
        except (FileNotFoundError, LLMError, ExecutionError, RuntimeError, ValueError) as exc:
            self._write_json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            self._write_json(500, {"error": f"internal_error: {exc}"})


def _cmd_ingest(args: argparse.Namespace) -> int:
    pdf_path = Path(args.pdf)
    out = ingest_pdf(
        pdf_path,
        run_id=args.run_id,
        sample_pages=max(0, int(args.sample_pages)),
        page_seed=int(args.page_seed),
        target_page=max(0, int(args.target_page)),
        target_table=max(0, int(args.target_table)),
        expected_columns=[c.strip() for c in (args.expected_columns or "").split(",") if c.strip()],
    )
    print(json.dumps(asdict(out), indent=2))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    result = run_query_for_run(args.run_id, args.ask)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    _SliceHandler.artifacts_root = DEFAULT_ARTIFACTS_DIR
    server = ThreadingHTTPServer((args.host, args.port), _SliceHandler)
    print(
        json.dumps(
            {
                "mode": "serve",
                "host": args.host,
                "port": args.port,
                "endpoint": "POST /query",
                "artifacts_root": str(DEFAULT_ARTIFACTS_DIR),
            },
            indent=2,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 0.5 PDF vertical-slice utility.")
    sp = p.add_subparsers(dest="command", required=True)

    ingest = sp.add_parser("ingest", help="Extract one table from a PDF into clean artifacts.")
    ingest.add_argument("--pdf", required=True, help="Path to PDF file.")
    ingest.add_argument("--run-id", default="", help="Optional run ID.")
    ingest.add_argument(
        "--sample-pages",
        type=int,
        default=0,
        help="Randomly sample this many pages (0 = scan all pages).",
    )
    ingest.add_argument(
        "--page-seed",
        type=int,
        default=42,
        help="Random seed for page sampling.",
    )
    ingest.add_argument(
        "--target-page",
        type=int,
        default=0,
        help="Prefer this 1-based page number (0 = no hint).",
    )
    ingest.add_argument(
        "--target-table",
        type=int,
        default=0,
        help="Prefer this 1-based table index on target page (0 = no hint).",
    )
    ingest.add_argument(
        "--expected-columns",
        default="",
        help="Comma-separated expected column names used as selection hint.",
    )
    ingest.set_defaults(func=_cmd_ingest)

    query = sp.add_parser("query", help="Run NL query against a prepared run_id.")
    query.add_argument("--run-id", required=True)
    query.add_argument("--ask", required=True)
    query.set_defaults(func=_cmd_query)

    serve = sp.add_parser("serve", help="Start local HTTP endpoint for POST /query.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.set_defaults(func=_cmd_serve)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


__all__ = [
    "DEFAULT_INPUTS_DIR",
    "DEFAULT_ARTIFACTS_DIR",
    "IngestArtifacts",
    "detect_header_row",
    "ingest_pdf",
    "run_query_for_run",
    "main",
]
