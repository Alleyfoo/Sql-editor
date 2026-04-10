from __future__ import annotations

import json
from pathlib import Path

from eval.cleaning_capability_eval import main


def test_cleaning_capability_eval_mock_runs_and_writes_report(tmp_path: Path) -> None:
    report_path = tmp_path / "cleaning_report.json"
    rc = main(
        [
            "--provider",
            "mock",
            "--model",
            "mock-test",
            "--max-cases",
            "2",
            "--output",
            str(report_path),
        ]
    )
    assert rc == 0
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "mock"
    assert payload["model"] == "mock-test"
    assert payload["cases_total"] == 2
    assert payload["metrics"]["case_pass_rate"] >= 0.0

