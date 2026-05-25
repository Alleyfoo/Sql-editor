from __future__ import annotations

from pathlib import Path

from eval.helsinki_yearbook_pack_tools import build_and_write, classify_case, load_pack, split_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = REPO_ROOT / "data" / "open_data" / "helsinki_yearbook_2024_analysis_test_set.json"


def test_helsinki_pack_split_counts() -> None:
    payload = load_pack(PACK_PATH)
    split = split_cases(payload["cases"])
    total = len(split.easy) + len(split.mid) + len(split.hard)
    assert total == 35
    assert len(split.easy) > 0
    assert len(split.mid) > 0
    assert len(split.hard) > 0


def test_helsinki_pack_classify_examples() -> None:
    easy_case = {
        "category": "lookup",
        "output_type": "scalar",
        "validator": "single_numeric_scalar_exact",
    }
    assert classify_case(easy_case) == "easy"

    mid_case = {
        "category": "ranking",
        "output_type": "ranked_list",
        "validator": "ranked_list_tolerance",
    }
    assert classify_case(mid_case) == "mid"

    hard_case = {
        "category": "ratio",
        "output_type": "scalar",
        "validator": "single_numeric_scalar_tolerance",
    }
    assert classify_case(hard_case) == "hard"


def test_helsinki_pack_write_outputs(tmp_path: Path) -> None:
    counts = build_and_write(PACK_PATH, tmp_path)
    assert counts["total"] == 35
    assert (tmp_path / "helsinki_yearbook_2024_analysis_easy.json").exists()
    assert (tmp_path / "helsinki_yearbook_2024_analysis_mid.json").exists()
    assert (tmp_path / "helsinki_yearbook_2024_analysis_hard.json").exists()
    assert (tmp_path / "helsinki_yearbook_2024_analysis_split_summary.md").exists()
