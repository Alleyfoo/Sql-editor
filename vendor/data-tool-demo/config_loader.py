# Vendored from Alleyfoo/Data-tool-demo @ dae9eeb14c8780ff86449895425fdc9e6c22ab61
# Path in source: src/core/config_loader.py
# License: MIT (Copyright (c) 2025 Alleyfoo)
#
# Kept here as reference for the YAML-loading pattern reused in
# ``src/config.py``. Not imported directly — ``src/config.py`` implements
# a slimmed-down generic version adapted for this project's needs.

"""Load UI configuration (synonyms) for Streamlit mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def load_synonyms(
    base_path: Path | None = None, user_path: Path | None = None
) -> Dict[str, List[str]]:
    base = base_path or Path("src/config.yaml")
    user = user_path or Path("src/config.user.yaml")

    merged: Dict[str, List[str]] = {}
    for path in (base, user):
        payload = _load_yaml(path)
        syns = payload.get("synonyms", {}) if isinstance(payload, dict) else {}
        if not isinstance(syns, dict):
            continue
        for key, values in syns.items():
            if not isinstance(values, list):
                continue
            items = [str(v).strip() for v in values if v not in (None, "")]
            if not items:
                continue
            merged.setdefault(str(key), [])
            merged[key].extend(items)

    # Deduplicate while preserving order
    for key, values in merged.items():
        seen = set()
        deduped = []
        for item in values:
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(item)
        merged[key] = deduped

    return merged


__all__ = ["load_synonyms"]
