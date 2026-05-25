"""YAML configuration loader.

Thin wrapper around ``yaml.safe_load`` that returns an empty dict if the
file is missing. The loading pattern (empty-on-missing, safe_load only,
dict-or-empty coercion) is adapted from Alleyfoo/Data-tool-demo — see
``vendor/data-tool-demo/config_loader.py`` for the vendored original.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


# Resolve relative to the repo root (two parents up from src/config.py)
# so the path is stable regardless of the working directory Streamlit
# is launched from.
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load a YAML config file. Returns ``{}`` if the file is missing."""
    p = Path(path)
    if not p.exists():
        return {}
    payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


__all__ = ["load_config", "DEFAULT_CONFIG_PATH"]
