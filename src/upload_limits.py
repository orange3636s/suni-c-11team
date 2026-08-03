"""Shared upload-size/row-count limits, loaded from config/upload_limits.yaml
rather than hardcoded, so api/settings.py (the /api/train upload path) and
src/runtime/datasets.py (the /api/datasets upload path) enforce the exact
same limit without either importing the other.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "upload_limits.yaml"
DEFAULT_LIMITS = {"max_upload_size_mb": 15, "max_row_count": 50000}


@lru_cache(maxsize=1)
def _limits() -> dict[str, int]:
    try:
        loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except OSError:
        return dict(DEFAULT_LIMITS)
    return {**DEFAULT_LIMITS, **loaded}


def max_upload_size_mb() -> int:
    return int(_limits()["max_upload_size_mb"])


def max_upload_size_bytes() -> int:
    return max_upload_size_mb() * 1024 * 1024


def max_row_count() -> int:
    return int(_limits()["max_row_count"])
