"""Shared recursive float-rounding for JSON API responses -- keeps payloads
from serializing artifacts like 0.15900000000000003 / 60.30000000000001,
which both waste bytes and look broken in a downloaded report or a
network-tab inspection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_DECIMALS = 4


def round_floats(value: Any, decimals: int = DEFAULT_DECIMALS) -> Any:
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return round(value, decimals)
    if isinstance(value, dict):
        return {key: round_floats(item, decimals) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(item, decimals) for item in value]
    return value
